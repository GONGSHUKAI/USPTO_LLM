#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os 
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
import json
from random import shuffle
import re
import tiktoken
import numpy as np
from typing import Dict, List, Set, Tuple, Union
import copy
import math
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

def make_mol(s: str, 
             keep_h: bool = False, 
             add_h: bool = False, 
             keep_atom_map: bool = False):
    """
    Builds an RDKit molecule from a SMILES string.
    
    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map and mol is not None:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)
    elif not keep_atom_map and mol is not None:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    return mol


def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h = False, add_h = False, keep_atom_map = False)
    if isinstance(mol, tuple):
        mol = copy.deepcopy(mol[0])
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = mol, includeChirality = include_chirality)

    return scaffold

def scaffold_to_index(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]]) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(list)
    for i, mol in tqdm(enumerate(mols), total = len(mols)):
        try:
            scaffold = generate_scaffold(mol)
            # 尝试能否转换成分子，转换失败的去掉
            scaffold_mol = make_mol(scaffold)
            if scaffold_mol is None:
                continue
        except:
            continue

        scaffolds[scaffold].append(i)

    return scaffolds

def mol_to_index(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]]) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    mol_list = defaultdict(list)
    for i, mol in tqdm(enumerate(mols), total = len(mols)):
        try:
            mol = make_mol(mol)
            # 尝试能否转换成分子，转换失败的去掉
            if mol is None:
                continue
            mol_smiles = Chem.MolToSmiles(mol)
        except:
            continue

        mol_list[mol_smiles].append(i)

    return mol_list

def date_to_index(key_dates: Union[List[str], List[int]]) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    dates = defaultdict(list)
    for i, date in tqdm(enumerate(key_dates), total = len(key_dates)):
        dates[date].append(i)

    return dates

def draw_mols(mols_list, hit_ats_list=None, subtitle=None, save_fig=None):
    plt.figure(figsize=(8, 8))
    if isinstance(mols_list[0], str):
        mols_list = [Chem.MolFromSmiles(smiles) for smiles in mols_list]
    if hit_ats_list is None:
        hit_ats_list = [[]] * len(mols_list)
    num_rows = math.ceil(math.sqrt(len(mols_list)))
    num_cols = math.ceil(1.0 * len(mols_list) / num_rows)
    for i, (mol, hit_ats) in enumerate(zip(mols_list, hit_ats_list)):
        img = Draw.MolToImage(mol, highlightAtoms=hit_ats)
        ax = plt.subplot(num_rows, num_cols, i + 1)
        if subtitle:
            ax.set_title(subtitle[i])
        plt.imshow(img)
    if save_fig:
        plt.savefig(save_fig, dpi=200)
    plt.show()

def get_xmlroot(path):
    # print(path)
    # 去掉URL,也就是namespace
    it = ET.iterparse(path)
    for _, el in it:
        _, _, el.tag = el.tag.rpartition('}') # strip ns
    return it.root

def get_filelist(path):
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            filelist.append(os.path.join(home, filename))
    return filelist

def get_date(path):
    filename = os.path.basename(path)
    pattern = r'[0-9]{8}'
    return re.findall(pattern, filename)[0]

def get_reaction(path):
    date = get_date(path)
    root = get_xmlroot(path)
    
    def get_mol_list(xml_list):
        mol_list = []
        for mol in xml_list:
            mol_dict = {'name':None, 'smiles':None,}

            if mol.find('molecule/name') is not None:
                mol_dict['name'] = mol.find('molecule/name').text
            if mol.find('molecule/nameResolved') is not None:
                mol_dict['name'] = mol.find('molecule/nameResolved').text
            
            for identifier in mol.findall('identifier'):
                if 'smiles' in identifier.attrib['dictRef'] or 'cml:smiles' in identifier.attrib.values():  
                    mol_dict['smiles'] = identifier.attrib['value']
                                       
            mol_list.append(mol_dict)
        
        # 将name长的放在前面,防止mask的时候短name的嵌套在长name里
        mol_list = sorted(mol_list, key=lambda x: len(x['name']), reverse=True)
        return mol_list
    
    all_reaction_list = []
    for node in root.findall('reaction'):
        reaction_dict = dict()
        reaction_dict['date'] = date
        reaction_dict['documentId'] = node.find('source/documentId').text
        if node.find('source/paragraphNum') is not None:
            reaction_dict['paragraphNum'] = node.find('source/paragraphNum').text
        else:
            reaction_dict['paragraphNum'] = None
        reaction_dict['paragraphText'] = node.find('source/paragraphText').text
        reaction_dict['reactantList'] = get_mol_list(node.findall('reactantList/reactant'))
        reaction_dict['spectatorList'] = get_mol_list(node.findall('spectatorList/spectator'))
        reaction_dict['productList'] = get_mol_list(node.findall('productList/product'))       

        all_reaction_list.append(reaction_dict)
        
    return all_reaction_list

def get_all_reaction(
        process_file='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Larrea/raw/uspto_full.json',
        rxn_idx_file='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Larrea/raw/rxn_idx.json'
    ):
    if os.path.exists(process_file) and os.path.exists(rxn_idx_file):
        with open(process_file,'r') as f:
            all_reaction_list = json.load(f)
        with open(rxn_idx_file,'r') as f:
            rxn_idx = json.load(f)['rxn_idx']
    else:
        raise ValueError
    
    print('len(all_reaction_list)')
    print(len(all_reaction_list))
    return all_reaction_list, rxn_idx

def write_rxn_idx(
        rxn_idx, 
        rxn_idx_file='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Larrea/raw/rxn_idx.json',
    ):
    with open(rxn_idx_file,'w') as f:
        json.dump({"rxn_idx":rxn_idx}, f)

def num_tokens_from_messages(message, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    return len(encoding.encode(message))


if __name__=='__main__':
    message = "You are now working as an excellent expert in chemistry and molecule discovery.  You are called SynthesisGPT. Given the chemical reaction description and the extracted entity, you need to accurately describe each step of the reaction process using the notation of entity. \n\nThe chemical reactions have multiple steps. Do not use one line to describe the reaction. Change the line for each substep. Each substep is represented as \u2018[reactants] > [reaction conditions] > [products]\u2019. We call this format \u201cHeterogeneous Graph\u201d. The notations of entities are as follows:\n(1) Reactants include \u2018Rx\u2019 and \u2018Mx\u2019. \u2018Rx\u2019 represents reactants, \u2018Mx\u2019 represents the mixture \u2018x\u2019 with uncertain substances. \n(2) Reaction conditions include \u2018Sx\u2019, \u2018Cx\u2019, \u2018Ex\u2019, and \u2018Tx\u2019. \n(3) Products include symbols \u2018Mx\u2019 and \u2018Px\u2019. \n\nThe numbers after entities show their position(indices) in the reaction description, which only helps to identify the entities in the reaction description.\n\nAlso, you need to know that\n(1) In each substep, [there are at most two \">\"], separating reactants, reaction conditions, and products! YOU MUST STRICTLY FOLLOW THIS FORMAT: Stuff before the first '>' is reactants \u201cRx\u201d; Stuff in between \"> >\" is reaction conditions \u2018Sx\u2019, \u2018Cx\u2019, \u2018Ex\u2019, and \u2018Tx\u2019; Stuff behind the second \">\" are products \u201cPx\u201d, \u201cMx\u201d. \n\n(2)A standard format is \u201cRx.Ry > Sx.Cx.Ex.Tx > Mx. If there is no reaction condition. There is no need to fill anything in between \u201c> >\u201d. Do not include Yields (Y1) into the reaction substeps because the reaction substeps always end with a \u201cP1\u201d. \n\n(3) Each reaction substep can have at most one occurrence of \u2018Ex\u2019 and \u2018Tx\u2019. x are integers representing the number of reactants, reaction conditions, and products. You CANNOT reuse the reaction conditions and reactants in different substeps!\n\n(4) The entities of reactants, reactant conditions (solvent, catalyst, time, temperature), and products are already listed after the total chemical reaction description. You need to extract the logic of chemical reactants to generate the Heterogeneous Graph correctly.\n\n(5) Solvents (Sx), temperature conditions (Ex), and time conditions (Tx) associated with postprocessing procedures like filtration, crystallization, distilling, drying, extraction, washing with solvents, and purification should not be included in the heterogeneous graph.\n\nNow you may learn from the following examples about how the Heterogeneous Graph fits the reaction logic. Pay attention to the labels under training examples: \n\nTraining Example 1\n[Typical example of Multistep Reaction. You should identify the breakpoints between substeps]\n\n[Input]\nReaction 20141222-US08902305B2-0231 description:\nTo a N,N-dimethylformamide (10 mL) suspension of sodium hydride (97%, 0.784 g, 32.7 mmol) was added methyl 2-oxoindoline-5-carboxylate (2.34 g, 12.3 mmol). The formed mixture was stirred for 10 min at room temperature followed by the addition of 4-[(6-chloro-1-oxidopyridin-3-yl)methyl]morpholine (1.87 g, 8.2 mmol). The resulting reaction mixture was set under N2 atmosphere and stirred for 1 h at 135\u00b0 C. The N,N-dimethylformamide solution was diluted with saturated aqueous sodium hydrogen carbonate (30 mL) and extracted with chloroform, and ethyl acetate (containing 5% methanol). The combined organic phases were concentrated in vacuo. The remaining N,N-dimethylformamide was removed by co-evaporation with toluene. The residue was dissolved in ethyl acetate/chloroform, (150 mL, 2:1), and phosphorus trichloride (4.5 g, 33 mmol) was added. The reaction mixture was stirred for 1 h at 60\u00b0 C., and then cooled to room temperature. The mixture was poured into a saturated aqueous sodium hydrogen carbonate solution followed by extraction of the aqueous phase with chloroform (4\u00d7). The combined organic extracts were concentrated in vacuo, and the residue was purified on a silica gel column using chloroform/methanol, (10:1), as the eluent to afford 1.05 g (35% yield) of the title compound as a yellow-brown solid: 1H NMR (DMSO-d6, 400 MHz) \u03b4 10.83 (br s, 1H), 8.11 (s, 1H), 8.04 (s, 1H), 7.91 (d, J=8.0 Hz, 1H), 7.63 (t, J=8.0 Hz, 2H), 7.00 (d, J=8.0 Hz, 1H), 3.87 (s, 3H), 3.62 (br s, 4H), 3.41 (s, 2H), 2.42 (br s, 4H); MS (EI) m/z 368 (M++1).\n\nReactant:\nR1:sodium hydride,49;\nR2:methyl 2-oxoindoline-5-carboxylate,100;\nR3:4-[(6-chloro-1-oxidopyridin-3-yl)methyl]morpholine,246;\nR4:phosphorus trichloride,796;\n\nProduct:\nP1:title compound,1280;\n\nSolvent:\nS1:N,N-dimethylformamide,5;\nS2:N,N-dimethylformamide,411;\nS3:sodium hydrogen carbonate,477;\nS4:N,N-dimethylformamide,656;\nS5:sodium hydrogen carbonate,984;\n\nCatalyst:\nNone\n\nTime:\nT1:10 min,191;\nT2:1 h,392;\nT3:1 h,884;\n\nTemperature:\nE1:room temperature,201;\nE2:135\u00b0 C,399;\nE3:60\u00b0 C.,891;\nE4:room temperature,918;\n\nYield:\nY1:35% yield,1262;\n\n[Output]:\nR1.R2>S1.E1.T1>M1\nM1.R3>E2.T2>M2\nM2.R4>E3.T3>P1\n\nTraining Example 2\n[Many post-process reaction procedures should be excluded from the heterogeneous graph.]\n\n[Input]\nReaction 20120214-US08114886B2-0551 description:\n5-[[(4-Chloro-3-methylphenyl)sulfonyl](2,5-difluorophenyl) methyl]-4-methylpyridine-2-carboxylic acid (300 mg, 0.66 mmol), 2-aminoethanol (60 \u03bcl, 0.99 mmol), 1-ethyl-3-(3-dimethylaminopropyl)carbodiimide hydrochloride (191 mg, 0.99 mmol), 1-hydroxybenzotriazole (89 mg, 0.66 mmol), and triethylamine (275 \u03bcl, 1.98 mmol) were dissolved in methylene chloride (60 ml), and the resulting mixture was stirred overnight at room temperature. Water was added to the reaction mixture, and the mixture was extracted twice with methylene chloride. The combined organic layer was dried over anhydrous sodium sulfate and filtered, and then the filtrate was concentrated under reduced pressure. The resulting residue was purified by preparative thin-layer chromatography (developed with 5% methanol/methylene chloride, eluted with 30% methanol/methylene chloride), to obtain the title compound (190 mg, 0.38 mmol, 58%) as a white amorphous substance.\n\nREACTANTs:\nR1,5-[[(4-Chloro-3-methylphenyl)sulfonyl](2,5-difluorophenyl) methyl]-4-methylpyridine-2-carboxylic acid,0;\nR2,1-ethyl-3-(3-dimethylaminopropyl)carbodiimide hydrochloride,158;\nR3,1-hydroxybenzotriazole,239;\nR4,2-aminoethanol,123;\nR5,triethylamine,286;\n\n\nPRODUCTs:\nP1,title compound,865;\n\n\nSOLVENTs:\nS1,methylene chloride,338;\nS2,methylene chloride,517;\nS3,methylene chloride,785;\nS4,methylene chloride,830;\nS5,Water,435;\n\n\nCATALYSTs:\nNone\n\nTIMEs:\nT1,overnight,404;\n\n\nTEMPERATUREs:\nE1,room temperature,417;\n\n\nYIELD:\nY1,58%,900;\n\n[Output]\nR1.R2.R3.R4.R5>S1.E1.T1>P1\n\n\nTraining Example 3\n[Typical example with the catalyst in the reaction condition]\n\n[Input]\nReaction 20160225-US20160056388A1-0282 description:\nThen, under an argon stream, 2-[3-chloro-5-(9-phenanthryl)phenyl]-4,6-diphenyl-1,3,5-triazine (5.20 g, 10.0 mmol), 4,4,4\u2032,4\u2032,5,5,5\u2032,5\u2032-octamethyl-2,2\u2032-bi-1,3,2-dioxaborolane (3.81 g, 15.0 mmol), palladium acetate (22.5 mg, 0.10 mmol), 2-dicyclohexylphosphino-2\u2032,4\u2032,6\u2032-triisopropyl biphenyl (95.4 mg, 0.20 mmol) and potassium acetate (2.95 g, 30 mmol), were suspended in 1,4-dioxane (200 mL), and the mixture was stirred for 4 hours at 100\u00b0 C. After cooling, the precipitate was removed by filtration using a filter paper. Further, liquid separation was conducted with chloroform, and the organic layer was concentrated to obtain a crude solid. Hexane was added to the crude solid, followed by cooling to ice temperature, and then, the solid was separated by filtration, followed by drying under vacuum to obtain a white solid of 4,6-diphenyl-2-[5-(9-phenanthryl)-3-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)phenyl]-1,3,5-triazine as an intermediate (amount: 6.07 g, yield: 99%).\n\nREACTANTs:\nR1: 2-[3-chloro-5-(9-phenanthryl)phenyl]-4,6-diphenyl-1,3,5-triazine,29;\nR2: 4,4,4\u2032,4\u2032,5,5,5\u2032,5\u2032-octamethyl-2,2\u2032-bi-1,3,2-dioxaborolane,115;\nR3: 2-dicyclohexylphosphino-2\u2032,4\u2032,6\u2032-triisopropyl biphenyl,235;\nR4: potassium acetate,315;\n\nPRODUCTs:\nP1: 4,6-diphenyl-2-[5-(9-phenanthryl)-3-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)phenyl]-1,3,5-triazine,829;\n\nSOLVENTs:\nS1: 1,4-dioxane,370;\nS2: Hexane,644;\n\nCATALYSTs:\nC1: palladium acetate,195;\n\nTIMEs:\nT1: 4 hours,424;\n\nTEMPERATUREs:\nE1: 100\u00b0 C,435;\n\nYIELD:\nY1: yield: 99%,969;\n\n[Output]\nR1.R2.R3.R4>C1.S1.E1.T1>M1\nM1>S2>P1\n\nTraining Example 4\n[Typical example of single-step reaction]\n\n[Input]\nReaction 20100427-US07705028B2-0287 description:\nA solution of [3-[2-(2,6-dichlorophenyl)ethyl]-5-(1-methylethyl)-4-isoxazolyl]methanol (0.085 g, 0.27 mmol), methyl 6-(4-hydroxyphenyl)-2-naphthalenecarboxylate (0.075 g, 0.27 mmol), triphenyl phosphine (0.071 g, 0.27 mmol) and diisopropyl azodicarboxylate (0.049 mL, 0.27 mmol) in toluene (2.7 mL) was placed in microwave reaction tube and heated to 80\u00b0 C. for 1000 seconds. The solution was concentrated and the residue dissolved in a solution of ethyl acetate and methanol, filtered and concentrated. The filtrate was purified by chromatography (silica gel, hexane to 3:7 ethyl acetate:hexanes) to provide the title compound (0.038 g, 24.5%). 1H NMR (DMSO-d6): \u03b4 8.62 (s, 1H), 8.25 (s, 1H), 8.18 (d, J=9 Hz, 1H), 8.06 (d, J=9 Hz, 1H), 7.97 (dd, J=1, 9 Hz, 1H), 7.92 (dd, J=2, 9 Hz, 1H), 7.81 (d, J=9 Hz, 2H), 7.42 (d, J=8 Hz, 2H), 7.25 (t, J=8 Hz, 1H), 7.14 (d, J=9 Hz, 2H), 4.99 (s, 2H), 3.90 (s, 3H), 3.35 (septet, J=7 Hz, overlapping H2O 1H), 3.24-3.20 (m, 2H), 2.89-2.85 (m, 2H), 1.25 (d, J=7 Hz, 6H). ESI-LCMS m/z 574 (M+H)+.\n\nREACTANTs:\nR1: [3-[2-(2,6-dichlorophenyl)ethyl]-5-(1-methylethyl)-4-isoxazolyl]methanol,14;\nR2: methyl 6-(4-hydroxyphenyl)-2-naphthalenecarboxylate,109;\nR3: diisopropyl azodicarboxylate,228;\nR4: triphenyl phosphine,183;\n\n\nPRODUCTs:\nP1: title compound,613;\n\n\nSOLVENTs:\nS1: toluene,282;\n\nCATALYSTs:\nNone\n\nTIMEs:\nT1: 1000 seconds,362;\n\n\nTEMPERATUREs:\nE1: 80\u00b0 C.,351;\n\n\nYIELD:\n24.5%,638;\n\n[Output]\nR1.R2.R3.R4>S1.E1.T1>P1"
    
    print(num_tokens_from_messages(message))
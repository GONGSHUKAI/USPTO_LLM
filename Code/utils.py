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
from fixed_prompt import FIXED_PROMPT, SAMPLE_REQUEST

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
        process_file='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Larrea/raw/uspto_full.json',
        rxn_idx_file='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Larrea/raw/rxn_idx.json'
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
        rxn_idx_file='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Larrea/raw/rxn_idx.json',
    ):
    with open(rxn_idx_file,'w') as f:
        json.dump({"rxn_idx":rxn_idx}, f)

def num_tokens_from_messages(message, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    return len(encoding.encode(message))


if __name__=='__main__':
    message = SAMPLE_REQUEST
    
    print(num_tokens_from_messages(message))
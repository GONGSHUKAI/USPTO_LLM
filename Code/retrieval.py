#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Shen Yuan

import numpy as np
from bisect import bisect_left
import json
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import os

from rdkit import Chem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from utils import scaffold_to_index, mol_to_index, date_to_index, generate_scaffold, make_mol
except:
    from .utils import scaffold_to_index, mol_to_index, date_to_index, generate_scaffold, make_mol

def create_retrieval(
    data,
    retrieve_by: str = 'scaffold',
    key_molecule_index: int = 0,
    key_molecule_type: str = 'PRODUCT',
):
    
    if retrieve_by == 'scaffold':
        key_mols = [m[key_molecule_type][key_molecule_index][-1] for m in data]
        scaffold_to_indices = scaffold_to_index(key_mols)
        return scaffold_to_indices
    
    elif retrieve_by == 'product_fp':
        key_mols = [m[key_molecule_type][key_molecule_index][-1] for m in data]
        mol_to_indices = mol_to_index(key_mols)
        return mol_to_indices
        
    elif retrieve_by == 'date':
        key_dates = [m['ID'].split('-')[0] for m in data]
        date_to_indices = date_to_index(key_dates)
        return date_to_indices
    
    

def get_similar_by_fp(query_fp, key_fps):
    similarity_fp_list = [DataStructs.TanimotoSimilarity(query_fp, fp) for fp in key_fps]
    similarity_index_sorted = sorted(
        range(len(key_fps)),
        key = lambda x: similarity_fp_list[x],
        reverse=True,
    )
    return similarity_index_sorted
    
def retrieve_rxn(
    data: List,
    to_indices: dict = None,
    number: int = 20,
    retrieve_by: str = 'scaffold',
    key_molecule_index: int = 0,
    key_molecule_type: str = 'PRODUCT',
):
    if to_indices is None:
        save_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\retrieval\\'
        with open(save_path+retrieve_by+'\\indices.json','r') as f:
            to_indices = json.load(f)
            
    if retrieve_by == 'scaffold':
        key_mols = [m[key_molecule_type][key_molecule_index][-1] for m in data]
        key_scaffolds = [generate_scaffold(m) for m in key_mols]
        key_fps = [GetMorganFingerprintAsBitVect(make_mol(s), 2) for s in key_scaffolds]
        to_indices_keys = list(to_indices.keys())
        to_indices_key_fps = [GetMorganFingerprintAsBitVect(make_mol(s), 2) for s in to_indices_keys]
        similarity_indices_sorted = [get_similar_by_fp(fp, to_indices_key_fps) for fp in key_fps]
        rxns_by_scaffolds = []
        for idx, indices in enumerate(similarity_indices_sorted):
            rxns = set(to_indices.get(key_scaffolds[idx], []))
            i = 0
            while len(rxns) < number:
                rxns.update(to_indices[to_indices_keys[indices[i]]])
                i += 1
            rxns_by_scaffolds.append(list(rxns)[:number])
            
        return rxns_by_scaffolds
    
    elif retrieve_by == 'product_fp':
        key_mols = [m[key_molecule_type][key_molecule_index][-1] for m in data]
        key_mols = [Chem.MolToSmiles(make_mol(m)) for m in key_mols]
        key_fps = [GetMorganFingerprintAsBitVect(make_mol(s), 2) for s in key_mols]
        to_indices_keys = list(to_indices.keys())
        to_indices_key_fps = [GetMorganFingerprintAsBitVect(make_mol(s), 2) for s in to_indices_keys]
        similarity_indices_sorted = [get_similar_by_fp(fp, to_indices_key_fps) for fp in key_fps]
        rxns_by_fps = []
        for idx, indices in enumerate(similarity_indices_sorted):
            rxns = set(to_indices.get(key_mols[idx], []))
            i = 0
            while len(rxns) < number:
                rxns.update(to_indices[to_indices_keys[indices[i]]])
                i += 1
            rxns_by_fps.append(list(rxns)[:number])
            
        return rxns_by_fps
        
    elif retrieve_by == 'date':
        key_dates = [m['ID'].split('-')[0] for m in data]
        to_indices_keys = sorted(to_indices.keys())
        rxns_by_date = []
        for idx, date in enumerate(key_dates):
            rxns = set(to_indices.get(date, []))
            i = bisect_left(to_indices_keys, date)
            while len(rxns) < number:
                if i >= len(to_indices_keys):
                    print(f'exceed the max length of date list!')
                    break
                rxns.update(to_indices[to_indices_keys[i]])
                i += 1
            rxns_by_date.append(list(rxns)[:number])
            
        return rxns_by_date


if __name__ == '__main__':  
    full_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\full\\uspto.json'
    save_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\retrieval\\'
    retrieve_by = 'product_fp'
    
    if not os.path.exists(save_path+retrieve_by+'\\indices.json'):
        all_rxns_list = []
        with open(full_path, 'r') as f:
            rxns = f.__iter__()
            while True:
                try:
                    rxn = json.loads(next(rxns))
                    all_rxns_list.append(rxn)
                    
                except StopIteration:
                    print("finish loading.")
                    print(f"reaction number: {len(all_rxns_list):}")
                    break
                
        to_indices = create_retrieval(
            data=all_rxns_list,
            retrieve_by=retrieve_by,
        )
        
        print(f'Total {retrieve_by:} = {len(to_indices):,}')
        with open(save_path+retrieve_by+'\\indices.json','w') as f:
            json.dump(to_indices, f)

    with open(save_path+retrieve_by+'\\indices.json','r') as f:
        to_indices = json.load(f)
            
    test_rxns = [{"ID": "20080527-US07378418B2-0328", "STARTING_MATERIAL": [["2-(4-(trifluoromethyl)benzyl)-8-(4-chlorophenyl)-7-chloro-[1,2,4]triazolo[4,3-b]pyridazin-3(2H)-one", "FC(C1=CC=C(CN2N=C3N(N=CC(=C3C3=CC=C(C=C3)Cl)Cl)C2=O)C=C1)(F)F"], ["K2CO3", "C(=O)([O-])[O-].[K+].[K+]"], ["phenol", "C1(=CC=CC=C1)O"]], "REAGENT_CATALYST": [], "PRODUCT": [["2-(4-(Trifluoromethyl)benzyl)-8-(4-chlorophenyl)-7-phenoxy-[1,2,4]triazolo[4,3-b]pyridazin-3(2H)-one", "FC(C1=CC=C(CN2N=C3N(N=CC(=C3C3=CC=C(C=C3)Cl)OC3=CC=CC=C3)C2=O)C=C1)(F)F"], ["8-(4-chlorophenyl)-7-phenoxy-[1,2,4]triazolo[4,3-b]pyridazin-3(2H)-one", "ClC1=CC=C(C=C1)C=1C=2N(N=CC1OC1=CC=CC=C1)C(NN2)=O"]], "SOLVENT": [], "TIME": "16 hours", "TEMPERATURE": "room temperature", "YIELD_PERCENT": "62 %"}]
    
    rxns_by_date = retrieve_rxn(
        data = test_rxns,
        to_indices = to_indices,
        retrieve_by = retrieve_by,
    )
    print(rxns_by_date)
    

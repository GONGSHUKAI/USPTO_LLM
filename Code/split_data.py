from collections import defaultdict
import logging
from random import Random
from typing import Dict, List, Set, Tuple, Union
import warnings
import copy
import json

from tqdm import tqdm
import numpy as np

from utils import *

def create_split_data(data: List,
                   split_by: str = 'scaffold',
                   reverse: bool = True,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   key_molecule_index: int = 0,
                   key_molecule_type: str = 'PRODUCT',
                   seed: int = 0) -> Tuple[List, List, List]:
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.

    :param data: A :class:`MoleculeDataset`.
    :param split_by: ['scaffold', 'date', 'random']
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    
    # Seed randomness
    random = Random(seed)

    if split_by == 'random':
        index_sets = list(range(len(data)))
        random.shuffle(index_sets)
        
    elif split_by == 'date':
        key_dates = [m['ID'].split('-')[0] for m in data]
        date_to_indices = date_to_index(key_dates)
        index_sets = sorted(list(date_to_indices.values()),
                                key=lambda index_set: len(index_set),
                                reverse=reverse)
    
    elif split_by == 'scaffold':
        # Map from scaffold to index in the data
        key_mols = [m[key_molecule_type][key_molecule_index][-1] for m in data]
        scaffold_to_indices = scaffold_to_index(key_mols)

        if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.seed(seed)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:  # Sort from largest to smallest scaffold sets
            index_sets = sorted(list(scaffold_to_indices.values()),
                                key=lambda index_set: len(index_set),
                                reverse=reverse)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_count += 1
        else:
            test += index_set
            test_count += 1
    
    print(
        f'Total {split_by:}s = {len(index_sets):,} | '
        f'train {split_by:}s = {train_count:,} | '
        f'val {split_by:}s = {val_count:,} | '
        f'test {split_by:}s = {test_count:,}'
        )
    # if logger is not None:
    #     logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
    #                  f'train scaffolds = {train_scaffold_count:,} | '
    #                  f'val scaffolds = {val_scaffold_count:,} | '
    #                  f'test scaffolds = {test_scaffold_count:,}')

    # if logger is not None and not data.is_atom_bond_targets:
    #     log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return train, val, test


def log_scaffold_stats(data: List,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A logger for recording output.
    :return: A list of tuples where each tuple contains a list of average target values
             across the first :code:`num_labels` labels and a list of the number of non-zero values for
             the first :code:`num_scaffolds` scaffolds, sorted in decreasing order of scaffold frequency.
    """
    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels:')

    stats = []
    index_sets = sorted(index_sets, key=lambda idx_set: len(idx_set), reverse=True)
    for scaffold_num, index_set in enumerate(index_sets[:num_scaffolds]):
        data_set = [data[i] for i in index_set]
        targets = np.array([d.targets for d in data_set], dtype=float)

        with warnings.catch_warnings():  # Likely warning of empty slice of target has no values besides NaN
            warnings.simplefilter('ignore', category=RuntimeWarning)
            target_avgs = np.nanmean(targets, axis=0)[:num_labels]

        counts = np.count_nonzero(~np.isnan(targets), axis=0)[:num_labels]
        stats.append((target_avgs, counts))

        if logger is not None:
            logger.debug(f'Scaffold {scaffold_num}')
            for task_num, (target_avg, count) in enumerate(zip(target_avgs, counts)):
                logger.debug(f'Task {task_num}: count = {count:,} | target average = {target_avg:.6f}')
            logger.debug('\n')

    return stats


if __name__=='__main__':
    full_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\full\\uspto.json'
    split_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\split\\'
    split_by = 'scaffold'
    
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

    train, val, test = create_split_data(
        all_rxns_list, 
        split_by = split_by,
        reverse = True,
        )
    
    with open(split_path+split_by+'\\train.json','w') as f:
        for rxn in train:
            json_string = json.dumps(rxn)
            f.write(json_string + "\n")
            
    with open(split_path+split_by+'\\val.json','w') as f:
        for rxn in val:
            json_string = json.dumps(rxn)
            f.write(json_string + "\n")
            
    with open(split_path+split_by+'\\test.json','w') as f:
        for rxn in test:
            json_string = json.dumps(rxn)
            f.write(json_string + "\n")
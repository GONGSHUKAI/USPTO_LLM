import json
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
from random import shuffle, randint
import re
from typing import Dict, List, Set, Tuple, Union

from prompt_to_json import FIXED_PROMPT
from utils import get_all_reaction, num_tokens_from_messages, write_rxn_idx
from retrieval import retrieve_rxn

def write_requests_file(
    requests, 
    filename='/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Larrea/requests/uspto_requests.json', 
    model_name='gpt-4', 
    temperature=0.2,
    n=1,
    prompt=None
    ):
    if prompt is None:
        prompt = FIXED_PROMPT
    with open(filename, "w") as f:
        for request in requests:
            # Create a list of messages for each request
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": request}
            ]
            # Write the messages to the JSONL file
            json_string = json.dumps(
                {
                    "model": model_name, 
                    "messages": messages,
                    "temperature": temperature,
                    "n":n,
                }
            )
            f.write(json_string + "\n")
            
def generate_rxn(rxn):
    if rxn['paragraphNum'] is not None:
        Id = "%s-%s-%s" % (rxn['date'], rxn['documentId'], rxn['paragraphNum']) 
    else:
        Id = "%s-%s" % (rxn['date'], rxn['documentId'])
        
    format_rxn = "Reaction %s description:\n%s\n" % (Id, rxn['paragraphText'])
    
    # Creates a dictionary of the molecule types and their corresponding entities
    types = {'REACTANT':'R', 'PRODUCT':'P', 'SOLVENT':'S', 'CATALYST':'C', 'TIME':'T', 'TEMPERATURE':'E', 'YIELD':'Y'}

    def get_format_mol(mol_list, mol_type='PRODUCT'):
        format_mol = "\n" + mol_type + "s:"
        if len(mol_list) == 0:
            format_mol += "\nNone\n"
        else:
            format_mol += "\n"
            count = 1
            for mol in mol_list:
                format_mol += "%s: %s,%s;\n" % (types[mol_type]+str(count),mol['name'], mol['pos'])
                count += 1
            format_mol += "\n"
        return format_mol
    
    def get_format_cond(cond_list, cond_type='TIME'):
        if cond_type == 'YIELD':
            format_cond = "\n" + cond_type + ":"
        else:
            format_cond = "\n" + cond_type + "s:"
        if len(cond_list) == 0:
            format_cond += "\nNone\n"
        else:
            format_cond += "\n"
            count = 1
            for cond in cond_list:
                format_cond += "%s: %s,%s;\n" % (types[cond_type]+str(count), cond['text'], cond['pos'])
                count += 1
            format_cond += "\n"
        return format_cond
        
    format_reactant = get_format_mol(rxn['reactantList'], 'REACTANT')
    format_product = get_format_mol(rxn['productList'], 'PRODUCT')
    format_solvent = get_format_mol(rxn['solventList'], 'SOLVENT')
    format_catalyst = get_format_mol(rxn['catalystList'], 'CATALYST')
    format_time = get_format_cond(rxn['timeList'], 'TIME')
    format_temperature = get_format_cond(rxn['temperatureList'], 'TEMPERATURE')
    format_yield = get_format_cond([rxn['yield']], 'YIELD')
    # if 'None' in format_catalyst:
    #     return None
    
    format_rxn += format_reactant + format_product + format_solvent + format_catalyst + format_time + format_temperature + format_yield
    return format_rxn 

def generate_datasets(
    rxns: List,
    given_prop_names: List[str] = ['ID', 'STARTING_MATERIAL', 'REAGENT_CATALYST', 'PRODUCT', 'SOLVENT', 'TIME', 'TEMPERATURE'],
    predict_prop_names: List[str] = ['YIELD_PERCENT'],
    few_shot: bool = False,
    few_shot_rxns: List[dict] = None,
    few_shot_num: int = 5,
    few_shot_retrieve_by: str = 'scaffold',
):
    if few_shot:
        print('retrieve few shot.')
        few_shot_rxn_indices = retrieve_rxn(
            data = rxns,
            retrieve_by = few_shot_retrieve_by,
            number = few_shot_num + 1, # 需要预测的目标反应也有可能在提取出来的反应列表中，所以要多提取一个
        )
                
    all_input_prompts = []
    task_prompt = 'Given some properties in a chemical reaction including '+ ', '.join(given_prop_names) + \
        ', please predict ' + ', '.join(predict_prop_names) + '\n' + \
        'Output with no explanation, no introduction, only the predicted properties. ' + \
        'Do not output the examples.'
    for idx, rxn in tqdm(enumerate(rxns), total = len(rxns)):
        input_prompt = task_prompt
        
        if few_shot:
            # 需要预测的目标反应也有可能在提取出来的反应列表中，所以要去除一下
            few_shot_rxn_index, tmp = few_shot_rxn_indices[idx], []
            for index in few_shot_rxn_index:
                if few_shot_rxns[index]['ID'] != rxn['ID']:
                    tmp.append(index)
            few_shot_rxn_index = tmp[:few_shot_num]
            
            for i, index in enumerate(few_shot_rxn_index):
                input_prompt += f'\nExample {i+1}:\n'
                for prop_name in given_prop_names + predict_prop_names:
                    if prop_name in ['STARTING_MATERIAL', 'REAGENT_CATALYST', 'PRODUCT', 'SOLVENT']:
                        if prop_name in few_shot_rxns[index].keys():
                            string = ';'.join([','.join(mol) for mol in few_shot_rxns[index][prop_name]])
                            input_prompt += f'{prop_name}: {string}\n'
                        else:
                            input_prompt += f'{prop_name}: None\n'
                    else:
                        input_prompt += f'{prop_name}: {few_shot_rxns[index].get(prop_name, None)}\n'
                if i == len(few_shot_rxn_index)-1:
                    input_prompt += f'\nExample {i+2}:\n'
        
        for prop_name in given_prop_names:
            if prop_name in ['STARTING_MATERIAL', 'REAGENT_CATALYST', 'PRODUCT', 'SOLVENT']:
                string = ';'.join([','.join(mol) for mol in rxn[prop_name]])
                input_prompt += f'{prop_name}: {string}\n'
            else:
                input_prompt += f'{prop_name}: {rxn.get(prop_name, None)}\n'
            
        for prop_name in predict_prop_names:
            input_prompt += f'{prop_name}: ***answer***\n'
    
        all_input_prompts.append(input_prompt)
        
    return all_input_prompts
        

if __name__=='__main__':
    # path = 'D:\\Projects\\InstructMolPT\\datasets\\5104873\\applications'
    # path2 = 'D:\\Projects\\InstructMolPT\\datasets\\5104873\\grants'
    # file_list = get_filelist(path) + get_filelist(path2)

    max_req_len = 16000
    
    rxn_list, rxn_idx = get_all_reaction()

    req_num = 5

    # new_rxn_list = []
    # num_rxn = 0
    # for rxn in rxn_list:
    #     if len(rxn['temperatureList']) == 1 and len(rxn['timeList']) == 1:
    #         num_rxn += 1
    #         new_rxn_list.append(rxn)
    # print(num_rxn) 
    # with open('D:\\Projects\\InstructMolPT\\datasets\\Larrea\\raw\\uspto_onetime.json','w') as f:
    #     json.dump(new_rxn_list, f, indent=2)
    
    # with open('D:\\Projects\\InstructMolPT\\datasets\\Larrea\\raw\\uspto_onetime.json','r') as f:
    #     all_reaction_list = json.load(f)
    
    # 1/0
    
    num_token_prompt = num_tokens_from_messages(FIXED_PROMPT)

    reqs = []
    total_rxn_num = 0
    reaction_per_request = 3        # One request can include at most `reaction_per_request` reactions
    for i in range(req_num):
        req = ""
        rxn_num = 0
        num_token_req = num_token_prompt
        while True:
            if rxn_num == reaction_per_request:    
                break
            
            format_rxn = generate_rxn(rxn_list[rxn_idx])
            if format_rxn is None:
                rxn_idx += 1
                continue
                
            num_token_rxn = num_tokens_from_messages("{\n%s}\n\n" % format_rxn)
            if num_token_rxn > max_req_len:
                rxn_idx += 1
            elif num_token_req+num_token_rxn > max_req_len and rxn_num == 0:
                rxn_idx += 1
            elif num_token_req+num_token_rxn > max_req_len and rxn_num != 0:
                break
            else:
                req += "[Input]\n%s" % format_rxn
                num_token_req += num_token_rxn
                rxn_num += 1
                rxn_idx += 1
                # req += "Remember to check the 5 rules stated before, especially the first rule, make sure you don't have more than two '<' in one substep!\n"
                req += "[Output]:\n"
        reqs.append(req)
        total_rxn_num += rxn_num
    
    write_rxn_idx(rxn_idx)        # Change the rxn_idx in the file to get the next batch of reactions
    print('generate requests: %d'% req_num)
    print('include reactions: %d'% total_rxn_num)
    write_requests_file(reqs)
    
    
    
    
    

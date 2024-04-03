import re
import json

def parse_mols_string(mols_string):
    if 'None' in mols_string:
        return []
    
    mols = []
    mols_string = mols_string.split(';')[:-1]
    for mol in mols_string:
        items = mol.split(',')
        smiles = items[-1]
        mol_name = mol[:-(len(smiles)+1)]
        mols.append((mol_name, smiles))
    return mols
    
    
def parse_reaction_string(reaction_string):
    reactions = {}
    current_reaction = None

    lines = reaction_string.split('\n')
    for line in lines:
        if line.startswith("Reaction"):
            # New reaction found, extract the ID
            match = re.search(r'Reaction (\d+-\w+(-\d*)*):', line)
            if match:
                current_reaction_id = match.group(1)
                current_reaction = {"ID": current_reaction_id}
                reactions[current_reaction_id] = current_reaction
        elif current_reaction:
            # Inside a reaction, extract other attributes
            match = re.search(r'([^:]+): (.+)', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key in ['STARTING_MATERIAL', 'REAGENT_CATALYST', 'PRODUCT', 'SOLVENT']:
                    value = parse_mols_string(value)
                current_reaction[key] = value

    reactions = list(reactions.values())
    return reactions

def parse_yield_string(
    yield_string: str,
):
    match = re.findall(r'(\d*) *%', yield_string)
    if match:
        try:
            yield_percent = float(match[-1])
            if yield_percent > 100 or yield_percent < 0:
                return None
            else:
                return yield_percent
        except:
            return None
    else:
        return None
    
def parse_ID_string(
    ID_string: str,
    key_index: int = -1,
):
    match = re.findall(r'ID: (\d+-\w+-*\d*)', ID_string)
    if match:
        try:
            return match[key_index]
        except:
            return None
    else:
        return None

def filter_rxns(rxns_list):
    available_rxns_list = []
    for rxn in rxns_list:
        # 每个反应必须拥有以下全部key
        requisite_keys_list = ['ID', 'STARTING_MATERIAL', 'PRODUCT', 'TIME', 'TEMPERATURE', 'YIELD_PERCENT']
        flag = True
        for key in requisite_keys_list:
            if key not in rxn.keys():
                flag = False
                break
        if not flag:
            continue
        
        # 反应物和产物不能为空
        if len(rxn['STARTING_MATERIAL']) == 0 or len(rxn['PRODUCT']) == 0:
            continue
        
        # 反应时间，反应温度，产率不能全为None
        if rxn['TIME'] is None and rxn['TEMPERATURE'] is None and rxn['YIELD_PERCENT'] is None:
            continue
        
        # 以下key如果没有，可以补上空的list
        optional_keys_list = ['REAGENT_CATALYST', 'SOLVENT']
        for key in optional_keys_list:
            if key not in rxn.keys():
                rxn[key] = []
        
        # 如果产物有多个，按照smiles从长到短排列
        if len(rxn['PRODUCT']) > 1:
            rxn['PRODUCT'] = sorted(rxn['PRODUCT'],
                            key=lambda x: len(x[-1]),
                            reverse=True)
        
        # 如果产物有多个，按照smiles从长到短排列
        if len(rxn['STARTING_MATERIAL']) > 1:
            rxn['STARTING_MATERIAL'] = sorted(rxn['STARTING_MATERIAL'],
                            key=lambda x: len(x[-1]),
                            reverse=True)
        
        new_rxn = dict()
        total_prop_names = ['ID', 'STARTING_MATERIAL', 'SOLVENT', 'REAGENT_CATALYST', 'PRODUCT',  'TIME', 'TEMPERATURE', 'YIELD_PERCENT']
        for key in total_prop_names:
            new_rxn[key] = rxn[key]
        
        available_rxns_list.append(new_rxn)
        
    return available_rxns_list

if __name__=='__main__':

    # # 你的输入字符串
    # input_string = "Reaction 20151231-US20150376208A1-1618: \nSTARTING_MATERIAL: 2-amino-3-nitrophenol,NC1=C(C=CC=C1[N+](=O)[O-])O;bromine,BrBr;\nREAGENT_CATALYST: None\nPRODUCT: 2-amino-5-bromo-3-nitrophenol,NC1=C(C=C(C=C1[N+](=O)[O-])Br)O;\nSOLVENT: acetic acid,C(C)(=O)O;\nTIME: None\nTEMPERATURE: 0\u00b0C\nYIELD_PERCENT: 60 %\n\nReaction 20101130-US07842694B2-1107: \nSTARTING_MATERIAL: 3-hydroxy-5-{[(1S)-1-methyl-2-(methyloxy)ethyl]oxy}-N-(5-methylpyrazin-2-yl)benzamide,OC=1C=C(C(=O)NC2=NC=C(N=C2)C)C=C(C1)O[C@H](COC)C;Trimethylsilyl iodide,C[Si](C)(C)I;\nREAGENT_CATALYST: None\nPRODUCT: 3-Hydroxy-5-{[(1S)-2-hydroxy-1-methylethyl]oxy}-N-(5-methylpyrazin-2-yl)benzamide,OC=1C=C(C(=O)NC2=NC=C(N=C2)C)C=C(C1)O[C@H](CO)C;\nSOLVENT: dry acetonitrile,C(C)#N;\nTIME: 24 hours\nTEMPERATURE: None\nYIELD_PERCENT: None\n\nReaction 19980331-US05733914: \nSTARTING_MATERIAL: 6-(2,6-dichlorophenyl)-8-methyl-2-methylsulfanyl-8H-pyrido[2,3-d]pyrimidin-7-one,ClC1=C(C(=CC=C1)Cl)C1=CC2=C(N=C(N=C2)SC)N(C1=O)C;3-aminopyridine base,None;3-aminopyridine hydrochloride,Cl.NC=1C=NC=CC1; \nREAGENT_CATALYST: None\nPRODUCT: 6-(2,6-Dichlorophenyl)-8-methyl-2-(pyridin-3-ylamino)-8H-pyrido[2,3-d]pyrimidin-7-one,ClC1=C(C(=CC=C1)Cl)C1=CC2=C(N=C(N=C2)NC=2C=NC=CC2)N(C1=O)C;\nSOLVENT: None\nTIME: 1 hour\nTEMPERATURE: 210\u00b0C\nYIELD_PERCENT: None\n\"\"\""

    # # 解析字符串并输出字典
    # reactions_dict = parse_reaction_string(input_string)
    # print(input_string)
    # print(reactions_dict)
    
    all_rxns_list = []
    request_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\requests\\results_uspto_requests.json'
    save_path = 'D:\\Projects\\InstructMolPT\\datasets\\uspto_shenme\\full\\uspto.json'
    with open(request_path, 'r') as f:
        requests = f.__iter__()
        while True:
            try:
                request_json = json.loads(next(requests))
                # print(request_json[0]["messages"][1]["content"])
                # print(request_json[1]["choices"][0]["message"]["content"])
                rxn_string = request_json[1]["choices"][0]["message"]["content"]
                rxns_list = parse_reaction_string(rxn_string)
                rxns_list = filter_rxns(rxns_list)
                all_rxns_list.extend(rxns_list)
                
            except StopIteration:
                print("finish extraction.")
                break
            
    with open(save_path,'w') as f:
        for rxn in all_rxns_list:
            json_string = json.dumps(rxn)
            f.write(json_string + "\n")
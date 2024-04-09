import json
import os
import copy
file_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Larrea/requests/result_uspto_requests.json'
file_path2 = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Larrea/requests/uspto_requests.json'
output_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Training_Info/'

with open(file_path, 'r') as raw_response: 
    lines = raw_response.readlines()
    total_data = [json.loads(line) for line in lines]

example_data = total_data[0]

def extract_description(example_data):
    total_description = example_data[0]['messages'][1]['content']

    total_description = total_description.split('[Input]\nReaction ')[1:]

    for i, each_description in enumerate(total_description):
        first_split = each_description.index('\n')
        total_description[i] = each_description[first_split+1:]
    
    return total_description

def extract_gpt_response(example_data):
    total_response = example_data[1]['choices'][0]['message']['content']

    total_response = total_response.split('Reaction ')[1:]

    reaction_id = []
    for i, response in enumerate(total_response):
        first_split = response.index('\n')
        last_split = response.index('P1')
        reaction_id.append(response[:first_split])
        total_response[i] = response[first_split+1:last_split+2]
    
    return total_response, reaction_id

def check_invalid_answer(answer)->bool:
    # Check the validity of output heterogeneous graph
    # answer is a string in the form of ""R1.R2.R4>S1.E1.T1>M1\nM1.R3>E2.T2>P1""
    # split it by '\n' to get each 'substep'
    substeps = answer.split('\n')
    for substep in substeps:
        # If each step has more than 2 '>', it is invalid
        if substep.count('>') != 2:
            return False
        # If any entity with R and M is not in front of the first '>', it is invalid
        if 'R' in substep:
            if substep.index('R') > substep.index('>'):
                return False
        if 'M' in substep:
            if substep.index('M') > substep.index('>') and substep.index('M') < substep.rindex('>'):
                return False
        if 'P' in substep:
            if substep.index('P') < substep.rindex('>'):
                return False
        # If any entity with S, E, T, C is not behind the second '>', it is invalid
        if 'S' in substep:
            if substep.index('S') < substep.index('>') or substep.index('S') > substep.rindex('>'):
                return False
        if 'E' in substep:
            if substep.index('E') < substep.index('>') or substep.index('E') > substep.rindex('>'):
                return False
        if 'T' in substep:
            if substep.index('T') < substep.index('>') or substep.index('T') > substep.rindex('>'):
                return False
        if 'C' in substep:
            if substep.index('C') < substep.index('>') or substep.index('C') > substep.rindex('>'):
                return False
    return True

def output_structured_data(request_in_each_line = 0, threshold = 3):
    # Read raw response first
    with open(file_path, 'r') as raw_response: 
        lines = raw_response.readlines()
        total_data = [json.loads(line) for line in lines]
    
    modified_request = copy.deepcopy(total_data[0][0])     # Fixed Prompt
    modified_request['messages'][1]['content'] = ""

    for dataline in total_data:
        total_description = extract_description(dataline)
        total_response, reaction_id = extract_gpt_response(dataline)

        for idx, response in enumerate(total_response):
            if check_invalid_answer(response):
                with open(output_path + "uspto_multiple_step.json", 'a') as structured_data:
                    structured_data.write(json.dumps({'reaction_id': reaction_id[idx], 'description': total_description[idx], 'response': response}) + '\n')
            else:
                with open(output_path + "uspto_invalid.json", 'a') as invalid_data:
                    if request_in_each_line < threshold:
                        modified_request['messages'][1]['content'] += ("[Input]\nReaction " + reaction_id[idx] + " description:\n" + total_description[idx] + "Don't make the mistake of '" + response + "' .Check the 5 rules again!\n")
                        request_in_each_line += 1
                    else:
                        request_in_each_line = 0
                        invalid_data.write(json.dumps(modified_request) + '\n')
                        modified_request['messages'][1]['content'] = ""
                        modified_request['messages'][1]['content'] += ("[Input]\nReaction " + reaction_id[idx] + " description:\n" + total_description[idx] + "Don't make the mistake of '" + response + "' .Check the 5 rules again!\n")
                        request_in_each_line += 1
        
    # Less than threshold requests in one line, still need to write them
    if request_in_each_line != 0:
        with open(output_path + "uspto_invalid.json", 'a') as invalid_data:
            invalid_data.write(json.dumps(modified_request) + '\n')
    # If no error arise, print success
    print("Structured data output success!")

if os.path.exists(output_path + "uspto_multiple_step.json"):
    os.remove(output_path + "uspto_multiple_step.json")
if os.path.exists(output_path + "uspto_invalid.json"):
    os.remove(output_path + "uspto_invalid.json")
output_structured_data()
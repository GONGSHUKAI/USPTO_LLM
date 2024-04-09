import json
file_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Prompt/prompt_engineering.txt'
output_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO_LLM/Prompt/prompt_engineering.json'
with open(file_path, 'r') as file:
    content = file.read()

json_data = {
    'content': content
}

json_string = json.dumps(json_data)

with open(output_path, 'w') as file:
    file.write(json_string)

FIXED_PROMPT = json_string[13:-2] 
# Remove the first 13 characters            {content: "
# Remove the last 2 characters                        "}

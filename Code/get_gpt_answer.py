import json
file_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Larrea/requests/result_uspto_requests.json'
output_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Training_Info/'

with open(file_path, 'r') as file: 
    lines = file.readlines()
    total_data = [json.loads(line) for line in lines]


output_index = 1
for data in total_data:
    with open(output_path + "reaction" + str(output_index) + ".txt", 'w') as file:
        file.write(data[0]['messages'][1]['content'] + '\n')
        for index in range(len(data[1]['choices'])):
            file.write(data[1]['choices'][index]['message']['content'] + '\n')
            file.write('\n')
    output_index += 1


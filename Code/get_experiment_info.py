import json
request_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Larrea/requests/uspto_requests.json'
request_list = []
with open(request_path, 'r') as f:
    requests = f.__iter__()
    while True:
        try:
            request_json = json.loads(next(requests))
            request_list.append(request_json)
        except StopIteration:
            print("finish extraction.")
            break

num = 1
experiment_info_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Experiment_info/info_{}.txt'.format(num)
for request in request_list:
    try:
        # write the information into a txt file
        with open(experiment_info_path, 'a') as f:
            f.write(request[0]["messages"][1]["content"])
            f.write(request[1]["choices"][0]["message"]["content"])
    except:
        with open(experiment_info_path, 'a') as f:
            f.write(request["messages"][1]["content"])
    num += 1
    experiment_info_path = '/Users/gongshukai/Desktop/ML RESEARCH/Ongoing Project/USPTO dataset/Experiment_info/info_{}.txt'.format(num)
    
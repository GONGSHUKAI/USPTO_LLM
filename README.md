# USPTO-LLM

## Introduction about USPTO-LLM
USPTO-LLM is an information-enriched chemical reaction dataset that provides more side information (reaction conditions and reaction steps division) for developing new reaction prediction and retrosynthesis methods and inspires new problems, such as reaction condition prediction. 

The dataset comprises over 247K chemical reactions extracted from the patent documents of USPTO (United States Patent and Trademark Office), encompassing abundant information on reaction conditions. 

We employ large language models to expedite the data collection procedures automatically with a reliable quality control process. The extracted chemical reactions are organized as heterogeneous directed graphs, allowing us to formulate a series of prediction tasks, such as reaction prediction, retrosynthesis, and reaction condition prediction, in a unified graph-filling framework.

The chemical reactions are organized as heterogeneous directed graphs along with their reaction condition information, such as:
|id|class|rs>>ps|solvents|catalyst|temperature|time|
|-|-|-|-|-|-|-|
|20160114-US20160007601A1-0113| -1  | `[Cl-].C(C)N=C=NCCC[NH+](C)C.C1(=CC=CC=C1)CCC(=O)O.CC(\C=C/C)O>>C1(=CC=CC=C1)C#CC(=O)OC(C)\C=C/C` | ['`C(Cl)Cl`'] | [`'CN(C)C=1C=CN=CC1'`] | [`'room temperature'`] | [`'25200'`]|


## About this repository
This repository contains the code for generating the USPTO-LLM dataset.

### Code 
In 'USPTO-LLM/Code', there are several python files and jupyter notebooks.

1. `api_request_parallel_processor.py` is used to call the API to generate the heterogeneous directed graph of USPTO-LLM. Our default API is gpt-4-1106-preview.

2. `generate_request.py` is used to generate the json file `uspto_request.json` for each API submission based on the reaction data in `uspto_full.json` and the `FIXED_PROMPT`.

3. `extract_api_outcome.ipynb` is used to extract the reaction data from the `result_uspto_request.json` generated by the API, including the "pattern repair" technique mentioned in the paper.

4. `prompt_to_json.py` is used to convert the written prompt into json format and store it in the string variable `FIXED_PROMPT`.

The running order is: `prompt_to_json.py` → `generate_request.py` → `api_request_parallel_processor.py` → `extract_api_outcome.ipynb`

### Prompt

Prompt is used to guide the large language model to generate the standardized heterogeneous directed graph of USPTO-LLM. The prompt is stored in the `Prompt` folder, including the `.txt` and `.json` versions of the prompt.

### Supplement

In the `Supplement` folder, `annotation_process.pdf` provides the detailed annotation process of the USPTO-LLM dataset, including the detailed design of prompt, the process of the pattern repair technique, and the mechanism of our two-round generation strategy.

##### Training_Info

The following table shows the performance of different large language models used in this work to generate heterogeneous directed graphs in the USPTO-LLM dataset.

| LLM Type  | gpt-4-0613                                                    | gpt-4-1106-preview | gpt-4-0125-preview | gpt-3.5-turbo-0125 |
|-----------|---------------------------------------------------------------|--------------------|--------------------|--------------------|
|           | Dataset                                                       | USPTO_full         | USPTO_full         | USPTO_full         | USPTO_full |
|           | Reactions <br>per Request                                     | 3                  | 3                  | 3                  | 3          |
| 1st-round | API calls                                                     | 1030               | 2000               | 2000               | 800        |
|           | Success Rate<br>(Before Pattern Repair)                       | 0.67               | 0.65               | 0.66               | 0.32       |
|           | Success Rate<br>(After Pattern Repair)                        | 0.88               | 0.84               | 0.86               | 0.52       |
|           | Valid HeteroGraphs<br>per API call<br>(Before Pattern Repair) | 2.00               | 1.94               | 1.98               | 0.97       |
|           | Valid HeteroGraphs <br>per API call<br>(After Pattern Repair) | 2.64               | 2.53               | 2.57               | 1.57       |
| 2nd-round | API calls                                                     | 434                | 233                | 285                | 525        |
|           | Success Rate<br>(Before Pattern Repair)                       | 0.26               | 0.20               | 0.24               | 0.13       |
|           | Success Rate<br>(After Pattern Repair)                        | 0.51               | 0.31               | 0.32               | 0.36       |
|           | Valid HeteroGraphs<br>per API call<br>(Before Pattern Repair) | 0.78               | 0.59               | 0.71               | 0.40       |
|           | Valid HeteroGraphs <br>per API call<br>(After Pattern Repair) | 1.54               | 0.92               | 0.95               | 1.07       |
| Total     | API calls                                                     | 1464               | 2233               | 2285               | 1325       |
|           | Success Rate<br>(Before Pattern Repair)                       | 0.78               | 0.67               | 0.70               | 0.41       |
|           | Success Rate<br>(After Pattern Repair)                        | 0.88               | 0.90               | 0.90               | 0.76       |
|           | Valid HeteroGraphs<br>per API call<br>(Before Pattern Repair) | 1.64               | 1.80               | 1.83               | 0.74       |
|           | Valid HeteroGraphs <br>per API call<br>(After Pattern Repair) | 1.86               | 2.42               | 2.37               | 1.37       |





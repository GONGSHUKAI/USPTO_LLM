# USPTO_LLM

### Preview
 Transform natural-language-described chemical reaction process into standardized format with LLM

### 使用说明

##### Code 

Code文件夹下相关的代码文件为以下四个: 

1. `api_request_parallel_processor.py` 调用API生成反应流程图(异质图格式)。
	使用gpt-4 API

2. `generate_request.py` 根据`uspto_full.json`中的反应数据，结合`FIXED_PROMPT`生成每次递交API的json文件 `uspto_request.json`

3. `get_gpt_answer.py` 根据API生成的`result_uspto_request.json`，将每个反应单独写入一个txt文件

4. `prompt_to_json.py` 将写的Prompt转化为json格式，并存入字符串变量`FIXED_PROMPT`中

运行顺序: `prompt_to_json.py` → `generate_request.py` → `api_request_parallel_processor.py` → `get_gpt_answer.py` 

##### Larrea

raw文件夹下的`rxn_idx.json`是`uspto_full.json`中反应的计数下标（即从第几个反应开始）；`uspto_full.json`是原始数据集；`uspto_onetime.json`暂未用刀

requests文件夹下的 `uspto_request.json`是`generate_request.py` 写的文件；`result_uspto_request.json`储存API的生成结果

##### Prompt

Prompt文件夹下储存`GONGSHUKAI`写的Prompt的.txt版本和.json版本

##### Training_Info

储存 `get_gpt_answer.py` 生成的各个反应。每个txt文件都包含一个化学反应描述和其对应的异质图(反应流程图)

### 当前运行效果

目前基本解决反应流程图格式正确的问题(很少出现反应条件和反应物/生成物错位，100个反应中有1-5个反应存在错位)

关于每个子步骤（substep）的分划，由于对反应理解的不同，即使是由人进行手动标注也会有不同的意见。我随机选择了API批量生成的几个反应流程图进行检查，基本符合反应逻辑。

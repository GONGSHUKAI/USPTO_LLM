# USPTO_LLM

### Preview
 Transform natural-language-described chemical reaction process into standardized format with LLM

### 使用说明

##### Code 

Code文件夹下相关的代码文件为以下四个: 

1. `api_request_parallel_processor.py` 调用API生成反应流程图(异质图格式)。
	使用gpt-4 API

2. `generate_request.py` 根据`uspto_full.json`中的反应数据，结合`FIXED_PROMPT`生成每次递交API的json文件 `uspto_request.json`

3. `extract_api_outcome.ipynb` 根据API生成的`result_uspto_request.json`，将每个反应写入`uspto_multiple_step.ipynb`中

4. `prompt_to_json.py` 将写的Prompt转化为json格式，并存入字符串变量`FIXED_PROMPT`中

运行顺序: `prompt_to_json.py` → `generate_request.py` → `api_request_parallel_processor.py` → `get_gpt_answer.py` 

##### Prompt

Prompt文件夹下储存`GONGSHUKAI`写的Prompt的.txt版本和.json版本

##### Training_Info

`comparison_LLM.xlsx`储存了不同模型的对比信息

### 当前运行效果
308376个反应中，生成有效反应266756个，占比86.5%

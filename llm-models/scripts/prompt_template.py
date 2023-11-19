# stanford_alpaca prompt template
stanford_alpaca_template_with_input = """{system_prompt}
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"\\n"""
stanford_alpaca_template = """{system_prompt}
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"\\n"""
stanford_alpaca_template_end = "### Response:\\n"

llama2_template_with_input = """<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

Input:
{input}


{instruction}
[/INST]\\n"""
llama2_template = """<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

{instruction}
[/INST]\\n"""
llama2_template_end = "[/INST]\\n"

ChatGLM3_template_with_input = """<|System|>:
{system_prompt}

<|user|>:
Input: {input}
{instruction}

<|assistant|>:\\n"""

ChatGLM3_template = """<|System|>:
{system_prompt}

<|user|>:
{instruction}

<|assistant|>:\\n"""
ChatGLM3_template_end = "<|assistant|>:\\n"

NousResearch_template_with_input = """### Instruction:
{system_prompt}
{instruction}
### Input:
{input}

### Response:\\n"""
NousResearch_template = """### Instruction:
{system_prompt}
{instruction}

### Response:\\n"""
NousResearch_template_end = "### Response:\\n"

prompt_template_dict = {
    "stanford_alpaca": stanford_alpaca_template,
    "stanford_alpaca_with_input": stanford_alpaca_template_with_input,
    "llama2": llama2_template,
    "llama2_with_input": llama2_template_with_input,
    "ChatGLM3": ChatGLM3_template,
    "ChatGLM3_with_input": ChatGLM3_template_with_input,
    "NousResearch": NousResearch_template,
    "NousResearch_with_input": NousResearch_template_with_input,
}

prompt_template_end_dict = {
    "stanford_alpaca": stanford_alpaca_template_end,
    "llama2": llama2_template_end,
    "ChatGLM3": ChatGLM3_template_end,
    "NousResearch": NousResearch_template_end,
}
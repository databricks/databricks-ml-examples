# Databricks notebook source
# MAGIC %md
# MAGIC # Code Llama 34B Inference on Databricks
# MAGIC
# MAGIC [Code Llama](https://huggingface.co/codellama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. It is trained with 2T tokens and supports context length window upto 10K tokens. The model is designed for general code synthesis and understanding.
# MAGIC
# MAGIC This notebook demonstrates how to use
# MAGIC - [CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf)
# MAGIC - [CodeLlama-34b-hf-instructions](https://huggingface.co/codellama/CodeLlama-34b-hf-instructions)
# MAGIC - [CodeLlama-34b-hf-python](https://huggingface.co/codellama/CodeLlama-34b-hf-python)
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.3 GPU ML Runtime
# MAGIC - Instance: `g5.12xlarge` on AWS (4x A10 GPUs), `Standard_NC24ads_A100_v4` on Azure (1x A100 80G GPU)
# MAGIC
# MAGIC **License**: A custom commercial license is available at: https://ai.meta.com/resources/models-and-libraries/llama-downloads/

# COMMAND ----------

# MAGIC %pip install -U transformers==4.33.3
# MAGIC %pip install -U flash-attn==2.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

# MAGIC %md
# MAGIC ## CodeLlama-34b-hf Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.
# MAGIC
# MAGIC Model capabilities:
# MAGIC
# MAGIC * &#9745; Code completion.
# MAGIC * &#9745; Infilling.
# MAGIC * &#9744; Instructions / chat.
# MAGIC * &#9744; Python specialist.

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of CodeLlama-34b-hf in https://huggingface.co/codellama/CodeLlama-34b-hf/commits/main
model = "codellama/CodeLlama-34b-hf"
revision = "fda69408949a7c6689a3cf7e93e632b8e70bb8ad"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision=revision,
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# Define parameters to generate text
def gen_text(full_prompts, pipeline, use_template=False, **kwargs):
    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on a single input

# COMMAND ----------

results = gen_text(["import socket\n\ndef ping_exponential_backoff(host: str):"], pipeline)
print(results[0])

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control the code generation
results = gen_text(["import socket\n\ndef ping_exponential_backoff(host: str):"], pipeline, temperature=0.5, max_new_tokens=200)
print(results[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference

# COMMAND ----------

# From openai_humaneval
inputs = [
  'from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """ ',
  'from typing import List def separate_paren_groups(paren_string: str) -> List[str]: """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other Ignore any spaces in the input string. >>> separate_paren_groups("( ) (( )) (( )( ))") ["()", "(())", "(()())"] """ ',
  ' def truncate_number(number: float) -> float: """ Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part of the number. >>> truncate_number(3.5) 0.5 """ ',
  'from typing import List def below_zero(operations: List[int]) -> bool: """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account fallls below zero, and at that point function should return True. Otherwise it should return False. >>> below_zero([1, 2, 3]) False >>> below_zero([1, 2, -4, 5]) True """ '
]

# COMMAND ----------

# Set batch size
results = gen_text(inputs, pipeline, batch_size=4)

for output in results:
  print(output)
  print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measure inference speed
# MAGIC Text generation speed is often measured with token/s, which is the average number of tokens that are generated by the model per second.
# MAGIC

# COMMAND ----------

import time
import logging


def get_num_tokens(text):
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    return inputs.shape[1]
  

def get_gen_text_throughput(full_prompt, pipeline, tokenizer, use_template=True, **kwargs):
    """
    Return tuple ( number of tokens / sec, num tokens, output ) of the generated tokens
    """

    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    kwargs.update(
        {
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_tensors": True,  # make the pipeline return token ids instead of decoded text to get the number of generated tokens
        }
    )

    num_input_tokens = get_num_tokens(full_prompt)

    # measure the time it takes for text generation
    start = time.time()
    outputs = pipeline(full_prompt, **kwargs)
    duration = time.time() - start

    # get the number of generated tokens
    n_tokens = len(outputs[0]["generated_token_ids"])

    # show the generated text in logging
    result = tokenizer.batch_decode(
        outputs[0]["generated_token_ids"][num_input_tokens:], skip_special_tokens=True
    )
    result = "".join(result)

    return (n_tokens / duration, n_tokens, result)

# COMMAND ----------

throughput, n_tokens, result = get_gen_text_throughput("import socket\n\ndef ping_exponential_backoff(host: str):", pipeline, tokenizer)

print(f"{throughput} tokens/sec, {n_tokens} tokens (including full prompt)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CodeLlama-34b-Instruct-hf Inference
# MAGIC
# MAGIC Model capabilities:
# MAGIC
# MAGIC * &#9745; Code completion.
# MAGIC * &#9745; Infilling.
# MAGIC * &#9745; Instructions / chat.
# MAGIC * &#9744; Python specialist.

# COMMAND ----------

# Restart the Python repl to release the occupied GPU memory.
%pip install -U transformers==4.33.3
%pip install -U flash-attn==2.3.0
dbutils.library.restartPython()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of CodeLlama-34b-Instruct-hf in https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/commits/main
model = "codellama/CodeLlama-34b-Instruct-hf"
revision = "38a1e15d8524a1f0a7760a7acf8242b81ae4eb87"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision=revision,
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

PROMPT_TEMPLATE = """
[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]
"""
# Define parameters to generate text
def gen_text(prompts, pipeline, use_template=False, **kwargs):
    full_prompts = [
            PROMPT_TEMPLATE.format(prompt=prompt)
            for prompt in prompts
        ]
    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on a single input

# COMMAND ----------

results = gen_text(["import socket\n\ndef ping_exponential_backoff(host: str):"], pipeline)
print(results[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference

# COMMAND ----------

# From openai_humaneval
inputs = [
  'from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """ ',
  'from typing import List def separate_paren_groups(paren_string: str) -> List[str]: """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other Ignore any spaces in the input string. >>> separate_paren_groups("( ) (( )) (( )( ))") ["()", "(())", "(()())"] """ ',
  ' def truncate_number(number: float) -> float: """ Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part of the number. >>> truncate_number(3.5) 0.5 """ ',
  'from typing import List def below_zero(operations: List[int]) -> bool: """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account fallls below zero, and at that point function should return True. Otherwise it should return False. >>> below_zero([1, 2, 3]) False >>> below_zero([1, 2, -4, 5]) True """ '
]

# Set batch size
results = gen_text(inputs, pipeline, batch_size=4)

for output in results:
  print(output)
  print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measure inference speed

# COMMAND ----------

import time
import logging


def get_num_tokens(text):
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    return inputs.shape[1]
  

def get_gen_text_throughput(prompt, pipeline, tokenizer, use_template=True, **kwargs):
    """
    Return tuple ( number of tokens / sec, num tokens, output ) of the generated tokens
    """

    full_prompt = PROMPT_TEMPLATE.format(prompt=prompt)

    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    kwargs.update(
        {
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_tensors": True,  # make the pipeline return token ids instead of decoded text to get the number of generated tokens
        }
    )

    num_input_tokens = get_num_tokens(full_prompt)

    # measure the time it takes for text generation
    start = time.time()
    outputs = pipeline(full_prompt, **kwargs)
    duration = time.time() - start

    # get the number of generated tokens
    n_tokens = len(outputs[0]["generated_token_ids"])

    # show the generated text in logging
    result = tokenizer.batch_decode(
        outputs[0]["generated_token_ids"][num_input_tokens:], skip_special_tokens=True
    )
    result = "".join(result)

    return (n_tokens / duration, n_tokens, result)

# COMMAND ----------

throughput, n_tokens, result = get_gen_text_throughput("import socket\n\ndef ping_exponential_backoff(host: str):", pipeline, tokenizer)

print(f"{throughput} tokens/sec, {n_tokens} tokens (including full prompt)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CodeLlama-34b-Python-hf Inference
# MAGIC
# MAGIC Model capabilities:
# MAGIC
# MAGIC * &#9745; Code completion.
# MAGIC * &#9744; Infilling.
# MAGIC * &#9744; Instructions / chat.
# MAGIC * &#9745; Python specialist.

# COMMAND ----------

# Restart the Python repl to release the occupied GPU memory.
%pip install -U transformers==4.33.3
%pip install -U flash-attn==2.3.0
dbutils.library.restartPython()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of CodeLlama-34b-Python-hf in https://huggingface.co/codellama/CodeLlama-34b-Python-hf/commits/main
model = "codellama/CodeLlama-34b-Python-hf"
revision = "a998a81ca5b57a10404d4615e85ff3b6d62ae649"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision=revision,
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# Define parameters to generate text
def gen_text(full_prompts, pipeline, use_template=False, **kwargs):
    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on a single input

# COMMAND ----------

results = gen_text(["import socket\n\ndef ping_exponential_backoff(host: str):"], pipeline)
print(results[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference

# COMMAND ----------

# From openai_humaneval
inputs = [
  'from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """ ',
  'from typing import List def separate_paren_groups(paren_string: str) -> List[str]: """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other Ignore any spaces in the input string. >>> separate_paren_groups("( ) (( )) (( )( ))") ["()", "(())", "(()())"] """ ',
  ' def truncate_number(number: float) -> float: """ Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part of the number. >>> truncate_number(3.5) 0.5 """ ',
  'from typing import List def below_zero(operations: List[int]) -> bool: """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account fallls below zero, and at that point function should return True. Otherwise it should return False. >>> below_zero([1, 2, 3]) False >>> below_zero([1, 2, -4, 5]) True """ '
]

# Set batch size
results = gen_text(inputs, pipeline, batch_size=4)

for output in results:
  print(output)
  print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measure inference speed

# COMMAND ----------

import time
import logging


def get_num_tokens(text):
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    return inputs.shape[1]
  

def get_gen_text_throughput(full_prompt, pipeline, tokenizer, use_template=True, **kwargs):
    """
    Return tuple ( number of tokens / sec, num tokens, output ) of the generated tokens
    """

    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    kwargs.update(
        {
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_tensors": True,  # make the pipeline return token ids instead of decoded text to get the number of generated tokens
        }
    )

    num_input_tokens = get_num_tokens(full_prompt)

    # measure the time it takes for text generation
    start = time.time()
    outputs = pipeline(full_prompt, **kwargs)
    duration = time.time() - start

    # get the number of generated tokens
    n_tokens = len(outputs[0]["generated_token_ids"])

    # show the generated text in logging
    result = tokenizer.batch_decode(
        outputs[0]["generated_token_ids"][num_input_tokens:], skip_special_tokens=True
    )
    result = "".join(result)

    return (n_tokens / duration, n_tokens, result)

# COMMAND ----------

throughput, n_tokens, result = get_gen_text_throughput("import socket\n\ndef ping_exponential_backoff(host: str):", pipeline, tokenizer)

print(f"{throughput} tokens/sec, {n_tokens} tokens (including full prompt)")

# COMMAND ----------



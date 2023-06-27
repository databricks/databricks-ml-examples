# Databricks notebook source
# MAGIC %md
# MAGIC # Load and Inference MPT-7B model on Databricks
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS

# COMMAND ----------

# MAGIC %pip install xformers einops flash-attn==v1.0.3.post0 triton==2.0.0.dev20221202

# COMMAND ----------

import transformers
import torch

# COMMAND ----------

name = 'mosaicml/mpt-7b-instruct'
config = transformers.AutoConfig.from_pretrained(
  name,
  trust_remote_code=True
)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda'

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  cache_dir="/local_disk0/.cache/huggingface/"
)

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")

generator = transformers.pipeline("text-generation",
                                  model=model, 
                                  config=config, 
                                  tokenizer=tokenizer,
                                  torch_dtype=torch.float16,
                                  device=0)

# COMMAND ----------

def generate_text(prompt, **kwargs):
  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 512
  
  kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
  
  template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
  if isinstance(prompt, str):
    full_prompt = template.format(instruction=prompt)
    generated_text = generator(full_prompt, **kwargs)
    generated_text = generated_text[0]["generated_text"]
  elif isinstance(prompt, list):
    full_prompts = list(map(lambda promp: template.format(instruction=promp), prompt))
    outputs = generator(full_prompts, **kwargs)
    generated_text = [out[0]["generated_text"] for out in outputs]
  return generated_text

# COMMAND ----------

question = """
What is a large language model?
"""
print(generate_text(question))

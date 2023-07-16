# Databricks notebook source
# MAGIC %md
# MAGIC # Load and Inference MPT-30B-instruct model
# MAGIC
# MAGIC MPT-30B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. It includes an 8k token context window. It supports for context-length extrapolation via ALiBi. The size of MPT-30B was also specifically chosen to make it easy to deploy on a single GPU—either 1xA100-80GB in 16-bit precision or 1xA100-40GB in 8-bit precision.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `Standard_NC24ads_A100_v4` on Azure

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required libraries

# COMMAND ----------

# Skip this step if running on Databricks runtime 13.2 GPU and above.
!wget -O /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton fastertransformer
# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC
# MAGIC Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. This is because MosaicML uses a custom model architecture that is not yet part of the transformers package.

# COMMAND ----------

import transformers
from transformers import AutoTokenizer, pipeline
import torch

# COMMAND ----------

name = "mosaicml/mpt-30b-instruct"
revision = "2abf1163dd8c9b11f07d805c06e6ec90a1f2037e"

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 16384
config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config.init_device = 'cuda' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True,
  revision=revision,
  cache_dir="/local_disk0/.cache/huggingface/",
  device_map = 'auto',
)


tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-30b')

generator = pipeline("text-generation",
                     model=model, 
                     config=config, 
                     tokenizer=tokenizer,
                     torch_dtype=torch.bfloat16)

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
    generated_text = generator(full_prompt, **kwargs)[0]["generated_text"]
  elif isinstance(prompt, list):
    full_prompts = list(map(lambda promp: template.format(instruction=promp), prompt))
    outputs = generator(full_prompts, **kwargs)
    generated_text = [out[0]["generated_text"] for out in outputs]
  return generated_text

# COMMAND ----------

generate_text("Tell me a funny joke.\nDon't make it too funny though.", temperature=0.5, max_new_tokens=1024)

# COMMAND ----------

generate_text(["What is ML?", "Name 10 colors"], max_new_tokens=100)

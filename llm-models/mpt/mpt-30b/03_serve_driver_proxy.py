# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving MPT-30B-instruct with a cluster driver proxy app
# MAGIC
# MAGIC This notebook enables you to run MPT-30B-Instruct on a Databricks cluster and expose the model to LangChain via [driver proxy](https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS or `Standard_NC24ads_A100_v4` on Azure
# MAGIC
# MAGIC GPU instances that have at least 2 A10 GPUs would be enough for inference on single input (batch inference requires slightly more memory).
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
# MAGIC Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. This is because MosaicML uses a custom model architecture that is not yet part of the transformers package.

# COMMAND ----------

import transformers
from transformers import AutoTokenizer, pipeline
import torch

# COMMAND ----------

# Load model to text generation pipeline
name = "mosaicml/mpt-30b-instruct"
revision = "2abf1163dd8c9b11f07d805c06e6ec90a1f2037e"

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 16384
config.attn_config['attn_impl'] = 'triton' # change this to use triton-based FlashAttention
config.init_device = 'cuda' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  revision=revision,
  cache_dir="/local_disk0/.cache/huggingface/"
  device_map = 'auto',
)

tokenizer = transformers.AutoTokenizer.from_pretrained('mosaicml/mpt-30b')

pipeline = transformers.pipeline("text-generation",
                                  model=model, 
                                  config=config, 
                                  tokenizer=tokenizer,
                                  torch_dtype=torch.bfloat16)

# COMMAND ----------

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

# Define parameters to generate text
def gen_text_for_serving(prompt, **kwargs):
    prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    return pipeline(prompt, **kwargs)[0]['generated_text']

# COMMAND ----------

print(gen_text_for_serving("How to master Python in 3 days?"))

# COMMAND ----------

# See full list of configurable args: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
print(gen_text_for_serving("How to master Python in 3 days?", temperature=0.1, max_new_tokens=100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("mpt-30b-instruct")

@app.route('/', methods=['POST'])
def serve_falcon_30b_instruct():
  resp = gen_text_for_serving(**request.json)
  return jsonify(resp)

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Keep `app.run` running, and it could be used with Langchain ([documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app)), or by call the serving endpoint with:
# MAGIC ```python
# MAGIC import requests
# MAGIC import json
# MAGIC
# MAGIC def request_mpt_30b(prompt, temperature=1.0, max_new_tokens=1024):
# MAGIC   token = ... # TODO: fill in with your Databricks personal access token that can access the cluster that runs this driver proxy notebook
# MAGIC   url = ...   # TODO: fill in with the driver_proxy_api output above
# MAGIC   
# MAGIC   headers = {
# MAGIC       "Content-Type": "application/json",
# MAGIC       "Authentication": f"Bearer {token}"
# MAGIC   }
# MAGIC   data = {
# MAGIC     "prompt": prompt,
# MAGIC     "temperature": temperature,
# MAGIC     "max_new_tokens": max_new_tokens,
# MAGIC   }
# MAGIC
# MAGIC   response = requests.post(url, headers=headers, data=json.dumps(data))
# MAGIC   return response.text
# MAGIC
# MAGIC
# MAGIC request_mpt_30b("What is databricks?")
# MAGIC ```

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------



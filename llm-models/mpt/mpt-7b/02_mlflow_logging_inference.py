# Databricks notebook source
# MAGIC %md
# MAGIC # Manage MPT-7B-instruct model with MLFlow on Databricks
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# # Skip this step if running on Databricks runtime 13.2 GPU and above.
# !wget -O /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
#   dpkg -i /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
#   wget -O /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
#   dpkg -i /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
#   wget -O /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
#   dpkg -i /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
#   wget -O /local_disk0/tmp/libcurand-11-7_10.2.10.91-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcurand-11-7_10.2.10.91-1_amd64.deb && \
#   dpkg -i /local_disk0/tmp/libcurand-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.6.0"
# MAGIC %pip install --upgrade "transformers==4.32.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model to MLFlow

# COMMAND ----------

# Define prompt template

def build_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    return f"""{INTRO_BLURB}
    {INSTRUCTION_KEY}
    {instruction}
    {RESPONSE_KEY}
    """

# COMMAND ----------

# MAGIC %md
# MAGIC Download the model

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mosaicml/mpt-7b-instruct"
revision = "bbe7a55d70215e16c00c1825805b81e4badb57d7"

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16, cache_dir="/local_disk0/.cache/huggingface/")
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# COMMAND ----------

# MAGIC %md
# MAGIC Log the model to MLFlow

# COMMAND ----------

import mlflow
import transformers
import accelerate
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define model signature including params
input_example = {"prompt": build_prompt("What is Machine Learning?")}
inference_config = {
  "temperature": 1.0,
  "max_new_tokens": 100,
  "do_sample": True,
}
signature = infer_signature(
  model_input=input_example,
  model_output="Machien Learning is...",
  params=inference_config
)

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 5 minutes to complete
torch_version = torch.__version__.split("+")[0]
with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
        transformers_model={
        "model": model,
        "tokenizer": tokenizer,
        },
        task = "text-generation",
        artifact_path="model",
        pip_requirements=[f"torch=={torch_version}", 
                          f"transformers=={transformers.__version__}", 
                          f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature,
        # Add the metadata task so that the model serving endpoint created later will be optimized
        metadata={"task": "llm/v1/completions"}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model

# COMMAND ----------

# Register model in MLflow Model Registry
# This may take about 6 minutes to complete
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name="mpt-7b-instruct",
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model from model registry
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------

import mlflow
import pandas as pd
loaded_model = mlflow.pyfunc.load_model(f"models:/mpt-7b-instruct/latest")

# Make a prediction using the loaded model
print(loaded_model.predict(
    {"prompt": build_prompt("what is ML?")}, 
    params={
        "temperature": 0.5,
        "max_new_tokens": 100,
    }
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Optimized Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the MPT-7B-Instruct model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, see the documentation([AWS](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints#gpu)). The feature is in Public Preview.
# MAGIC
# MAGIC Models in MPT family are supported for Optimized LLM Serving, which provides an order of magnitute better throughput and latency improvement. For more information, see the documentation([AWS](https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/llm-optimized-model-serving)). In this section, the endpoint will have optimized LLM serving enabled by default. To disable it, remove the `metadata = {"task": "llm/v1/completions"}` when calling `log_model` and run the notebook again.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'mpt-7b-instruct-example'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
# Choose `GPU_MEDIUM` on AWS, and `GPU_LARGE` on Azure
workload_type = "GPU_MEDIUM"

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": workload_type,
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------



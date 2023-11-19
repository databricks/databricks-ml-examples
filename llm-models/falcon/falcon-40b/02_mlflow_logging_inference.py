# Databricks notebook source
# MAGIC %md
# MAGIC # Manage `falcon-40b-instruct` model with MLFlow on Databricks
# MAGIC
# MAGIC The [falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) Large Language Model (LLM) is a instruct fine-tuned version of the [falcon-40b](https://huggingface.co/tiiuae/falcon-40b) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance:
# MAGIC     - `g5.12xlarge` on aws
# MAGIC     - `Standard_NC24ads_A100_v4` on azure
# MAGIC     - `g2-standard-48` on gcp


# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# MAGIC %pip install -U "mlflow-skinny[databricks]>=2.6.0"
# MAGIC %pip install -U  torch==2.1.0  torchvision==0.16.0  torchvision==0.15.2  transformers==4.35.0  accelerate==0.24.1  einops==0.7.0  sentencepiece==0.1.99 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of `falcon-40b-instruct`. in https://huggingface.co/tiiuae/falcon-40b-instruct/commits/main
model = "tiiuae/falcon-40b-instruct"
revision = "ecb78d97ac356d098e79f0db222c9ce7c5d9ee5f"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16,
                                             cache_dir="/local_disk0/.cache/huggingface/")
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def build_prompt(instruction):
    return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""


# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

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
with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        task="text-generation",
        artifact_path="model",
        pip_requirements=["torch==2.0.1", "transformers==4.34.0", "accelerate==0.21.0", "torchvision==0.15.2"],
        input_example=input_example,
        signature=signature,
        # Add the metadata task so that the model serving endpoint created later will be optimized
        metadata={
            "task": "llm/v1/completions",
            "databricks_model_source": "example-notebooks",
            "databricks_model_size_parameters": "40b"
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC  By default, MLflow registers models in the Databricks workspace model registry. To register models in Unity Catalog instead, we follow the [documentation](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html) and set the registry server as Databricks Unity Catalog.
# MAGIC
# MAGIC  In order to register a model in Unity Catalog, there are [several requirements](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#requirements), such as Unity Catalog must be enabled in your workspace.
# MAGIC

# COMMAND ----------

# Configure MLflow Python client to register model in Unity Catalog
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Register model to Unity Catalog
# This may take 2 minutes to complete

registered_name = "models.default.falcon-40b-instruct"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Choose the right model version registered in the above cell.
client.set_registered_model_alias(name=registered_name, alias="Champion", version=result.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model from Unity Catalog

# COMMAND ----------

import mlflow

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
    {"prompt": "What is large language model?"},
    params={
        "temperature": 0.5,
        "max_new_tokens": 100,
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Optimized Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `Mistral-7B-Instruct` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, see the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu). The feature is in Public Preview.
# MAGIC
# MAGIC Models in Mistral family are supported for Optimized LLM Serving, which provides an order of magnitude better throughput and latency improvement. For more information, see the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html). In this section, the endpoint will have optimized LLM serving enabled by default. To disable it, remove the `metadata = {"task": "llm/v1/completions"}` when calling `log_model` and run the notebook again.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'falcon-40b-instruct'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
served_name = f'{model_version.name.replace(".", "_")}_{model_version.version}'

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
# Choose:
#    - MULTIGPU_MEDIUM on aws
#    - GPU_LARGE on azure

workload_type = "GPU_MEDIUM"

endpoint_config = {
    "name": endpoint_name,
    "config": {
        "served_models": [{
            "name": served_name,
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
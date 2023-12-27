# Databricks notebook source
# MAGIC %md
# MAGIC # Manage Llama 2 7B chat model with MLFlow on Databricks
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.6.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# It is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main
model_name = "meta-llama/Llama-2-7b-chat-hf"
revision = "08751db2aca9bf2f7f80d2e516117a53d7450235"

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
    model_output="Machine Learning is...",
    params=inference_config
)

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 1.7 minutes to complete
with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        task="text-generation",
        artifact_path="model",
        pip_requirements=["torch", "transformers", "accelerate", "sentencepiece"],
        input_example=input_example,
        signature=signature,
        # Add the metadata task so that the model serving endpoint created later will be optimized
        metadata={"task": "llm/v1/completions"}
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
# This may take 2.2 minutes to complete

registered_name = "models.default.llama2_7b_completions"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

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
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `LLaMAV2-7b` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, see the documentation([AWS](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints#gpu)). The feature is in Public Preview.
# MAGIC
# MAGIC Models in LLaMA-V2 family are supported for Optimized LLM Serving, which provides an order of magnitude better throughput and latency improvement. For more information, see the documentation([AWS](https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/llm-optimized-model-serving)). In this section, the endpoint will have optimized LLM serving enabled by default. To disable it, remove the `metadata = {"task": "llm/v1/completions"}` when calling `log_model` and run the notebook again.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'llama2-7b-completions'

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
workload_type = "GPU_LARGE"

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

# Databricks notebook source
# MAGIC %md
# MAGIC # Manage `mpt-7b-8k-instruct` model with MLFlow on Databricks
# MAGIC
# MAGIC The [mpt-7b-8k-instruct](https://huggingface.co/mosaicml/mpt-7b-8k-instruct) Large Language Model (LLM) is a instruct fine-tuned version of the [mpt-7b-8k](https://huggingface.co/mosaicml/mpt-7b-8k) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.3 GPU ML Runtime
# MAGIC - Instance:
# MAGIC     - `g5.xlarge` on aws
# MAGIC     - `Standard_NV36ads_A10_v5` on azure
# MAGIC     - `g2-standard-4` on gcp
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# MAGIC %pip install -U mlflow-skinny[databricks]>=2.6.0
# MAGIC %pip install -U  torch==2.0.1+cu118  torchvision==0.15.2+cu118  transformers==4.37.2  accelerate==0.26.1  einops==0.7.0  flash-attn==2.5.2 
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of `mpt-7b-8k-instruct`. in https://huggingface.co/mosaicml/mpt-7b-8k-instruct/commits/main
model_name = "mosaicml/mpt-7b-8k-instruct"
revision = "fa099ce469116153c8c0238c1d220c01e871a992"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                             cache_dir="/local_disk0/.cache/huggingface/")
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def build_prompt(instruction):
    return """
### Instruction:
{system_prompt}
{instruction}

### Response:\n""".format(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        instruction=instruction
    )

# COMMAND ----------

# DBTITLE 1,Machine Learning Model Logger
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
        pip_requirements=["torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "transformers==4.37.2", "accelerate==0.26.1", "einops==0.7.0", "flash-attn==2.5.2"],
        input_example=input_example,
        signature=signature,
        # Add the metadata task so that the model serving endpoint created later will be optimized
        metadata={
            "task": "llm/v1/completions",
            "databricks_model_source": "example-notebooks",
            "databricks_model_size_parameters": "7b"
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

registered_name = "models.default.mpt-7b-8k-instruct"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

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
# MAGIC ## Deploying the model to Model Serving
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `mpt-7b-8k-instruct` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, see the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu). The feature is in Public Preview.
# MAGIC
# MAGIC Models in `mpt-7b-8k-instruct` family are supported for Optimized LLM Serving, which provides an order of magnitude better throughput and latency improvement. 
# MAGIC You can deploy this model directly to Optimized LLM serving ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html#input-and-output-schema-format)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/llm-optimized-model-serving)) for improved throughput and latency.
# MAGIC Databricks recommends using the provisioned throughput ([AWS](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis)) experience for optimized inference of LLMs.

# COMMAND ----------

model_version = result  # the returned result of mlflow.register_model
served_name = f'{model_version.name.replace(".", "_")}_{model_version.version}'

# COMMAND ----------

# DBTITLE 1,Model Deployment Throughput Setter
import requests
import json

# To deploy your model in provisioned throughput mode via API, you must specify `min_provisioned_throughput` and `max_provisioned_throughput` fields in your request.
# Minimum desired provisioned throughput
min_provisioned_throughput = 980

# Maximum desired provisioned throughput
max_provisioned_throughput = 2940

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# send the POST request to create the serving endpoint
data = {
    "name": served_name,
    "config": {
        "served_models": [
            {
                "model_name": model_version.name,
                "model_version": model_version.version,
                "min_provisioned_throughput": min_provisioned_throughput,
                "max_provisioned_throughput": max_provisioned_throughput,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers
)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------



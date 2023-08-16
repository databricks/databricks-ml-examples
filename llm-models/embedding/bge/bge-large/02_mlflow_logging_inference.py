# Databricks notebook source
# MAGIC %md
# MAGIC # Manage `bge-large-en` model with MLFlow on Databricks
# MAGIC
# MAGIC [bge-large-en (BAAI General Embedding) model](https://huggingface.co/BAAI/bge-large-en) can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification, clustering, or semantic search. And it also can be used in vector database for LLMs.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.3 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure.
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

from sentence_transformers import SentenceTransformer
model_name = "BAAI/bge-large-en"

model = SentenceTransformer(model_name)

# COMMAND ----------

import mlflow
import pandas as pd

# Define input and output schema
sentences = ["This is an example sentence", "Each sentence is converted"]
signature = mlflow.models.infer_signature(
    sentences,
    model.encode(sentences),
)
with mlflow.start_run() as run:  
    mlflow.sentence_transformers.log_model(
      model, 
      "bge-embedding", 
      signature=signature,
      input_example=sentences)

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

registered_name = "models.default.bge-large-en" # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/bge-embedding",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# Choose the right model version registered in the above cell.
client.set_registered_model_alias(name=registered_name, alias="Champion", version=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model from Unity Catalog

# COMMAND ----------

import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
  ["What is ML?", "What is large language model?"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `bge-large-en` model.
# MAGIC

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'bge-embedding'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": "GPU_MEDIUM",
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

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain running in the same workspace.

# Databricks notebook source
# MAGIC %md
# MAGIC # Manage `bge-m3` model with MLFlow on Databricks
# MAGIC
# MAGIC In this example, we demonstrate how to log the [bge-m3 model](https://huggingface.co/BAAI/bge-m3) to MLFLow with the `sentence_transformers` flavor, manage the model with Unity Catalog, and create a model serving endpoint.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.3 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS, `Standard_NC4as_T4_v3` on Azure,  or `g2-standard-4` on GCP
# MAGIC

# COMMAND ----------

# Upgrade to use the newest Databricks SDK
%pip install --upgrade databricks-sdk
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

from sentence_transformers import SentenceTransformer
model_name = "BAAI/bge-m3"

model = SentenceTransformer(model_name)

# COMMAND ----------

# DBTITLE 1,Name: Model Logging with MLflow
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
      "bge-m3-embedding", 
      signature=signature,
      input_example=sentences,
      metadata={"source": "huggingface",
                "source_model_name": "bge-m3",
                "task": "llm/v1/embedding",
                "databricks_model_source": "databricks-ml-examples",
                "databricks_model_family": "XLMRobertaModel (bge-m3)",
                "databricks_model_size_parameters": "568M"
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

registered_name = "models.default.bge_m3" # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/bge-m3-embedding",
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
  ["What is ML?", "What is large language model?"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `bge-large-en` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team or sign up [here](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit).

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'bge-m3-embedding'

# COMMAND ----------

# DBTITLE 1,CodeSnipKeepEndpointsWithNameModel
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": result.name,
            "model_version": result.version,
            "workload_type": "GPU_SMALL",
            "workload_size": "Small",
            "scale_to_zero_enabled": "True",
        }
    ]
})
model_details = w.serving_endpoints.create(name=endpoint_name, config=config)
model_details.result(timeout=datetime.timedelta(minutes=40))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the text by querying the serving endpoint
# MAGIC With the newest Databricks SDK installed, query the serving endpoint as follows:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Change it to your own input
dataframe_records = [
    "MLFlow is tailored to assist ML practitioners throughout the various stages of ML development and deployment."
]

w = WorkspaceClient()
w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=dataframe_records,
)


# COMMAND ----------



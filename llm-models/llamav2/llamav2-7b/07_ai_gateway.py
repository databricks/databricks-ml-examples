# Databricks notebook source
# MAGIC %md
# MAGIC # Manage access to Databricks Serving Endpoint with AI Gateway
# MAGIC
# MAGIC This example notebook demonstrates how to use MLflow AI Gateway ([see announcement blog](https://www.databricks.com/blog/announcing-mlflow-ai-gateway)) with a Databricks Serving Endpoint.
# MAGIC
# MAGIC Requirement:
# MAGIC - A Databricks serving endpoint that is in the "Ready" status. Please refer to the `02_mlflow_logging_inference` example notebook for steps to create a Databricks serving endpoint.
# MAGIC
# MAGIC Environment:
# MAGIC - MLR: 13.3 ML
# MAGIC - Instance: `i3.xlarge` on AWS, `Standard_DS3_v2` on Azure

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[gateway]>=2.6"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# TODO: Please change endpoint_name to your Databricks serving endpoint name if it's different
# The below assumes you've create an endpoint "llama2-7b-chat" according to 02_mlflow_logging_inference
endpoint_name = "llama2-7b-chat"
gateway_route_name = f"{endpoint_name}_completion"

# COMMAND ----------

# Databricks URL and token that would be used to route the Databricks serving endpoint
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import mlflow.gateway
mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an AI Gateway Route

# COMMAND ----------

mlflow.gateway.create_route(
    name=gateway_route_name,
    route_type="llm/v1/completions",
    model= {
        "name": endpoint_name, 
        "provider": "databricks-model-serving",
        "databricks_model_serving_config": {
          "databricks_api_token": token,
          "databricks_workspace_url": databricks_url
        }
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query an AI Gateway Route
# MAGIC The below code uses `mlflow.gateway.query` to query the `Route` created in the above cell.
# MAGIC
# MAGIC Note that `mlflow.gateway.query` doesn't need to be run in the same notebook nor the same cluster, and it doesn't require the Databricks URL or API token to query it, which makes it convenient for multiple users within the same organization to access a served model.

# COMMAND ----------

response = mlflow.gateway.query(
    route=gateway_route_name,
    data={"prompt": "What is MLflow?", "temperature": 0.3, "max_new_tokens": 200}
)

print(response['candidates'][0]['text'])

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
# The below assumes you've create an endpoint "mistral-7b-instruct " according to 02_mlflow_logging_inference
endpoint_name = "mistral-7b-instruct"
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

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def build_prompt(instruction):
    return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

# COMMAND ----------

response = mlflow.gateway.query(
    route=gateway_route_name,
    data={"prompt": build_prompt("What is MLflow?"), "temperature": 0.3, "max_tokens": 512}
)

print(response['candidates'][0]['text'])

# COMMAND ----------



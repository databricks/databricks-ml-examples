# Databricks notebook source
# MAGIC %md
# MAGIC # Load Mistral-7B-Instruct as chat completion from LangChain on Databricks
# MAGIC
# MAGIC This example notebook shows how to wrap Databricks endpoints as LLMs in LangChain. It supports two endpoint types:
# MAGIC
# MAGIC - Serving endpoint, recommended for production and development. See `02_[chat]_mlflow_logging_inference` for how to create one.
# MAGIC - Cluster driver proxy app, recommended for iteractive development. See `03_[chat]_serve_driver_proxy` for how to create one.
# MAGIC
# MAGIC Environment tested:
# MAGIC - MLR: 14.0 ML
# MAGIC - Instance:
# MAGIC   - Wrapping a serving endpoint: `i3.xlarge` on AWS, `Standard_DS3_v2` on Azure
# MAGIC   - Wrapping a cluster driver proxy app: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure (same instance as the driver proxy app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrapping Databricks endpoints as LLMs in LangChain
# MAGIC - If the model is a serving endpoint, it requires a model serving endpoint (see `02_[chat]_mlflow_logging_inference` for how to create one) to be in the "Ready" state.
# MAGIC - If the model is a cluster driver proxy app, it requires the driver proxy app of the `03_[chat]_serve_driver_proxy` example notebook running.
# MAGIC   - If running a Databricks notebook attached to the same cluster that runs the app, you only need to specify the driver port to create a `Databricks` instance.
# MAGIC   - If running on different cluster, you can manually specify the cluster ID to use, as well as Databricks workspace hostname and personal access token.

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.llms import Databricks
def transform_input(**request):
    request["prompt"] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request["prompt"]},
        ]
    request["stop"] = []
    return request
  
def transform_output(response):
    #Extract the answer from the responses.
    return response[0]["candidates"][0]["message"]["content"]


# COMMAND ----------

# If using serving endpoint, the model serving endpoint is created in `02_[chat]_mlflow_logging_inference`
# llm = Databricks(endpoint_name='llama2-7b-chat',
#                  transform_input_fn=transform_input,
#                  transform_output_fn=transform_output,)

# If the model is a cluster driver proxy app on the same cluster, you only need to specify the driver port.
llm = Databricks(cluster_driver_port="7777",
                 transform_input_fn=transform_input,
                 transform_output_fn=transform_output,)

# If the model is a cluster driver proxy app on the different cluster, you need to provide the cluster id
# llm = Databricks(cluster_id="0000-000000-xxxxxxxx"
#                  cluster_driver_port="7777",
#                  transform_input_fn=transform_input,
#                  transform_output_fn=transform_output,)

print(llm("How to master Python in 3 days?"))

# COMMAND ----------



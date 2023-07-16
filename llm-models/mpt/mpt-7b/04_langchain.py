# Databricks notebook source
# MAGIC %md
# MAGIC # Load MPT-7B model from LangChain on Databricks
# MAGIC
# MAGIC This example notebook shows how to wrap Databricks endpoints as LLMs in LangChain.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS

# COMMAND ----------

# MAGIC %pip install -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.llms import Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrapping a cluster driver proxy app with Langchain
# MAGIC
# MAGIC Prerequisites:
# MAGIC Run `03_serve_driver_proxy` to loaded `MPT-7B-Instruct` model on a Databricks interactive cluster in "single user" or "no isolation shared" mode.
# MAGIC - A local HTTP server running on the driver node to serve the model at "/" using HTTP POST with JSON input/output.
# MAGIC - It uses a port number `7777` in the notebook, you can change to any port number between [3000, 8000].
# MAGIC - You have "Can Attach To" permission to the cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC If running a Databricks notebook attached to the same cluster that runs the app, you only need to specify the driver port to create a `Databricks` instance.
# MAGIC

# COMMAND ----------

llm = Databricks(cluster_driver_port="7777")

print(llm("How to master Python in 3 days?"))

# COMMAND ----------

# MAGIC %md
# MAGIC If the driver proxy is running on different cluster, you can manually specify the cluster ID to use, as well as Databricks workspace hostname and personal access token.
# MAGIC ```
# MAGIC llm = Databricks(cluster_id="0000-000000-xxxxxxxx", cluster_driver_port="7777")
# MAGIC ```

# COMMAND ----------

# If the app accepts extra parameters like `temperature`,
# you can set them in `model_kwargs`.
llm = Databricks(cluster_driver_port="7777", model_kwargs={"temperature": 0.1})

print(llm("How to master Python in 3 days?"))

# COMMAND ----------

# Use `transform_input_fn` and `transform_output_fn` if the app
# expects a different input schema and does not return a JSON string,
# respectively, or you want to apply a prompt template on top.


def transform_input(**request):
    full_prompt = f"""{request["prompt"]}
    Be Concise.
    """
    request["prompt"] = full_prompt
    return request


def transform_output(response):
    return response.upper()


llm = Databricks(
    cluster_driver_port="7777",
    transform_input_fn=transform_input,
    transform_output_fn=transform_output,
)

print(llm("How to master Python in 3 days?"))

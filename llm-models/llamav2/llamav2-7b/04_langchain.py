# Databricks notebook source
# MAGIC %md
# MAGIC # Load Llama-2-7b-chat-hf from LangChain on Databricks
# MAGIC
# MAGIC This example notebook is adapts the [LangChain integration documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/databricks), and shows how to wrap Databricks cluster driver proxy app as LLMs in LangChain. See `03_serve_driver_proxy` for how to create one.
# MAGIC
# MAGIC Environment tested:
# MAGIC - MLR: 13.2 ML
# MAGIC - Instance:
# MAGIC   - `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure (same instance as the driver proxy app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrapping a cluster driver proxy app
# MAGIC The LangChain Databricks integration also works when given the port that runs a proxy.
# MAGIC
# MAGIC It requires the driver proxy app of the `03_serve_driver_proxy` example notebook running.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Same cluster
# MAGIC If using the same cluster that runs the `03_serve_driver_proxy` notebook, specifying `cluster_driver_port` is required.

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.llms import Databricks

# COMMAND ----------

llm = Databricks(cluster_driver_port="7777")

print(llm("How to master Python in 3 days?"))

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
  """
  Add more instructions into the prompt.
  """
  full_prompt = f"""[INST] Let's think step by step. [/INST]
  User: {request["prompt"]}
  """
  request["prompt"] = full_prompt
  return request


def transform_output(response):
  """
  Add timestamps for the anwsers.
  """
  from datetime import datetime
  now = datetime.now()
  current_time = now.strftime("%d/%m/%Y %H:%M:%S")
  return f"[{current_time}] llama2: {response}"


llm = Databricks(
    cluster_driver_port="7777",
    transform_input_fn=transform_input,
    transform_output_fn=transform_output,
)

print(llm("How to master Python in 3 days?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Different cluster
# MAGIC If using a different cluster, it's required to also specify `cluster_id`, which you can find in the cluster configuration page.

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.llms import Databricks

# TODO: this cluster ID is a place holder, please replace `cluster_id` with the actual cluster ID of the server proxy app's cluster
llm = Databricks(cluster_id="0714-163944-e7f4e8mo", cluster_driver_port="7777")

print(llm("How to master Python in 3 days?"))

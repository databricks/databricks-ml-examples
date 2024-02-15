# Databricks notebook source
# MAGIC %md
# MAGIC # Load `mpt-7b-8k-instruct` model from Langchain on Databricks
# MAGIC
# MAGIC This example notebook shows how to wrap Databricks endpoints as LLMs in LangChain.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# MAGIC %pip install -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrapping a serving endpoint with Langchain
# MAGIC Prerequisites:
# MAGIC - Run `02_mlflow_logging_inference` to deploy the model to a Databricks serving endpoint

# COMMAND ----------

from langchain_community.llms import Databricks

# If running a Databricks notebook attached to an interactive cluster in "single user"
# or "no isolation shared" mode, you only need to specify the endpoint name to create
# a `Databricks` instance to query a serving endpoint in the same workspace.

registered_name = "models_default_mpt-7b-8k-instruct_1"

llm = Databricks(endpoint_name=registered_name)

llm("How are you?")

# COMMAND ----------

# MAGIC %md
# MAGIC You can define `transform_input_fn` and `transform_output_fn` if the app
# MAGIC expects a different input schema and does not return a JSON string,
# MAGIC respectively, or you want to apply a prompt template on top.

# COMMAND ----------

def transform_input(**request):
    """
    Add more instructions into the prompt.
    """
    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    full_prompt = """### Instruction:
{system_prompt}
{instruction}

### Response:\n""".format(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        instruction=request["prompt"]
    )
    request["prompt"] = full_prompt
    return request


def transform_output(response):
    """
    Add timestamps for the anwsers.
    """
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    return f"[{current_time}] mpt: {response}"


llm = Databricks(
    endpoint_name=registered_name,
    transform_input_fn=transform_input,
    transform_output_fn=transform_output,
)

print(llm("How to master Python in 3 days?"))
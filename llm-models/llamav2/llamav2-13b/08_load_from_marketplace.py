# Databricks notebook source
# MAGIC %md
# MAGIC # Loading Llama 2 13B Chat model from Marketplace
# MAGIC
# MAGIC This example notebook demonstrates how to load the Llama 2 13B Chat model from a Databricks Marketplace's Catalog ([see announcement blog](https://www.databricks.com/blog/llama-2-foundation-models-available-databricks-lakehouse-ai)).
# MAGIC
# MAGIC Environment:
# MAGIC - MLR: 13.3 ML
# MAGIC - Instance: `g5.12xlarge` on AWS, `Standard_NV72ads_A10_v5` or `Standard_NC24ads_A100_v4` on Azure

# COMMAND ----------

# To access models in Unity Catalog, ensure that MLflow is up to date
%pip install --upgrade mlflow-skinny[databricks]
dbutils.library.restartPython()

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# TODO: Please replace catalog_name with the name of the catalog containing this model
catalog_name = "marketplace_llama_2_models"
version = 1

# Create a Spark UDF to generate the response to a prompt
generate = mlflow.pyfunc.spark_udf(
    spark, f"models:/{catalog_name}.models.llama_2_13b_chat_hf/{version}", "string"
)

# COMMAND ----------

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# COMMAND ----------

from typing import List
import pandas as pd


def gen_text(instructions: List[str]):
    prompts = [
        PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
        for instruction in instructions
    ]

    # `generate` is a Spark UDF that takes a string column as input
    df = spark.createDataFrame(pd.DataFrame({"text": pd.Series(prompts)}))
    generated_df = df.select(generate(df.text).alias("generated_text"))

    # Get the rows of the 'generated_text' column in the dataframe 'generated_df' as a list, and truncate the instruction
    generated_text_list = [
        str(row.generated_text).split("[/INST]\n")[1] for row in generated_df.collect()
    ]

    return generated_text_list

# COMMAND ----------

# To have more than 1 input sequences in the same batch for inference, more GPU memory would be needed; swap to more powerful GPUs if needed, or use Databricks Model Serving
gen_text(
    [
        "What is a large language model?",
        # "Write a short announcement of Llama 2 models in Databricks Marketplace.",
    ]
)

# Databricks notebook source
# MAGIC %md
# MAGIC # Loading `falcon-7b-instruct` model from Marketplace
# MAGIC
# MAGIC This example notebook demonstrates how to load the [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) model from a Databricks Marketplace's Catalog ([see announcement blog](https://www.databricks.com/blog/llama-2-foundation-models-available-databricks-lakehouse-ai)).
# MAGIC
# MAGIC Environment:
# MAGIC - MLR: 14.0 GPU ML Runtime and above
# MAGIC - Instance:
# MAGIC     - `g5.8xlarge` on aws
# MAGIC     - `Standard_NV36ads_A10_v5` on azure
# MAGIC     - `g2-standard-8` or `a2-highgpu-1g` on gcp


# COMMAND ----------

%pip install -U  torch==2.1.0  torchvision==0.16.0  torchvision==0.15.2  transformers==4.35.0  accelerate==0.24.1  einops==0.7.0  sentencepiece==0.1.99 
# To access models in Unity Catalog, ensure that MLflow is up to date
%pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
dbutils.library.restartPython()

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

catalog_name = "databricks_falcon-7b-instruct_models" # Default catalog name when installing the model from Databricks Marketplace
version = 1

# Create a Spark UDF to generate the response to a prompt
generate = mlflow.pyfunc.spark_udf(
    spark, f"models:/{catalog_name}.models.falcon-7b-instruct/{version}", "string"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The Spark UDF `generate` could inference on Spark DataFrames.

# COMMAND ----------

import pandas as pd

# To have more than 1 input sequences in the same batch for inference, more GPU memory would be needed; swap to more powerful GPUs (e.g. Standard_NC24ads_A100_v4 on Azure), or use Databricks Model Serving

df = spark.createDataFrame(
    pd.DataFrame(
        {
            "text": [
                "What is a large language model?",
            ]
        }
    )
)
display(df)

generated_df = df.select(generate(df.text).alias("generated_text"))
display(generated_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We could also wrap the Spark UDF into a function that takes system prompts, and takes lists of text strings as input/output.

# COMMAND ----------

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

PROMPT_FOR_GENERATION_FORMAT = """
### Instruction:
{system_prompt}
{instruction}

### Response:\n
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
        str(row.generated_text).split("### Response:\n")[1] for row in generated_df.collect()
    ]

    return generated_text_list

# COMMAND ----------

# To have more than 1 input sequences in the same batch for inference, more GPU memory would be needed; swap to more powerful GPUs (e.g. Standard_NC24ads_A100_v4 on Azure), or use Databricks Model Serving
gen_text(
    [
        "What is a large language model?",
    ]
)
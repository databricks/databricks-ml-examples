# Databricks notebook source
# MAGIC %md
# MAGIC # Manage Llama-2-70b-chat-hf model with MLFlow on Databricks
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) is the 70B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.2 GPU ML Runtime
# MAGIC - Instance: `Standard_NC48ads_A100_v4` on Azure, `p4d.24xlarge` on AWS
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Llama-2-70b-chat-hf in https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/commits/main
model_name = "meta-llama/Llama-2-70b-chat-hf"
revision = "e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def build_prompt(instruction):
    return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

# Define model signature including params
input_example = {"prompt": build_prompt("What is Machine Learning?")}
inference_config = {
    "temperature": 1.0,
    "max_new_tokens": 100,
    "do_sample": True,
}
signature = infer_signature(
    model_input=input_example,
    model_output="Machine Learning is...",
    params=inference_config
)

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 4.5 minutes to complete
with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        task="text-generation",
        artifact_path="model",
        pip_requirements=["torch", "transformers", "accelerate", "sentencepiece"],
        input_example=input_example,
        signature=signature,
        # Add the metadata task so that the model serving endpoint created from this MLflow model will be optimized
        metadata={"task": "llm/v1/completions"}
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
# This may take 12 minutes to complete

registered_name = "models.default.llamav2_70b_completions"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
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

mlflow.set_registry_uri("databricks-uc")

registered_name = "models.default.llamav2_70b_completions"

# COMMAND ----------

# DBTITLE 1,ML Model Predictor
import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "prompt": ["What is ML?", "What is large language model?"],
        "temperature": [0.1, 0.5],
        "max_new_tokens": [100, 100],
    }
)

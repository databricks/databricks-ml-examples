# Databricks notebook source
# MAGIC %md
# MAGIC # Load and Inference MPT-30B model with MLFlow on Databricks
# MAGIC
# MAGIC MPT-30B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. It includes an 8k token context window. It supports for context-length extrapolation via ALiBi. The size of MPT-30B was also specifically chosen to make it easy to deploy on a single GPU—either 1xA100-80GB in 16-bit precision or 1xA100-40GB in 8-bit precision.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `Standard_NC24ads_A100_v4` on Azure

# COMMAND ----------

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton fastertransformer
# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.6.0"
# MAGIC %pip install --upgrade "transformers==4.32.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model to MLFlow

# COMMAND ----------

# Define prompt template

def build_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    return f"""{INTRO_BLURB}
    {INSTRUCTION_KEY}
    {instruction}
    {RESPONSE_KEY}
    """

# COMMAND ----------

# MAGIC %md
# MAGIC Download the model

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mosaicml/mpt-30b-instruct"
revision = "2d1dde986c9737e0ef4fc2280ad264baf55ea1cd"

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16, cache_dir="/local_disk0/.cache/huggingface/")
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# COMMAND ----------

# MAGIC %md
# MAGIC Log the model to MLFlow

# COMMAND ----------

import mlflow
import transformers
import accelerate
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define model signature including params
input_example = {"prompt": build_prompt("What is Machine Learning?")}
inference_config = {
  "temperature": 1.0,
  "max_new_tokens": 100,
  "do_sample": True,
}
signature = infer_signature(
  model_input=input_example,
  model_output="Machien Learning is...",
  params=inference_config
)

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 20 minutes to complete
torch_version = torch.__version__.split("+")[0]
with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        artifact_path="model",
        task = "text-generation",
        pip_requirements=[f"torch=={torch_version}", 
                          f"transformers=={transformers.__version__}", 
                          f"accelerate=={accelerate.__version__}", "einops", "sentencepiece", "xformers"],
        input_example=input_example,
        signature=signature,
        # Add the metadata task so that the model serving endpoint created later will be optimized
        metadata={"task": "llm/v1/completions"}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model

# COMMAND ----------

# Register model in MLflow Model Registry
# This may take about 30 minutes to complete
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name="mpt-30b-instruct",
    await_registration_for=5000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model from model registry
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------

import mlflow
import pandas as pd
loaded_model = mlflow.pyfunc.load_model(f"models:/mpt-30b-instruct/latest")

# Make a prediction using the loaded model
print(loaded_model.predict(
    {"prompt": build_prompt("what is ML?")}, 
    params={
        "temperature": 0.5,
        "max_new_tokens": 100,
    }
))

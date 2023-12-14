# Databricks notebook source
# MAGIC %md
# MAGIC # Manage `Mixtral-8x7B-Instruct-v0.1` model with MLFlow on Databricks
# MAGIC
# MAGIC The [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) Large Language Model (LLM) is a instruct fine-tuned version of the [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.1 GPU ML Runtime
# MAGIC   - There could be CUDA incompatability issues to install and use vllm on 13.x GPU ML Runtime.
# MAGIC - Instance: `g5.48xlarge` on AWS (8xA10)

# COMMAND ----------

# MAGIC %pip install -U "mlflow-skinny[databricks]>=2.6.0"
# MAGIC %pip install -U vllm==0.2.4 transformers==4.36.0 megablocks==0.5.0 ray
# MAGIC %pip install -U databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Mixtral-8x7B-Instruct-v0.1 in https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/commits/mainn
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
revision = "3de0408ae8b591d9ac516a2384925dd98ebc66f4"

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
snapshot_location = snapshot_download(repo_id=model, revision=revision, cache_dir="/local_disk0/mixtral-8x7b")

# COMMAND ----------

import json
import mlflow
import torch

from vllm import LLM
from vllm import SamplingParams

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


class MixtralInstruct(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    """
    This method initializes the tokenizer and language model
    using the specified model repository.
    """
    self.llm = LLM(model=context.artifacts["repository"], tensor_parallel_size=8)

  def _generate_response(self, prompts, **kwargs):
    full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    
    sampling_params = SamplingParams(**kwargs)

    outputs = self.llm.generate(full_prompts, sampling_params=sampling_params)
    responses = []
    for out in outputs:
      prompt_tokens = len(out.prompt_token_ids)
      completion_tokens = sum([len(output.token_ids) for output in out.outputs])
      responses.append({
        "request_id": out.request_id,
        "object": "text_completion",
        "model": "Mixtral-8x7B-Instruct-v0.1",
        "choices":[{"text": output.text, "index": output.index, "logprobs": output.logprobs, "finish_reason": output.finish_reason} for output in out.outputs],
        "usage": {
          "prompt_tokens": prompt_tokens,
          "completion_tokens": max([len(output.token_ids) for output in out.outputs]),
          "total_tokens": prompt_tokens + completion_tokens
        }
      })
    return responses

  def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        The input parameters are compatible with `llm/v1/chat`
        https://mlflow.org/docs/latest/gateway/index.html#chat
        """

        # The standard parameters for chat routes with type llm/v1/chat can be find at
        # https://mlflow.org/docs/latest/gateway/index.html#chat
        messages = model_input["prompt"]
        candidate_count = params.get("candidate_count", 1)
        temperature = params.get("temperature", 1.0)
        max_tokens = params.get("max_new_tokens", 100)
        stop = params.get("stop", [])

        response_messages = self._generate_response(
            messages, 
            n=candidate_count, 
            temperature=temperature, 
            max_tokens=max_tokens,
            stop=stop,
        )

        return response_messages


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
input_example = {"prompt": "What is Machine Learning?"}
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
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=MixtralInstruct(),
        artifacts={"repository":
           snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.1.1", "transformers==4.36.0", "accelerate==0.25.0", "torchvision==0.16.1", "vllm==0.2.4", "megablocks==0.5.0", "ray==2.8.1"],
        signature=signature,
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
# This may take 2 minutes to complete

registered_name = "models.default.mixtral_8x7b_instruct"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

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

registered_name = "models.default.mixtral_8x7b_instruct"
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
    {"prompt": "What is large language model?"},
    params={
        "temperature": 0.5,
        "max_new_tokens": 100,
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `mixtral-8x7b-instruct` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, see the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu). The feature is in Public Preview.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'mixtral-8x7b-instruct'

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

model_version = result  # the returned result of mlflow.register_model

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
# Choose GPU_MEDIUM on Azure, and `GPU_LARGE` on Azure
workload_type = "GPU_MEDIUM_8"

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
            "model_name": model_version.name,
            "model_version": model_version.version,
            "workload_type": workload_type,
            "workload_size": "Small",
            "scale_to_zero_enabled": "False",
        }
    ]
})
w.serving_endpoints.create(name=endpoint_name, config=config)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it.

# COMMAND ----------



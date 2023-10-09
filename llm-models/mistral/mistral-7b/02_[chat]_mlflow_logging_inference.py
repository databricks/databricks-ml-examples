# Databricks notebook source
# MAGIC %md
# MAGIC # Manage Mistral-7B-Instruct as chat completion model with MLFlow on Databricks
# MAGIC
# MAGIC The [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) Large Language Model (LLM) is a instruct fine-tuned version of the [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure

# COMMAND ----------

# MAGIC %pip install -U "mlflow-skinny[databricks]>=2.4.1"
# MAGIC %pip install -U transformers==4.34.0
# MAGIC %pip install -U databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Mistral-7B-Instruct-v0. in https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/commits/main
model = "mistralai/Mistral-7B-Instruct-v0.1"
revision = "3dc28cf29d2edd31a0a7b8f0b21637059815b4d5"

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
snapshot_location = snapshot_download(repo_id=model, revision=revision)

# COMMAND ----------

import json
import mlflow
import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList


class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop[2:] == input_ids[0][-(len(stop) - 2) :])).item():
                return True

        return False


# Define PythonModel which is compatible to OpenAI-compatible APIs to log with mlflow.pyfunc.log_model
class MistralChat(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts["repository"], padding_side="left"
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts["repository"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda",
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()

    def _generate_response(
        self, messages, candidate_count, temperature, max_tokens, stop
    ):
        """
        This method generates prediction for a single input.
        """
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        # Encode the input and generate prediction
        encoded_input = encodeds.to("cuda")
        generation_config = transformers.GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=candidate_count,
        )
        if stop:
          stop_words_ids = [
              self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
              for stop_word in stop
          ]
          stopping_criteria = StoppingCriteriaList(
            [ChatStoppingCriteria(stops=stop_words_ids)]
          )
        else:
          stopping_criteria=None
        
        output = self.model.generate(
            encoded_input,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response_messages = []
        prompt_length = len(encoded_input)

        # Decode the prediction to text
        output_tokens = 0
        for i in range(len(output)):
            generated_text = self.tokenizer.decode(output[i], skip_special_tokens=True)

            # Removing the prompt from the generated text
            generated_response = self.tokenizer.decode(
                output[i], skip_special_tokens=True
            )

            gen_length = len(output[i]) - prompt_length

            generated_response = {
                "message": {
                    "role": "assistant",
                    "content": generated_response[prompt_length:],
                },
                "metadata": {"finish_reason": "length" if gen_length==max_tokens else "stop"},
            }

            response_messages.append(generated_response)
            output_tokens += gen_length

        metadata = {
            "input_tokens": prompt_length,
            "output_tokens": output_tokens,
            "total_tokens": prompt_length+output_tokens,
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "route_type": "llm/v1/chat",
        }

        return response_messages, metadata

    def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        The input parameters are compatible with `llm/v1/chat`
        https://mlflow.org/docs/latest/gateway/index.html#chat
        """

        outputs = []

        # The standard parameters for chat routes with type llm/v1/chat can be find at
        # https://mlflow.org/docs/latest/gateway/index.html#chat
        messages = model_input["messages"][0]
        candidate_count = params.get("candidate_count", 1)
        temperature = params.get("temperature", 1.0)
        max_tokens = params.get("max_tokens", 100)
        stop = params.get("stop", [])

        response_messages, metadata = self._generate_response(
            messages, candidate_count, temperature, max_tokens, stop
        )

        outputs.append({"candidates": response_messages, "metadata": metadata})

        # {"candidates": [...]} is the required response format for MLflow AI gateway -- see 07_ai_gateway for example
        return outputs

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, ParamSchema, ParamSpec
import pandas as pd

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "messages")])
output_schema = Schema([ColSpec(DataType.string)])
param_schema = ParamSchema([
  ParamSpec("candidate_count", "long", 1),
  ParamSpec("temperature", "double", 1.0),
  ParamSpec("max_tokens", "long", 512),
  ParamSpec("stop", "string", None),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)

# Define input example
input_example = pd.DataFrame(
    {
        "messages": [
            [
                {"role": "user", "content": "What is ML?"},
            ]
        ],
    }
)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=MistralChat(),
        artifacts={"repository":
           snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.0.1", "transformers==4.34.0", "accelerate==0.21.0", "torchvision==0.15.2"],
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

registered_name = "models.default.mistral_7b_chat_completion"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name


result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Choose the right model version registered in the above cell.
client.set_registered_model_alias(
    name=registered_name, alias="Champion", version=result.version
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model from Unity Catalog

# COMMAND ----------

import mlflow
import pandas as pd

registered_name = "models.default.mistral_7b_chat_completion"
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "messages": [
            [
                {"role": "user", "content": "You are a helpful assistant. Answer the following question.\n What is ML?"},
            ]
        ],
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `LLaMAV2-7b` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team or sign up [here](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit).

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'mistral-7b-chat-completion'

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

model_version = result  # the returned result of mlflow.register_model

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
# Choose GPU_MEDIUM on Azure, and `GPU_LARGE` on Azure
workload_type = "GPU_LARGE"

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
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `04_langchain` for example code) running in the same workspace.

# COMMAND ----------



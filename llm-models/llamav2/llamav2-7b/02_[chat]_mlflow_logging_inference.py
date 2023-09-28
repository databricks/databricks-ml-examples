# Databricks notebook source
# MAGIC %md
# MAGIC # Manage Llama 2 7B chat model with MLFlow on Databricks
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLFlow

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main
model = "meta-llama/Llama-2-7b-chat-hf"
revision = "0ede8dd71e923db6258295621d817ca8714516d4"

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
snapshot_location = snapshot_download(
    repo_id=model, revision=revision, ignore_patterns="*.safetensors*"
)

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
class Llama2Chat(mlflow.pyfunc.PythonModel):
    # Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \n\n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

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
            device_map="cuda:0",
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()

    def _build_prompt(self, messages):
        """
        Assemble the message list to one prompt.
        message: A list of messages in a conversation from which to a new message (chat completion). Each chat message is a string dictionary containing `role` and `content`.
        """
        if isinstance(messages, str):
          messages= json.loads(messages)

        try:
          test = messages[0]["role"]
        except Exception as e:
          raise Exception(f"Input {messages} contains error: {e}") 

        if messages[0]["role"] == "system":
            prompt = "<s>" + self.B_SYS + messages[0]["content"] + self.E_SYS
            messages = messages[1:]
        else:
            prompt = "<s>" + self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS

        assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        assert (
            messages[-1]["role"] == "user"
        ), f"Last message must be from user, got {messages[-1]['role']}"
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "user":
                prompt += self.B_INST + content + self.E_INST
            else:
                prompt += content
        return prompt

    def _generate_response(
        self, messages, candidate_count, temperature, max_tokens, stop
    ):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(messages)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        generation_config = transformers.GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=candidate_count,
        )
        stop_words_ids = [
            self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop
        ]
        stopping_criteria = StoppingCriteriaList(
            [ChatStoppingCriteria(stops=stop_words_ids)]
        )
        output = self.model.generate(
            encoded_input,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )

        # Decode the prediction to text
        response_messages = []
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
        for i in range(len(output)):
            generated_text = self.tokenizer.decode(output[i], skip_special_tokens=True)

            # Removing the prompt from the generated text
            generated_response = self.tokenizer.decode(
                output[i][prompt_length:], skip_special_tokens=True
            )

            response_messages.append(generated_response)

        return response_messages

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        The input parameters are compatible with `llm/v1/chat`
        https://mlflow.org/docs/latest/gateway/index.html#chat
        """

        outputs = []

        # The standard parameters for chat routes with type llm/v1/chat can be find at
        # https://mlflow.org/docs/latest/gateway/index.html#chat
        messages = model_input["messages"][0]
        candidate_count = model_input.get("candidate_count", [1])[0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]
        stop = model_input.get("stop", [[]])[0]

        response_messages = self._generate_response(
            messages, candidate_count, temperature, max_tokens, stop
        )

        # {"candidates": [...]} is the required response format for MLflow AI gateway -- see 07_ai_gateway for example
        return {"candidates": response_messages}

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "messages"),
        ColSpec(DataType.long, "candidate_count", optional=True),
        ColSpec(DataType.double, "temperature", optional=True),
        ColSpec(DataType.long, "max_tokens", optional=True),
        ColSpec(DataType.string, "stop", optional=True),
    ]
)
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {
        "messages": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is ML?"},
            ]
        ],
        "candidate_count": [2],
        "temperature": [0.5],
        "max_tokens": [100],
        "stop": [["\n\n"]],
    }
)

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 1.7 minutes to complete
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2Chat(),
        artifacts={"repository":
           snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "sentencepiece"],
        input_example=input_example,
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
# This may take 2.2 minutes to complete

registered_name = "models.default.llama2_7b_chat_completion"  # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name


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

registered_name = "models.default.llama2_7b_chat_completion"
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "messages": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is ML?"},
            ]
        ],
        "candidate_count": [2],
        "temperature": [0.5],
        "max_tokens": [100],
        "stop": [["\n"]],
    }
)

# COMMAND ----------

loaded_model.predict(
    {
        "messages": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is ML?"},
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
endpoint_name = 'llama2-7b-chat-completion'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
deploy_url = f"{databricks_url}/api/2.0/serving-endpoints"

model_version = result  # the returned result of mlflow.register_model
endpoint_config = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
                "model_name": model_version.name,
                "model_version": model_version.version,
                "workload_type": "GPU_MEDIUM",
                "workload_size": "Small",
                "scale_to_zero_enabled": "False",
            }
        ]
    },
}
endpoint_json = json.dumps(endpoint_config, indent="  ")

# Send a POST request to the API
deploy_response = requests.request(
    method="POST", headers=deploy_headers, url=deploy_url, data=endpoint_json
)

if deploy_response.status_code != 200:
    raise Exception(
        f"Request failed with status {deploy_response.status_code}, {deploy_response.text}"
    )

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `04_langchain` for example code) running in the same workspace.

# Databricks notebook source
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
from huggingface_hub import notebook_login, login
import pandas as pd
import transformers
import mlflow
import torch
import logging

print(torch.__version__)

# notebook_login()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

# COMMAND ----------


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

from databricks_llm.notebook_utils import get_dbutils

get_dbutils().widgets.text("dbfs_model_location", "/dbfs/llm/", "dbfs_model_location")
get_dbutils().widgets.text("model_name", "my_llm", "model_name")

# COMMAND ----------

dbfs_model_location = get_dbutils().widgets.get("dbfs_model_location")
model_name = get_dbutils().widgets.get("model_name")
print(dbfs_model_location)
# COMMAND ----------


# MAGIC !ls -lah {dbfs_model_location}

# COMMAND ----------


class LLMPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
    ):
        pass

    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts["repository"], padding_side="left", trust_remote_code=True
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts["repository"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()

    def _build_prompt(self, query):
        """
        This method generates the prompt for the model.
        """

        return f"""<s>[INST] <<SYS>>Extract entities from the text below.<</SYS>> {query} [/INST] """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            encoded_input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        # Decode the prediction to text
        # generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
        generated_response = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )

        return generated_response


# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "prompt"),
        ColSpec(DataType.double, "temperature"),
        ColSpec(DataType.long, "max_tokens"),
    ]
)
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {"prompt": ["what is ML?"], "temperature": [0.5], "max_tokens": [100]}
)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=LLMPyFuncModel(),
        artifacts={"repository": dbfs_model_location},
        pip_requirements=[
            "torch==2.0.1",
            "transformers==4.28.1",
            "accelerate==0.18.0",
            "einops",
            "sentencepiece",
        ],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model("runs:/" + run.info.run_id + "/model")

# COMMAND ----------

# Make a prediction using the loaded model
input_example = pd.DataFrame(
    {"prompt": ["what is ML?"], "temperature": [0.5], "max_tokens": [100]}
)
loaded_model.predict(input_example)

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model("runs:/" + run.info.run_id + "/model", model_name)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.

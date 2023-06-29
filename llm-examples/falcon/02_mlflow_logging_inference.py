# Databricks notebook source
# MAGIC %md
# MAGIC # Manage Falcon-7b-instruct model with MLFlow on Databricks
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input (batch inference requires slightly more memory). On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.

# COMMAND ----------

# MAGIC %pip install -q -U torch==2.0.1
# MAGIC %pip install -q einops==0.6.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model

# COMMAND ----------

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
snapshot_location = snapshot_download(repo_id="tiiuae/falcon-7b-instruct",  ignore_patterns="coreml/*", revision="9f16e66a0235c4ba24e321e3be86dd347a7911a0")

# COMMAND ----------

import mlflow
import torch
import transformers

# Define PythonModel to log with mlflow.pyfunc.log_model

class Falcon(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id).to('cuda')
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
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

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [1.0])[i]
          max_new_tokens = model_input.get("max_new_tokens", [100])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        return outputs

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_new_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_new_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 4 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Falcon(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "einops","sentencepiece"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model

# COMMAND ----------

# Register model in MLflow Model Registry
# This may take 7 minutes to complete
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name="falcon-7b-instruct-demo",
    await_registration_for=600,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model from model registry
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC
# MAGIC If it's run in a separate notebook, make sure to install the Python libraries before loading from MLflow.
# MAGIC ```
# MAGIC %pip install -q -U torch==2.0.1
# MAGIC %pip install -q einops==0.6.1
# MAGIC ```

# COMMAND ----------

# Clear GPU memory

from numba import cuda
device = cuda.get_current_device()
device.reset()

# COMMAND ----------

import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/falcon-7b-instruct-demo/latest")

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "prompt": ["What is ML?", "What is large language model?"],
        "temperature": [0.1, 0.5],
        "max_new_tokens": [100, 100],
    }
)

# COMMAND ----------



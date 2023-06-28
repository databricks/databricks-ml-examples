# Databricks notebook source
# MAGIC %md
# MAGIC # Load and Inference MPT-7B model with MLFlow on Databricks
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS

# COMMAND ----------

# Skip this step if running on Databricks runtime 13.2 GPU and above.
!wget -O /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcurand-11-7_10.2.10.91-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcurand-11-7_10.2.10.91-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcurand-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install xformers einops flash-attn==v1.0.3.post0 triton==2.0.0.dev20221202

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model to MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC Define a customized PythonModel to log into MLFlow.

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import accelerate

class MPT(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.model.to(device='cuda')
        
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

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
          prompt = row["prompt"]
          # You can add other parameters here
          temperature = model_input.get("temperature", [1.0])[0]
          max_new_tokens = model_input.get("max_new_tokens", [100])[0]
          full_prompt = self._build_prompt(prompt)
          encoded_input = self.tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
          output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
          prompt_length = len(encoded_input[0])
          generated_text.append(self.tokenizer.batch_decode(output[:,prompt_length:], skip_special_tokens=True))
        return pd.Series(generated_text)

# COMMAND ----------

# MAGIC %md
# MAGIC Download the model

# COMMAND ----------

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
model_location = snapshot_download(repo_id="mosaicml/mpt-7b-instruct", cache_dir="/local_disk0/.cache/huggingface/", revision="bbe7a55d70215e16c00c1825805b81e4badb57d7")

# COMMAND ----------

# MAGIC %md
# MAGIC Log the model to MLFlow

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 5 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=MPT(),
        artifacts={'repository' : model_location},
        pip_requirements=[f"torch=={torch.__version__}", 
                          f"transformers=={transformers.__version__}", 
                          f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model

# COMMAND ----------

# Register model in MLflow Model Registry
# This may take about 6 minutes to complete
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name="mpt-7b-instruct",
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model from model registry
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------

import mlflow
import pandas as pd
loaded_model = mlflow.pyfunc.load_model(f"models:/mpt-7b-instruct/latest")

input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
print(loaded_model.predict(input_example))

# COMMAND ----------



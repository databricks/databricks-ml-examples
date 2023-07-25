# Databricks notebook source
# MAGIC %md
# MAGIC # Manage MPT-7B-instruct model with MLFlow on Databricks
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input (batch inference requires slightly more memory). On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

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

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

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

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
print(loaded_model.predict(input_example))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the MPT-7B-Instruct model.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'mpt-7b-instruct-example'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `04_langchain` for example code) running in the same workspace.




# Databricks notebook source
# MAGIC %md
# MAGIC # Manage XGen-7B-8K-Base model with MLFlow on Databricks
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model

# COMMAND ----------

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
snapshot_location = snapshot_download(repo_id="Salesforce/xgen-7b-8k-base", revision="3987e094377fae577bba039af1b300ee8086f9e1")

# COMMAND ----------

import mlflow
import torch
import transformers

# Define PythonModel to log with mlflow.pyfunc.log_model

class Xgen7b8k(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    model_name = "Salesforce/xgen-7b-8k-base"
    frozen_revision = context.artifacts['repository']
    """
    This method initializes the tokenizer and language model
    using the specified model repository.
    """
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
      context.artifacts['repository'],
      trust_remote_code=True,
    )
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
      context.artifacts['repository'], 
      torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True, 
      device_map="auto",
    ).to('cuda')
    self.model.eval()

  def _generate_response(self, prompt, temperature, max_new_tokens):  
    """
    This method generates prediction for a single input.
    """
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
  ColSpec(DataType.long, "max_new_tokens")],
)
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
  "prompt":["what is ML?"], 
  "temperature": [0.5],
  "max_new_tokens": [100],
})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take up to 15 minutes to complete
with mlflow.start_run() as run:  
  mlflow.pyfunc.log_model(
    "model",
    python_model=Xgen7b8k(),
    artifacts={'repository': snapshot_location},
    pip_requirements=["torch", "transformers", "accelerate", "einops", "sentencepiece"],
    input_example=input_example,
    signature=signature,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Register the model

# COMMAND ----------

# Register model to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
# This may take up to 10 minutes to complete
registered_model_name = "main.default.xgen-7b-8k-base-demo"
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name=registered_model_name,
    await_registration_for=600,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load the model from model registry

# COMMAND ----------

import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model("models:/main.default.xgen-7b-8k-base-demo/1")

# Make a prediction using the loaded model
loaded_model.predict(
  {
    "prompt": ["What is ML?", "What is a large language model?"],
    "temperature": [0.1, 0.5],
    "max_new_tokens": [100, 100],
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Model Serving Endpoint
# MAGIC
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the XGen-7B-8K-Base model.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'xgen-7b-8k-base-serving'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result
endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": "False",
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

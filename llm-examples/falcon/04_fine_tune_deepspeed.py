# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune Falcon-7B with DeepSpeed
# MAGIC Environment tested for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `Standard_NC48ads_A100_v4` on Azure (2 A100 80GB GPUs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Install Python packages.

# COMMAND ----------

# MAGIC %pip install -q torch==2.0.1 einops deepspeed
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC (If the MLR version is 13.2+, skip this cell.)
# MAGIC
# MAGIC Install CUDA dev libaries that are required by DeepSpeed.

# COMMAND ----------

# Skip this step if running on Databricks runtime 13.2 GPU and above.
!wget -O /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run deepspeed fine-tuning script
# MAGIC The fine-tuning script is in `scripts/fine_tune_deepspeed_falcon.py`. In the script,
# MAGIC
# MAGIC - the default training configurations are provided in the `click.option` lines; you can pass custom arguments into the program such as
# MAGIC   ```shell
# MAGIC   deepspeed --num_gpus=<num_gpus> scripts/fine_tune_deepspeed_falcon.py \
# MAGIC   --dbfs-output-dir="/dbfs/falcon_7b_openassistant" \
# MAGIC   --epochs=1 --max_steps=-1
# MAGIC   ```
# MAGIC   
# MAGIC - the deepspeed configurations are read from `../config/a100_config.json`, which is adapted from the ZeRO stage 3 configuration of https://huggingface.co/docs/transformers/main_classes/deepspeed
# MAGIC - the dataset used for fine-tuning is `timdettmers/openassistant-guanaco`, which is a subset of `OpenAssistant/oasst1`

# COMMAND ----------

# MAGIC %sh
# MAGIC deepspeed --num_gpus=2 scripts/fine_tune_deepspeed_falcon.py

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/local_disk0/output`.

# COMMAND ----------

# MAGIC %ls /local_disk0/output

# COMMAND ----------

# MAGIC %sh
# MAGIC # Remove checkpoints before logging the model to MLflow
# MAGIC rm -rf /local_disk0/output/checkpoint-*

# COMMAND ----------

import mlflow
import torch
import transformers

# Define PythonModel to log with mlflow.pyfunc.log_model

class Falcon(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the logged artifact.
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
            pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Human:"
        RESPONSE_KEY = "### Assistant:"

        return f"{INSTRUCTION_KEY} {instruction} {RESPONSE_KEY}"

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
        artifacts={'repository' : "/local_disk0/output"},
        pip_requirements=["torch", "transformers", "accelerate", "einops","sentencepiece"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model("runs:/" + run.info.run_id + "/model")

# COMMAND ----------

# Make a prediction using the loaded model
results = loaded_model.predict(
    {
        "prompt": ["什么是机器学习？", "What is machine learning?"],
        "temperature": [1.0, 0.5],
        "max_new_tokens": [100, 100],
    }
)

for result in results:
  print(result + '\n')

# COMMAND ----------



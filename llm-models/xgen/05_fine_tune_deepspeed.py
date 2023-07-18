# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine-tune XGen-7B-8k with DeepSpeed
# MAGIC
# MAGIC This example notebook demonstrates how to fine-tune the XGen-7B-8k model on [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), which is an instruction following dataset, with the [DeepSpeed integration of Hugging Face](https://huggingface.co/docs/transformers/main_classes/deepspeed).
# MAGIC
# MAGIC Environment tested for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.12xlarge` (4 A10 GPUs) on AWS
# MAGIC
# MAGIC On Azure, we suggest using `Standard_NC48ads_A100_v4` on Azure (2 A100-80GB GPUs).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Python packages.

# COMMAND ----------

# MAGIC %pip install -q deepspeed

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run deepspeed fine-tuning script
# MAGIC
# MAGIC The fine-tuning script is in `scripts/fine_tune_deepspeed.py`. In the script,
# MAGIC - The dataset used for fine-tuning is `databricks/databricks-dolly-15k`
# MAGIC - The default training configurations are provided in the `click.option` lines.
# MAGIC
# MAGIC   You can see the explanation of the configuration options in the script, and pass custom arguments into the program such as
# MAGIC   ```shell
# MAGIC   deepspeed --num_gpus=<num_gpus> scripts/fine_tune_deepspeed.py \
# MAGIC    --per-device-train-batch-size=16 --per-device-eval-batch-size=16 --epochs=1 --max-steps=-1
# MAGIC   ```
# MAGIC
# MAGIC - The deepspeed configurations are read from `../../config/a10_config.json`, which is adapted from the ZeRO stage 3 configuration of https://huggingface.co/docs/transformers/main_classes/deepspeed. If running on A100 GPUs, please set `CONFIG_PATH=../../config/a100_config.json` in the script.

# COMMAND ----------

# MAGIC %sh
# MAGIC # This takes about 30 minutes to complete on g5.12xlarge.
# MAGIC # If the progress bar fails to update, you can check that training is still running by monitoring GPU utilization.
# MAGIC deepspeed --num_gpus=4 scripts/fine_tune_deepspeed.py --per-device-train-batch-size=16 --per-device-eval-batch-size=16 --epochs=1 --max-steps=-1

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/local_disk0/output`.

# COMMAND ----------

# MAGIC %ls /local_disk0/output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save fine-tuned model with MLflow

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
        )
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
        artifacts={"repository": "/local_disk0/output"},
        pip_requirements=["torch", "transformers", "accelerate", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load fine-tuned model for inference

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model("runs:/" + run.info.run_id + "/model")

# COMMAND ----------

# Make a prediction using the loaded model
results = loaded_model.predict(
    {
        "prompt": ["Write a short poem about AI taking over the world.", "What is machine learning?"],
        "temperature": [0.8, 0.5],
        "max_new_tokens": [100, 100],
    }
)

for result in results:
    print(result + '\n')

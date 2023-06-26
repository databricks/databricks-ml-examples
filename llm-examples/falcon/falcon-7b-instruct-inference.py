# Databricks notebook source
# MAGIC %md
# MAGIC # Falcon-7b-instruct Inference on Databricks
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
# MAGIC ## Inference
# MAGIC The below snippets are adapted from [the model card of falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct). The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# Load model to text generation pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision="9f16e66a0235c4ba24e321e3be86dd347a7911a0", # it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of falcon-7b-instruct in https://huggingface.co/tiiuae/falcon-7b-instruct/commits/main
)

# COMMAND ----------

# Define prompt template, the format below is from: http://fastml.com/how-to-train-your-own-chatgpt-alpaca-style-part-one/

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

# COMMAND ----------

# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on a single input

# COMMAND ----------

results = gen_text(["What is a large language model?"])
print(results[0])

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["What is a large language model?"], temperature=0.5, max_new_tokens=100, use_template=True)
print(results[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference

# COMMAND ----------

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
pipeline.tokenizer.padding_side = 'left'

# COMMAND ----------

# From databricks-dolly-15k
inputs = [
  "Think of some family rules to promote a healthy family relationship",
  "In the series A Song of Ice and Fire, who is the founder of House Karstark?",
  "which weighs more, cold or hot water?",
  "Write a short paragraph about why you should not have both a pet cat and a pet bird.",
  "Is beauty objective or subjective?",
  "What is SVM?",
  "What is the current capital of Japan?",
  "Name 10 colors",
  "How should I invest my money?",
  "What are some ways to improve the value of your home?",
  "What does fasting mean?",
  "What is cloud computing in simple terms?",
  "What is the meaning of life?",
  "What is Linux?",
  "Why do people like gardening?",
  "What makes for a good photograph?"
]

# COMMAND ----------

# Set batch size
results = gen_text(inputs, use_template=True, batch_size=8)

for output in results:
  print(output)
  print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use MLflow to manage the model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model

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
# MAGIC ### Register the model

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
# MAGIC ### Load the model from model registry
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC
# MAGIC You can clear the GPU memory by "Detach & re-attach" the notebook, and re-install the Python libraries.

# COMMAND ----------

# MAGIC %pip install -q -U torch==2.0.1
# MAGIC %pip install -q einops==0.6.1

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



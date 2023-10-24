# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune llama-2-7b-hf model from marketplace with QLORA
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) is the 7B pretrained model, converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC This is to fine-tune [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) models on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.1 GPU ML Runtime
# MAGIC - Instance: `g5.8xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient finetuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# To access models in Unity Catalog, ensure that MLflow is up to date
%pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
%pip install peft==0.5.0 # git+https://github.com/huggingface/peft.git
%pip install datasets==2.14.6 bitsandbytes==0.41.1 einops==0.7.0 trl==0.7.2
%pip install accelerate==0.23.0 transformers==4.34.1
dbutils.library.restartPython()

# COMMAND ----------

import mlflow

# Set mlflow registry to databricks-uc
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

catalog_name = "databricks_llama_2_models" # Default catalog name when installing the model from Databricks Marketplace
version = 1

#model_mlflow_path = f"models:/{catalog_name}.models.llama_2_7b_hf/{version}"
model_mlflow_path = f"models:/databricks_marketplace.models_lu.llama_2_7b_hf/{version}"

model_local_path = "/local_disk0/llama_2_7b_hf/"
model_output_local_path = "/local_disk0/llama-2-7b-lora-fine-tune"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC
# MAGIC We will use the [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

# COMMAND ----------

from datasets import load_dataset

dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train")

# COMMAND ----------

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  response_key=RESPONSE_KEY,
  response="{response}",
  end_key=END_KEY
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  input_key=INPUT_KEY,
  input="{input}",
  response_key=RESPONSE_KEY,
  response="{response}",
  end_key=END_KEY
)

def apply_prompt_template(examples):
  instruction = examples["instruction"]
  response = examples["response"]
  context = examples.get("context")

  if context:
    full_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
  else:
    full_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
  return { "text": full_prompt }

dataset = dataset.map(apply_prompt_template)

# COMMAND ----------

dataset["text"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the model
# MAGIC
# MAGIC In this section we will load the [LLaMAV2](), quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

from mlflow.artifacts import download_artifacts

path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)

# COMMAND ----------

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

tokenizer_path = os.path.join(path, "components", "tokenizer")
model_path = os.path.join(path, "model")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config
)
model.config.use_cache = False

# COMMAND ----------

# MAGIC %md
# MAGIC Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. 

# COMMAND ----------

# Choose all linear layers from the model
import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

linear_layers = find_all_linear_names(model)
print(f"Linear layers in the model: {linear_layers}")

# COMMAND ----------

from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=linear_layers,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the trainer

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# COMMAND ----------

from transformers import TrainingArguments

output_dir = "/local_disk0/results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 100
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 10
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Then finally pass everthing to the trainer

# COMMAND ----------

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# COMMAND ----------

# MAGIC %md
# MAGIC We will also pre-process the model by upcasting the layer norms in float 32 for more stable training

# COMMAND ----------

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's train the model! Simply call `trainer.train()`

# COMMAND ----------

trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the LORA model

# COMMAND ----------

trainer.save_model(model_output_local_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Merge the model to one HF model
# MAGIC
# MAGIC By default, the PEFT library will only save the QLoRA adapters, we want to merge the weight and save the model to HF format.

# COMMAND ----------

import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_local_path = "/local_disk0/llama_2_7b_hf/"
model_output_local_path = "/local_disk0/llama-2-7b-lora-fine-tune"
merged_model_path = "/local_disk0/llama-2-7b-lora-merge"

tokenizer_path = os.path.join(model_local_path, "components", "tokenizer")
model_path = os.path.join(model_local_path, "model")

config = PeftConfig.from_pretrained(model_output_local_path)

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  device_map="auto",
  torch_dtype=torch.bfloat16,
)

peft_model = PeftModel.from_pretrained(model, model_output_local_path, config=config)

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the fine tuned model to MLFlow

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def build_prompt(instruction):
    return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

# Define input and output schema
input_example = {"prompt": build_prompt("What is Machine Learning?")}
inference_config = {
  "temperature": 1.0,
  "max_new_tokens": 100,
  "do_sample": True,
}
signature = infer_signature(
    model_input=input_example,
    model_output="Machien Learning is...",
    params=inference_config
)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
        transformers_model={
            "model": merged_model,
            "tokenizer": tokenizer,
        },
    task = "text-generation",
    artifact_path="model",
    input_example=input_example,
    signature=signature,
    # Add the metadata task so that the model serving endpoint created later will be optimized
    metadata={"task": "llm/v1/completions"}
  )

# COMMAND ----------

registered_name = "databricks_marketplace.models_lu.llama2_7b_fine_tune_completions" # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    registered_name,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}/1")

# Make a prediction using the loaded model
loaded_model.predict(
  {"prompt": "What is large language model?"}, 
  params={
    "temperature": 0.5,
    "max_new_tokens": 100,
    }
)

# COMMAND ----------



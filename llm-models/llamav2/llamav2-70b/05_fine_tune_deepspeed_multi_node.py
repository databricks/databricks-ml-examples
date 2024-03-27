# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune Llama 2 70B Chat with DeepspeedTorchDistributor
# MAGIC
# MAGIC This notebook provides an example of how to fine-tune Meta's [Llama-2-70b-hf model](https://huggingface.co/meta-llama/Llama-2-70b-hf) using Apache Spark's DeepspeedTorchDistributor and the Hugging Face `transformers` library. 
# MAGIC
# MAGIC ## Requirements
# MAGIC For this notebook, you need: 
# MAGIC
# MAGIC - A GPU, multi-node cluster. [DeepSpeed](https://www.deepspeed.ai/) does not currently support running on CPU.
# MAGIC - Databricks Runtime 14.0 ML and above
# MAGIC - 2xp4d.24xlarge as workers (2xA100) on AWS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC When running on MLR 14.0, `deepspeed` must be installed on the cluster.

# COMMAND ----------

# MAGIC %pip install -q deepspeed
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the parameters

# COMMAND ----------

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
print(f"Number of workers: {NUM_WORKERS}, number of GPUs per worker: {NUM_GPUS_PER_WORKER}")
HF_auth_token=<HF-API-TOKEN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the DeepSpeed configuration
# MAGIC
# MAGIC You can choose to pass in a DeepSpeed configuration to the distributor. If you do not provide one, the default configuration is applied.
# MAGIC
# MAGIC The configuration can be passed as either a Python dictionary or a string that represents a file path containing the `json` configuration.

# COMMAND ----------

deepspeed_config = {
  "fp16": {
    "enabled": False
  },
  "bf16": {
    "enabled": True
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 5e7,
    "reduce_bucket_size": "auto",
    "reduce_scatter": True,
    "stage3_max_live_parameters" : 1e9,
    "stage3_max_reuse_distance" : 1e9,
    "stage3_prefetch_bucket_size" : 5e8,
    "stage3_param_persistence_threshold" : 1e6,
    "stage3_gather_16bit_weights_on_model_save": True,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    }
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": False
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the DeepSpeed distributor
# MAGIC
# MAGIC When you create the distributor you can specify how many nodes and GPUs per node to use.

# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

dist = DeepspeedTorchDistributor(
  numGpus=NUM_GPUS_PER_WORKER,
  nnodes=NUM_WORKERS,
  localMode=False,  # Distribute training across workers.
  deepspeedConfig=deepspeed_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the training function
# MAGIC
# MAGIC This example uses HuggingFace's `transformers` package to fine-tune Llama2.

# COMMAND ----------

from datasets import Dataset, load_dataset
import os
from transformers import AutoTokenizer

TOKENIZER_PATH = 'meta-llama/Llama-2-70b-hf'
DEFAULT_TRAINING_DATASET = "databricks/databricks-dolly-15k"

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  response_key=RESPONSE_KEY,
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  input_key=INPUT_KEY,
  input="{input}",
  response_key=RESPONSE_KEY,
)


def load_training_dataset(
  tokenizer,
  path_or_dataset: str = DEFAULT_TRAINING_DATASET,
) -> Dataset:
  print(f"Loading dataset from {path_or_dataset}")
  # Update the following path to point to a UC Volumes location if DBFS is not available
  dataset = load_dataset(path_or_dataset, cache_dir='/dbfs/llama2-deepspeed')["train"]
  print(f"Found {dataset.num_rows} rows")

  def _reformat_data(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    context = rec.get("context")

    if context:
      questions = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, input=context)
    else:
      questions = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction)
    return {"text": f"{{ 'prompt': {questions}, 'response': {response} }}"}
  
  dataset = dataset.map(_reformat_data)

  def tokenize_function(allEntries):
    return tokenizer(allEntries['text'], truncation=True, max_length=512,)
  
  dataset = dataset.map(tokenize_function)
  split_dataset = dataset.train_test_split(test_size=1000)
  train_tokenized_dataset = split_dataset['train']
  eval_tokenized_dataset = split_dataset['test']

  return train_tokenized_dataset, eval_tokenized_dataset

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_auth_token=HF_auth_token)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, eval_dataset = load_training_dataset(tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC The following command defines the training function that Deepspeed will run. The training function is sent to each worker to run and uses the `transformers` library to load the model, configure training args, then uses the HF trainer to train.

# COMMAND ----------

from functools import partial
import json
import logging
import os
import numpy as np
from pathlib import Path
import torch

import transformers
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  DataCollatorForLanguageModeling,
  PreTrainedTokenizer,
  Trainer,
  TrainingArguments,
)

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

MODEL_PATH = 'meta-llama/Llama-2-70b-hf'
TOKENIZER_PATH = 'meta-llama/Llama-2-70b-hf'
LOCAL_OUTPUT_DIR = "/dbfs/llama-2-fine-tune/output"


def load_model(pretrained_model_name_or_path: str) -> AutoModelForCausalLM:
  print(f"Loading model for {pretrained_model_name_or_path}")
  model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=HF_auth_token
  )
  model.config.use_cache = False
  return model

def fine_tune_llama2(
  *,
  local_rank: str = None,
  input_model: str = MODEL_PATH,
  local_output_dir: str = LOCAL_OUTPUT_DIR,
  dbfs_output_dir: str = None,
  epochs: int = 3,
  per_device_train_batch_size: int = 1,
  per_device_eval_batch_size: int = 1,
  lr: float = 1e-5,
  gradient_checkpointing: bool = True,
  gradient_accumulation_steps: int = 8,
  bf16: bool = False,
  logging_steps: int = 10,
  save_steps: int = 400,
  max_steps: int = 200,
  eval_steps: int = 50,
  save_total_limit: int = 10,
  warmup_steps: int = 20,
  training_dataset: str = DEFAULT_TRAINING_DATASET,
):
  os.environ["HF_HOME"] = "/local_disk0/hf"
  os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
  os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

  model = load_model(input_model)
  
  fp16 = not bf16

  training_args = TrainingArguments(
    local_rank=local_rank,
    output_dir=local_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_checkpointing=gradient_checkpointing,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    num_train_epochs=epochs,
    weight_decay=1,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    fp16=fp16,
    bf16=bf16,
    deepspeed=deepspeed_config,
    logging_strategy="steps",
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=save_steps,
    max_steps=max_steps,
    save_total_limit=save_total_limit,
    warmup_steps=warmup_steps,
    report_to=[],
    push_to_hub=False,
    disable_tqdm=True,
  )
  
  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
  )

  print("Training the model")
  trainer.train()

  print(f"Saving Model to {local_output_dir}")
  trainer.save_model(output_dir=local_output_dir)
  tokenizer.save_pretrained(local_output_dir)

  if dbfs_output_dir:
    print(f"Saving Model to {dbfs_output_dir}")
    trainer.save_model(output_dir=dbfs_output_dir)
    tokenizer.save_pretrained(dbfs_output_dir)

  print("Training finished.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the distributor

# COMMAND ----------

dist.run(fine_tune_llama2, epochs=1, max_steps=1)

# COMMAND ----------

# MAGIC %md
# MAGIC After run() completes, you can load the model from the local output path (/dbfs/llama-2-fine-tune/output in this notebook).

# COMMAND ----------

from transformers import AutoTokenizer
import transformers
import torch

# Update the following path to point to a UC Volumes location if DBFS is not available
LOCAL_OUTPUT_DIR = "/dbfs/llama-2-fine-tune/output"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_OUTPUT_DIR)
tokenizer.pad_token = tokenizer.eos_token
pipeline = transformers.pipeline(
    "text-generation",
    model= LOCAL_OUTPUT_DIR,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    return_full_text=False
)
pipeline("What is ML?")

# COMMAND ----------



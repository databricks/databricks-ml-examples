# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune llama-2-70b-hf with QLORA
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) is the 70B pretrained model, converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC This is to fine-tune [llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) models on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC We recommend running this notebook using A100 GPUs. Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `Standard_NC24ads_A100_v4` on Azure
# MAGIC
# MAGIC We will leverage PEFT library from Hugging Face ecosystem, as well as QLoRA for more memory efficient finetuning.
# MAGIC
# MAGIC Requirements:
# MAGIC   - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept the license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/peft.git
# MAGIC %pip install bitsandbytes==0.40.1 einops==0.6.1 trl==0.4.7
# MAGIC %pip install torch==2.0.1 accelerate==0.21.0 transformers==4.31.0
# MAGIC %pip install -U datasets
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC
# MAGIC We will use the [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

# COMMAND ----------

from huggingface_hub import notebook_login
# Login to Huggingface to get access to the model
notebook_login()

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
# MAGIC In this section we will load Llama-2-70b-hf model, quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Llama-2-70b-chat-hf in https://huggingface.co/meta-llama/Llama-2-70b-hf/commits/main
model = "meta-llama/Llama-2-70b-hf"
revision = "cc8aa03a000ff08b4d5c5b39673321a2a396c396"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
  model,
  quantization_config=bnb_config,
  device_map="cuda:0",
  revision=revision,
  trust_remote_code=True
)
model.config.use_cache = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. Therefore we will add `q_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `k_proj`, and `v_proj` layers in the target modules in addition to the mixed query key value layer.

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
  target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'] # Choose all linear layers from the model
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the trainer
# MAGIC
# MAGIC Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# COMMAND ----------

from transformers import TrainingArguments

output_dir = "/local_disk0/results"
per_device_train_batch_size = 8
gradient_accumulation_steps = 8
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1500
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
# MAGIC Pass everything to the trainer.

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
# MAGIC We will also pre-process the model by upcasting the layer norms in float 32 for more stable training.

# COMMAND ----------

for name, module in trainer.model.named_modules():
  if "norm" in name:
    module = module.to(torch.float32)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model
# MAGIC
# MAGIC Now let's train the model! Simply call `trainer.train()`

# COMMAND ----------

trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the LORA model

# COMMAND ----------

trainer.save_model("/local_disk0/llamav2-70b-lora-fine-tune")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the fine tuned model to MLFlow

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig

peft_model_id = "/local_disk0/llamav2-70b-lora-fine-tune"
fine_tuned_model = PeftModel.from_pretrained(model, peft_model_id)


# COMMAND ----------

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
If we were playing a game where we had to identify things that can be found inside a house, which of these would we call out: car, chair, table, park, cloud, microwave.

### Response: """

batch = tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
output_tokens = fine_tuned_model.generate(
        input_ids = batch.input_ids, 
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
      )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)

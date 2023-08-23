# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune Falcon-40b with QLORA
# MAGIC
# MAGIC This example notebook demonstrates how to fine-tune the Falcon-40B model on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `Standard_NC24ads_A100_v4`
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient finetuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# MAGIC %pip install peft==0.5.0
# MAGIC %pip install datasets==2.12.0 bitsandbytes==0.40.1 einops==0.6.1 trl==0.4.7
# MAGIC %pip install torch==2.0.1 accelerate==0.22.0 transformers==4.31.0
# MAGIC dbutils.library.restartPython()

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
# MAGIC In this section we will load the Falcon-40B model, quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model = "tiiuae/falcon-40b"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

model.config.use_cache = False

# COMMAND ----------

# MAGIC %md
# MAGIC Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. Therefore we will add `dense`, `dense_h_to_4_h` and `dense_4h_to_h` layers in the target modules in addition to the mixed query key value layer.

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
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
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
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 100
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
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

trainer.save_model("/dbfs/falcon-40b-fine-tune")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load adapters from the checkpoint
# MAGIC With the fine-tuned saved to DBFS, we can load it afterwards for inference.
# MAGIC
# MAGIC The below code assumes that it is run in a separate notebook, and thus requires installing the packages.

# COMMAND ----------

# MAGIC %pip install peft==0.5.0
# MAGIC %pip install datasets==2.12.0 bitsandbytes==0.40.1 einops==0.6.1 trl==0.4.7
# MAGIC %pip install torch==2.0.1 accelerate==0.22.0 transformers==4.31.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "/dbfs/falcon-40b-fine-tune"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    return_dict=True, 
    load_in_8bit=True, 
    device_map={"":0},
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# COMMAND ----------

def generate_text(prompt, temperature=0.1, max_tokens=250):
    batch = tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = model.generate(
          input_ids = batch.input_ids, 
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return generated_text

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

### Response: """

print(generate_text(prompt))

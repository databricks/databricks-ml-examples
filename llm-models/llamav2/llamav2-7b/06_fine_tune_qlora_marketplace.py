# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune llama-2-7b-hf model from marketplace with QLORA
# MAGIC
# MAGIC Databricks hosts the llama-2 models in [Databricks marketplace](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models). This is a tutorial to show how to fine tune the models on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
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

model_mlflow_path = f"models:/{catalog_name}.models.llama_2_7b_hf/{version}"

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
# MAGIC In this section we will load the [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model installed from [Databricks marketplace]((https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models) saved in Unity Catalog to local disk, quantize it in 4bit and attach LoRA adapters on it.

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
    quantization_config=bnb_config,
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
    bf16=True,
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
# MAGIC ## Log the fine tuned model to MLFlow

# COMMAND ----------

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, PeftConfig

class LLAMAQLORA(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['tokenizer'])
    self.tokenizer.pad_token = self.tokenizer.eos_token
    config = PeftConfig.from_pretrained(context.artifacts['lora'])
    base_model = AutoModelForCausalLM.from_pretrained(
      context.artifacts['base_model'], 
      return_dict=True, 
      load_in_4bit=True, 
      device_map={"":0},
      trust_remote_code=True,
    )
    self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])
  
  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [1.0])[0]
    max_tokens = model_input.get("max_tokens", [100])[0]
    batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = self.model.generate(
          input_ids = batch.input_ids, 
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=self.tokenizer.eos_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
      )
    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import mlflow

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

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=LLAMAQLORA(),
        artifacts={'tokenizer' : tokenizer_path, "base_model": model_path,  "lora": model_output_local_path},
        pip_requirements=["torch", "transformers", "accelerate", "einops", "loralib", "bitsandbytes", "peft"],
        input_example=pd.DataFrame({"prompt":["what is ML?"], "temperature": [0.5],"max_tokens": [100]}),
        signature=signature
    )

# COMMAND ----------

registered_name = f"{catalog_name}.models.llama2_7b_fine_tune" # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

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


prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

### Response: """
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}/1")

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Predict on a Pandas DataFrame.
loaded_model.predict(text_example)

# COMMAND ----------



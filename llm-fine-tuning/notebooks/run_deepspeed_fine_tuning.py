# Databricks notebook source
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
from huggingface_hub import notebook_login, login

# notebook_login()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

# COMMAND ----------

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

from databricks_llm.notebook_utils import get_dbutils

# COMMAND ----------

DEFAULT_INPUT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "HuggingFaceH4/starchat-beta",
]

# COMMAND ----------

get_dbutils().widgets.text("num_gpus", "8", "num_gpus")
get_dbutils().widgets.text("dbfs_output_location", "/dbfs/llm/", "dbfs_output_location")
get_dbutils().widgets.combobox(
    "pretrained_name_or_path",
    DEFAULT_INPUT_MODEL,
    SUPPORTED_INPUT_MODELS,
    "pretrained_name_or_path",
)
get_dbutils().widgets.text(
    "dataset",
    "mlabonne/guanaco-llama2",
    "dataset",
)

# COMMAND ----------

num_gpus = get_dbutils().widgets.get("num_gpus")
pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
dataset = get_dbutils().widgets.get("dataset")
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")

# COMMAND ----------

# MAGIC !mkdir -p {dbfs_output_location}


# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '/local_disk0/output/runs'

# COMMAND ----------

# MAGIC  !cd .. && deepspeed \
# MAGIC --num_gpus="{num_gpus}" \
# MAGIC --module databricks_llm.fine_tune \
# MAGIC --final_model_output_path="{dbfs_output_location}" \
# MAGIC --output_dir="/local_disk0/output" \
# MAGIC --dataset={dataset} \
# MAGIC --model={pretrained_name_or_path} \
# MAGIC --tokenizer={pretrained_name_or_path} \
# MAGIC --use_lora=false \
# MAGIC --use_4bit=false \
# MAGIC --deepspeed_config="ds_configs/ds_zero_3_cpu_offloading.json" \
# MAGIC --fp16=false \
# MAGIC --bf16=true \
# MAGIC --per_device_train_batch_size=16 \
# MAGIC --per_device_eval_batch_size=48 \
# MAGIC --gradient_checkpointing=true \
# MAGIC --gradient_accumulation_steps=1 \
# MAGIC --learning_rate=5e-6 \
# MAGIC --adam_beta1=0.9 \
# MAGIC --adam_beta2=0.999 \
# MAGIC --adam_epsilon=1e-8 \
# MAGIC --lr_scheduler_type="cosine" \
# MAGIC --warmup_steps=100 \
# MAGIC --weight_decay=0.0 \
# MAGIC --evaluation_strategy="steps" \
# MAGIC --save_strategy="steps" \
# MAGIC --save_steps=100 \
# MAGIC --num_train_epochs=1

# COMMAND ----------

# MAGIC !ls -lah {dbfs_output_location}

# COMMAND ----------

print(dbfs_output_location)

# COMMAND ----------

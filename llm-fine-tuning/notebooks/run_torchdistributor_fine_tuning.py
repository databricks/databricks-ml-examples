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
import pathlib

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
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

from databricks_llm.notebook_utils import get_dbutils
from databricks_llm.utils import (
    copy_source_code,
    remote_login,
    check_mount_dev_shm,
    ExtendedTrainingArguments,
    resolve_fsdp_config,
    resolve_deepspeed_config,
)
from databricks_llm.fine_tune import train

from pyspark.ml.torch.distributor import TorchDistributor

check_mount_dev_shm()
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

get_dbutils().widgets.combobox("local_mode", "true", ["true", "false"], "local_mode")
get_dbutils().widgets.combobox("use_fsdp", "false", ["true", "false"], "use_fsdp")
get_dbutils().widgets.text("num_gpus", "4", "num_gpus")
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

num_gpus = int(get_dbutils().widgets.get("num_gpus"))
local_mode = get_dbutils().widgets.get("local_mode") == "true"
use_fsdp = get_dbutils().widgets.get("use_fsdp") == "true"
pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
dataset = get_dbutils().widgets.get("dataset")
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")

# COMMAND ----------
remote_login()
# COMMAND ----------

# MAGIC !cd .. && pip install -e .
# COMMAND ----------

deepspeed_config_dict = None
fsdp_config_dict = None

if use_fsdp:
    fsdp_config_dict = resolve_fsdp_config("fsdp_cpu_offloading.yaml")
else:
    deepspeed_config_dict = resolve_deepspeed_config("ds_zero_3_cpu_offloading.json")

args = ExtendedTrainingArguments(
    final_model_output_path=dbfs_output_location,
    output_dir="/local_disk0/output",
    dataset=dataset,
    model=pretrained_name_or_path,
    tokenizer=pretrained_name_or_path,
    use_lora=False,
    use_4bit=False,
    deepspeed_config=deepspeed_config_dict,
    fsdp_config=fsdp_config_dict,
    fp16=False,
    bf16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.0,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=1,
)


distributor = TorchDistributor(
    num_processes=num_gpus, local_mode=local_mode, use_gpu=True
)
distributor.run(train, args)

# COMMAND ----------

# MAGIC !ls -lah {dbfs_output_location}

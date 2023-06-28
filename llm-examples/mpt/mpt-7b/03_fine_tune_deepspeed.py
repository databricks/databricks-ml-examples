# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine tune MPT-7b with deepspeed
# MAGIC
# MAGIC This is to fine-tune [MPT-7b](https://huggingface.co/mosaicml/mpt-7b) models on the [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset. The MPT models are [Apache 2.0 licensed](https://huggingface.co/mosaicml/mpt-7b). sciq is under CC BY-SA 3.0 license.
# MAGIC
# MAGIC  Environment:
# MAGIC - 

# COMMAND ----------

# MAGIC %pip install torch==2.0.1 accelerate==0.18.0 deepspeed==0.9.4 transformers[torch]==4.28.1 peft einops
# MAGIC %pip install xformers einops flash-attn==v1.0.3.post0 triton==2.0.0.dev20221202

# COMMAND ----------

import os

os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf'

# COMMAND ----------

# MAGIC %md
# MAGIC The fine tune logic is written in `fine_tune_deepspeed.py`.

# COMMAND ----------

# MAGIC %md
# MAGIC Fine tune with `deepspeed`.

# COMMAND ----------

# MAGIC %sh
# MAGIC deepspeed --num_gpus=2 scripts/fine_tune_deepspeed.py

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/local_disk0/output`.

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /local_disk0/output/checkpoint-20

# COMMAND ----------



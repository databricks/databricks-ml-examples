# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine tune llama-2-70b with LoRA and deepspeed on a single node
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) is the 7B pretrained model, converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC This is to fine-tune [llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) models on the [dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `Standard_NC48ads_A100_v4` on Azure with 2 A100-80GB GPUs, `p4d.24xlarge` on AWS with 8 A100-40GB GPUs
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Install the missing libraries

# COMMAND ----------

# MAGIC %pip install deepspeed==0.9.5 xformers
# MAGIC %pip install git+https://github.com/huggingface/peft.git
# MAGIC %pip install bitsandbytes==0.40.1 einops==0.6.1 trl==0.4.7
# MAGIC %pip install -U torch==2.0.1 accelerate==0.21.0 transformers==4.31.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine tune the model with `deepspeed`
# MAGIC
# MAGIC The fine tune logic is written in `scripts/fine_tune_deepspeed.py`. The dataset used for fine tune is [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC deepspeed \
# MAGIC --num_gpus 2 \
# MAGIC scripts/fine_tune_lora.py \
# MAGIC --output_dir="/local_disk0/output"

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/local_disk0/final_model`.

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /local_disk0/final_model

# COMMAND ----------



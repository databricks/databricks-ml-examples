# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune llama-2-7b with composer
# MAGIC
# MAGIC This notebook demonstrates fine-tuning the `Llama-2-7b-hf` model with the deep learning library [Composer](https://github.com/mosaicml/composer), which is developed by MosaicML.
# MAGIC
# MAGIC The main benefits of using Composer for fine-tuning include:
# MAGIC - Integration with Pytorch's FSDP: Similar as DeepSpeed, FSDP is a framework for efficient parallelism among workers.
# MAGIC - Auto-resumption: Composer could autoresume the training job from saved checkpoints.
# MAGIC - CUDA OOM Prevention: With the training microbatch size set to “auto”, Composer will automatically select the biggest one that fits on your GPUs.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.3 GPU ML Runtime
# MAGIC - Instance: `Standard_NC96ads_A100_v4` on Azure with 4 A100-80GB GPUs, `p4d.24xlarge` on AWS with 8 A100-40GB GPUs
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %pip install protobuf==3.20.2
# MAGIC %pip install mosaicml
# MAGIC %pip install llm-foundry
# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC In the below command,
# MAGIC - `scripts/fine_tune_composer.yaml` is the YAML file that contains configurations including the Hugging Face model name, training and evaluation dataset ([mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf)), FSDP configuration, checkpoint settting etc. This is the file you would change if you want to customize the fine-tuning setup.
# MAGIC - `scripts/fine_tune_composer.py` is a simplified copy from MosaicML's [LLM Foundry](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/train.py), which parses the YAML file and calls the Composer library.
# MAGIC
# MAGIC  To fully utilize Composer's features, in the YAML file:
# MAGIC   - Although not required for training, `run_name` and `save_folder` are set so that auto-resume is enabled.
# MAGIC   - `device_train_microbatch_size` is set as `auto` to enable Composer to automatically select the local batch size.

# COMMAND ----------

NUM_GPUS = 4  # number of gpus in this node

!composer --world_size {NUM_GPUS} scripts/fine_tune_composer.py scripts/fine_tune_composer.yaml

# COMMAND ----------

# MAGIC %md
# MAGIC A symlink `latest-rank0.pt` would be created in the `save_folder`, and it points to the latest checkpoint saved in the folder.

# COMMAND ----------

# MAGIC %ls /local_disk0/composer-llama-2-7b-fine-tune/checkpoints

# COMMAND ----------

# MAGIC %md
# MAGIC Checkpoints created by Composer are not exactly the same as Hugging Face checkpoints. To load the fine-tuned model for inference with Hugging Face, we need to run [convert_composer_to_hf.py](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/convert_composer_to_hf.py).

# COMMAND ----------

COMPOSER_CHECKPOINT_PATH = "/local_disk0/composer-llama-2-7b-fine-tune/checkpoints/latest-rank0.pt"
# Update the following path to point to a UC Volumes location if DBFS is not available
# Make sure to change all related paths in the notebook 
HF_CHECKPOINT_PATH = "/dbfs/composer-llama-2-7b-fine-tune/"

!python scripts/convert_composer_to_hf.py \
  --composer_path {COMPOSER_CHECKPOINT_PATH} \
  --hf_output_path {HF_CHECKPOINT_PATH} \
  --output_precision bf16

# COMMAND ----------

# MAGIC %ls /dbfs/composer-llama-2-7b-fine-tune/

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer
import transformers
import torch

fine_tuned_model = "/dbfs/composer-llama-2-7b-fine-tune"

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
pipeline = transformers.pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    return_full_text=False
)

pipeline("Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: What are 6 different reasons to go on vacation? ### Response:", max_new_tokens=512)

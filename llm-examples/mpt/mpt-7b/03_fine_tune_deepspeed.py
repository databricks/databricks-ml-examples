from functools import partial
import json
import logging
import numpy as np
import os
import torch

from datasets import Dataset, load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf'

logger = logging.getLogger(__name__)

MODEL_PATH = 'mosaicml/mpt-7b-instruct'
TOKENIZER_PATH = 'EleutherAI/gpt-neox-20b'
DEFAULT_TRAINING_DATASET = "sciq"

def load_training_dataset(
  tokenizer,
  path_or_dataset: str = DEFAULT_TRAINING_DATASET,
  ) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)
    logger.info("Found %d rows", dataset.num_rows)

    def _reformat_data(rec):
      PROMPT_FORMAT = f"{rec['question']}\n wrong answers are:{rec['distractor1']} - {rec['distractor2']} - {rec['distractor3']}\n Correct Answer: "
      questions = PROMPT_FORMAT
      answer = rec["correct_answer"]
      return {"text": f"{{ 'prompt': {questions}, 'response': {answer} }}"}
    
    dataset = dataset.map(_reformat_data)

    def tokenize_function(allEntries):
      return tokenizer(allEntries['text'], padding='max_length', truncation=True)
    
    dataset = dataset.map(tokenize_function)
    train_tokenized_dataset = dataset['train']
    eval_tokenized_dataset = dataset['validation']

    return train_tokenized_dataset, eval_tokenized_dataset
      


def load_model(
    pretrained_model_name_or_path: str = MODEL_PATH,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                        trust_remote_code=True
                                        )
    config.attn_config['attn_impl'] = 'triton'

    model = transformers.AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path,
      config=config,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True
    )
    return model

def get_tokenizer(
    pretrained_tokenizer_name_or_path: str = TOKENIZER_PATH,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

ds_config_dict = {
  "gradient_accumulation_steps": 'auto',
  "train_micro_batch_size_per_gpu": 'auto',
  "steps_per_print": 100,
  "optimizer": {
    "type": "Adam",
    "params": {
        "lr": 2e-5,
        "weight_decay": 1
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
  "flops_profiler": {
    "enabled": False,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": True
  },
  "fp16": {
    "enabled": True,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "cpu_offload": True,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    },
    "contiguous_gradients": True,
    "overlap_comm": True,
    "reduce_scatter": False,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7
  },
  "activation_checkpointing": {
    "partition_activations": False,
    "contiguous_memory_optimization": False,
    "cpu_checkpointing": False
  },
  "wall_clock_breakdown": False,
  "zero_allow_untested_optimizer": True,
}

def train():
  model = load_model()
  device = 'cuda'
  model.to(device)
  tokenizer = get_tokenizer()
  train_dataset, val_dataset = load_training_dataset(tokenizer)

  training_args = TrainingArguments(
    output_dir="/local_disk0/output", 
    learning_rate=2e-5,
    num_train_epochs=1, # 3
    weight_decay=1,
    do_eval=True,
    evaluation_strategy="epoch",
    fp16=True,
    deepspeed=ds_config_dict,
    save_strategy="steps",
    save_steps=20, # 500
    max_steps=20,
    report_to=[],
  )
  
  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
  )

  trainer.train()

if __name__ == "__main__":
  train()
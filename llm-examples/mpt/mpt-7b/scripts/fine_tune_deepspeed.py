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

MODEL_PATH = 'mosaicml/mpt-7b'
TOKENIZER_PATH = 'EleutherAI/gpt-neox-20b'
DEFAULT_TRAINING_DATASET = "sciq"
CONFIG_PATH = "../../config/a10_config.json"

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
    config.init_device = 'cuda'
    model_hidden_size = config.d_model

    model = transformers.AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path,
      config=config,
      torch_dtype=torch.float16,
      trust_remote_code=True
    )
    return model, model_hidden_size

def get_tokenizer(
    pretrained_tokenizer_name_or_path: str = TOKENIZER_PATH,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

with open(CONFIG_PATH) as config_json:
  ds_config_dict = json.load(config_json)

def train():
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
  # Enable tf32 for better performance
  torch.backends.cuda.matmul.allow_tf32 = True

  model, model_hidden_size = load_model()
  device = 'cuda'
  model.to(device)
  tokenizer = get_tokenizer()
  train_dataset, val_dataset = load_training_dataset(tokenizer)
  ds_config_dict["hidden_size"] = model_hidden_size
  ds_config_dict["zero_optimization"]["reduce_bucket_size"] = model_hidden_size*model_hidden_size
  ds_config_dict["zero_optimization"]["stage3_prefetch_bucket_size"] = 0.9 * model_hidden_size * model_hidden_size
  ds_config_dict["zero_optimization"]["stage3_param_persistence_threshold"] = 10 * model_hidden_size

  training_args = TrainingArguments(
    output_dir="/local_disk0/output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    num_train_epochs=1, # 3
    weight_decay=1,
    do_eval=True,
    evaluation_strategy="epoch",
    fp16=False,
    bf16=True,
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
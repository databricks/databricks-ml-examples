import click
from dataclasses import field, dataclass
from functools import partial
import json
import logging
import os
import numpy as np
from pathlib import Path
import torch
from typing import Optional, Union

from datasets import Dataset, load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    IntervalStrategy,
    PreTrainedTokenizer,
    SchedulerType,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  response_key=RESPONSE_KEY,
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  input_key=INPUT_KEY,
  input="{input}",
  response_key=RESPONSE_KEY,
)


ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = 'meta-llama/Llama-2-7b-hf'
TOKENIZER_PATH = 'meta-llama/Llama-2-7b-hf'
DEFAULT_TRAINING_DATASET = "databricks/databricks-dolly-15k"
CONFIG_PATH = "../../config/a10_config.json"
LOCAL_OUTPUT_DIR = "/dbfs/llama-2-fine-tune/output"
DEFAULT_SEED = 68


@dataclass
class HFTrainingArguments:
  local_rank: Optional[str] = field(default="-1")
  dataset: Optional[str] = field(default=DEFAULT_TRAINING_DATASET)
  model: Optional[str] = field(default=MODEL_PATH)
  tokenizer: Optional[str] = field(default=TOKENIZER_PATH)
  config_path: Optional[str] = field(default=CONFIG_PATH)
  max_seq_len: Optional[int] = field(default=256)

  final_model_output_path: Optional[str] = field(default="/local_disk0/final_model")
  deepspeed_config: Optional[str] = field(default=CONFIG_PATH)

  output_dir: Optional[str] = field(default="/local_disk0/final_model")
  per_device_train_batch_size: Optional[int] = field(default=1)
  per_device_eval_batch_size: Optional[int] = field(default=1)
  gradient_checkpointing: Optional[bool] = field(default=True)
  gradient_accumulation_steps: Optional[int] = field(default=1)
  learning_rate: Optional[float] = field(default=3e-4)
  optim: Optional[str] = field(default="adamw_hf")
  num_train_epochs: Optional[int] = field(default=3)
  max_steps: Optional[int] = field(default=-1)
  adam_beta1: float = field(default=0.9)
  adam_beta2: float = field(default=0.95)
  adam_epsilon: float = field(default=1e-4)
  lr_scheduler_type: Union[SchedulerType, str] = field(
      default="cosine",
  )
  warmup_steps: int = field(default=5)
  weight_decay: Optional[float] = field(default=0.1)
  logging_strategy: Optional[Union[str, IntervalStrategy]] = field(
      default=IntervalStrategy.STEPS
  )
  evaluation_strategy: Optional[Union[str, IntervalStrategy]] = field(
      default=IntervalStrategy.STEPS
  )
  save_strategy: Optional[Union[str, IntervalStrategy]] = field(
      default=IntervalStrategy.STEPS
  )
  fp16: Optional[bool] = field(default=False)
  bf16: Optional[bool] = field(default=True)
  save_steps: Optional[int] = field(default=100)
  logging_steps: Optional[int] = field(default=10)


def load_training_dataset(
  tokenizer,
  path_or_dataset: str = DEFAULT_TRAINING_DATASET,
  max_seq_len: int = 256,
  seed: int = DEFAULT_SEED,
  ) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info(f"Found {dataset.num_rows} rows", )

    def _reformat_data(rec):
      instruction = rec["instruction"]
      response = rec["response"]
      context = rec.get("context")

      if context:
        questions = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, input=context)
      else:
        questions = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction)
      return {"text": f"{questions}\n {response}"} 

    def tokenize_function(element):
      outputs = tokenizer(
        _reformat_data(element)["text"], 
        truncation=True,
        padding=True,
        max_length=max_seq_len,
        return_overflowing_tokens=False,
        )
      return outputs
    
    dataset = dataset.map(tokenize_function)
  
    split_dataset = dataset.train_test_split(test_size=1000, seed=seed)
    train_tokenized_dataset = split_dataset['train']
    eval_tokenized_dataset = split_dataset['test']

    return train_tokenized_dataset, eval_tokenized_dataset
      


def load_model(
    pretrained_model_name_or_path: str = MODEL_PATH,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path,
      torch_dtype=torch.bfloat16,
      device_map=None,
      low_cpu_mem_usage=True,
      trust_remote_code=True
    )

    return model

def get_tokenizer(
    pretrained_tokenizer_name_or_path: str = TOKENIZER_PATH,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def train(args: HFTrainingArguments):
  set_seed(DEFAULT_SEED)
  torch.backends.cuda.matmul.allow_tf32 = True

  tokenizer = get_tokenizer(args.tokenizer)
  train_dataset, val_dataset = load_training_dataset(tokenizer, max_seq_len=256, seed=DEFAULT_SEED)
  model = load_model(pretrained_model_name_or_path=args.model)

  with open(args.config_path) as config_json:
    ds_config_dict = json.load(config_json)

  training_args = TrainingArguments(
    local_rank=args.local_rank,
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    learning_rate=args.learning_rate,
    optim=args.optim,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_epsilon,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    logging_strategy=args.logging_strategy,
    evaluation_strategy=args.evaluation_strategy,
    save_strategy=args.save_strategy,
    fp16=args.fp16,
    bf16=args.bf16,
    deepspeed=ds_config_dict,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=5,
    push_to_hub=False,
    disable_tqdm=True,
    report_to=[],
  )
  
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
  )

  logger.info("Training the model")
  trainer.train()

  logger.info(f"Saving Model to {local_output_dir}")
  trainer.save_model(output_dir=local_output_dir)
  tokenizer.save_pretrained(local_output_dir)

  if dbfs_output_dir:
    logger.info(f"Saving Model to {dbfs_output_dir}")
    trainer.save_model(output_dir=dbfs_output_dir)
    tokenizer.save_pretrained(dbfs_output_dir)

  logger.info("Training finished.")


def main(**kwargs):
  parser = HfArgumentParser(HFTrainingArguments)
  parsed = parser.parse_args_into_dataclasses()
  args = parsed[0]


  train(args)

if __name__ == "__main__":
  os.environ['HF_HOME'] = '/local_disk0/hf'
  os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
  os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf'

  logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
  )
  try:
    main()
  except Exception:
    logger.exception("main failed")
    raise
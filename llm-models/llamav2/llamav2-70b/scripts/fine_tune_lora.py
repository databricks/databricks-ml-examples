import bitsandbytes as bnb
import logging
import math
import os
import sys
import json
import random
from pathlib import Path

from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional,List,Union

import datasets
import evaluate
import torch
from datasets import Dataset, load_dataset
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    IntervalStrategy,
    LlamaTokenizer,
    Seq2SeqTrainer,
    PreTrainedTokenizer,
    SchedulerType,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    set_seed,
)

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = 'meta-llama/Llama-2-70b-hf'
TOKENIZER_PATH = 'meta-llama/Llama-2-70b-hf'
DEFAULT_TRAINING_DATASET = "mosaicml/dolly_hhrlhf"
CONFIG_PATH = "../../config/a10_config_zero2.json"
LOCAL_OUTPUT_DIR = "/dbfs/llama-2-fine-tune/output"
TRANSFORMER_CACHE = "/local_disk0/hf"
DEFAULT_PAD_TOKEN = "[PAD]"
IGNORE_INDEX = -100
DEFAULT_SEED = 68


@dataclass
class HFTrainingArguments:
    local_rank: Optional[str] = field(default="-1")
    dataset: Optional[str] = field(default=DEFAULT_TRAINING_DATASET)
    cache_dir: Optional[str] = field(default=TRANSFORMER_CACHE)
    use_auth_token: Optional[str] = field(default="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")
    model: Optional[str] = field(default=MODEL_PATH)
    tokenizer: Optional[str] = field(default=TOKENIZER_PATH)
    max_seq_len: Optional[int] = field(default=256)

    final_model_output_path: Optional[str] = field(default="/local_disk0/final_model")

    deepspeed_config: Optional[str] = field(default=CONFIG_PATH)

    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    output_dir: Optional[str] = field(default="/local_disk0/output")
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-6)
    optim: Optional[str] = field(default="adamw_hf")
    num_train_epochs: Optional[int] = field(default=1)
    max_steps: Optional[int] = field(default=-1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="cosine",
    )
    warmup_steps: int = field(default=0)
    weight_decay: Optional[float] = field(default=1)
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


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
  

def load_training_dataset(
    tokenizer,
    path_or_dataset: str = DEFAULT_TRAINING_DATASET,
    max_seq_len: int = 256,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)
    logger.info(f"Training: found {dataset['train'].num_rows} rows")
    logger.info(f"Eval: found {dataset['test'].num_rows} rows")

    # Reformat input data, add prompt template if needed
    def _reformat_data(row):
      return row["prompt"] + row["response"]

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        input_batch = []
        attention_masks = []

        outputs = tokenizer(
            _reformat_data(element),
            truncation=True,
            padding=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=True,
        )

        for length, input_ids, attention_mask in zip(
            outputs["length"], outputs["input_ids"], outputs["attention_mask"]
        ):
            if length == max_seq_len:
                input_batch.append(input_ids)
                attention_masks.append(attention_mask)

        return {"input_ids": input_batch, "attention_mask": attention_masks}

    train_tokenized_dataset = dataset["train"].map(
        tokenize, batched=True, remove_columns=dataset["train"].column_names
    )
    eval_tokenized_dataset = dataset["test"].map(
        tokenize, batched=True, remove_columns=dataset["test"].column_names
    )

    return train_tokenized_dataset, eval_tokenized_dataset

def get_model(args) -> AutoModelForCausalLM:
    logger.info(f"Loading model: {args.model}")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        use_auth_token=args.use_auth_token
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    logger.info("Loading adapters.")
    modules = find_all_linear_names(args, model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def get_tokenizer(args, model) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama',
        use_auth_token=args.use_auth_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # LLaMA tokenizer may not have correct special tokens set.
    # Check and add them if missing to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    logger.info('Adding special tokens.')
    tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
    })
    return tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )  

def train(args: HFTrainingArguments):
  set_seed(DEFAULT_SEED)
  torch.backends.cuda.matmul.allow_tf32 = True

  if args.deepspeed_config:
    with open(args.deepspeed_config) as json_data:
      deepspeed_config_dict = json.load(json_data)
  else:
    deepspeed_config_dict = None
  
  training_args = TrainingArguments(
    local_rank=args.local_rank,
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
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
    deepspeed=deepspeed_config_dict,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    push_to_hub=False,
    disable_tqdm=True,
    report_to=[],
    # group_by_length=True,
    ddp_find_unused_parameters=False,
    # fsdp=["full_shard", "offload"],
  )
  model = get_model(args)
  model.print_trainable_parameters()
  tokenizer = get_tokenizer(args, model)
  train_dataset, eval_dataset = load_training_dataset(
    tokenizer, path_or_dataset=args.dataset, max_seq_len=args.max_seq_len
  )

  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
  )

  logger.info("Training the model")
  trainer.train()

  logger.info(f"Saving Model to {args.final_model_output_path}")
  trainer.save_model(output_dir=args.final_model_output_path)
  tokenizer.save_pretrained(args.final_model_output_path)

  logger.info("Training finished.")


def main():
  parser = HfArgumentParser(HFTrainingArguments)

  parsed = parser.parse_args_into_dataclasses()
  args: HFTrainingArguments = parsed[0]

  train(args)


if __name__ == "__main__":
  os.environ["HF_HOME"] = "/local_disk0/hf"
  os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
  os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

  logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  try:
    main()
  except Exception:
    logger.exception("main failed")
    raise

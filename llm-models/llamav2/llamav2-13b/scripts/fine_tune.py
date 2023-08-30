from dataclasses import field, dataclass
import json
import logging
import os
from pathlib import Path
import torch
import transformers
from typing import Optional, Union, Tuple

from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

from huggingface_hub import login

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    IntervalStrategy,
    PreTrainedTokenizer,
    SchedulerType,
    TrainingArguments,
    Trainer,
    set_seed,
)

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = 'meta-llama/Llama-2-13b-hf'
TOKENIZER_PATH = 'meta-llama/Llama-2-13b-hf'
DEFAULT_TRAINING_DATASET = "mosaicml/dolly_hhrlhf"
CONFIG_PATH = "../../config/a10_config.json"
LOCAL_OUTPUT_DIR = "/dbfs/llama-2-13b-fine-tune/output"
DEFAULT_SEED = 68



@dataclass
class HFTrainingArguments:
    local_rank: Optional[str] = field(default="-1")
    dataset: Optional[str] = field(default=DEFAULT_TRAINING_DATASET)
    model: Optional[str] = field(default=MODEL_PATH)
    tokenizer: Optional[str] = field(default=TOKENIZER_PATH)
    max_seq_len: Optional[int] = field(default=256)

    final_model_output_path: Optional[str] = field(default="/local_disk0/final_model")

    deepspeed_config: Optional[str] = field(default=CONFIG_PATH)

    output_dir: Optional[str] = field(default=LOCAL_OUTPUT_DIR)
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-6)
    optim: Optional[str] = field(default="adamw_hf")
    num_train_epochs: Optional[int] = field(default=None)
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


def get_model(
    pretrained_name_or_path: str,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model: {pretrained_name_or_path}")

    config = AutoConfig.from_pretrained(pretrained_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_name_or_path,
        config=config,
        trust_remote_code="true",
        torch_dtype=torch.bfloat16,
        device_map= None,
    )

    model.config.use_cache = False

    return model


def get_tokenizer(
    pretrained_name_or_path: str,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name_or_path, trust_remote_code="true", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_and_tokenizer(
    pretrained_name_or_path: str,
    pretrained_name_or_path_tokenizer: str = None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = get_tokenizer(
        pretrained_name_or_path_tokenizer
        if pretrained_name_or_path_tokenizer is not None
        else pretrained_name_or_path,
    )
    model = get_model(pretrained_name_or_path)
    return model, tokenizer


def load_training_dataset(
    tokenizer,
    path_or_dataset: str,
    split: str,
    dataset_text_field: str = "text",
    max_seq_len: int = 512,
    formatting_func=None,
) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    if path_or_dataset.startswith("/"):
        dataset = load_from_disk(path_or_dataset)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
            print(
                f"Loaded dataset {path_or_dataset} from disk for split {split} with {len(dataset)} rows."
            )
    else:
        dataset = load_dataset(path_or_dataset, split=split)
        print(
            f"Loaded dataset {path_or_dataset} from HF Hub for split {split} with {len(dataset)} rows."
        )
    logger.info("Found %d rows", dataset.num_rows)
    logger.info("Found %d rows", len(dataset))

    use_formatting_func = formatting_func is not None and dataset_text_field is None

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        input_batch = []
        attention_masks = []

        outputs = tokenizer(
            formatting_func(element),
            truncation=True,
            padding=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=True,
        )

        for length, input_ids, attention_mask in zip(
            outputs["length"], outputs["input_ids"], outputs["attention_mask"]
        ):
            # if length == max_seq_len:
            input_batch.append(input_ids)
            attention_masks.append(attention_mask)

        return {"input_ids": input_batch, "attention_mask": attention_masks}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )

    print(len(tokenized_dataset))

    return tokenized_dataset


def setup_hf_trainer(
    train_dataset, eval_dataset=None, **config
) -> Tuple[Trainer, AutoModelForCausalLM, PreTrainedTokenizer]:
    args: ExtendedTrainingArguments = config["args"]

    torch.backends.cuda.matmul.allow_tf32 = True

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
        deepspeed=config.get("deepspeed_config_dict", None),
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        report_to=["tensorboard"],
        # group_by_length=True,
        ddp_find_unused_parameters=False,
        # fsdp=["full_shard", "offload"],
    )

    model, tokenizer = get_model_and_tokenizer(args.model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer, model, tokenizer

def _reformat_data(row):
  return row["prompt"] + row["response"]

def train(args: HFTrainingArguments):
  set_seed(DEFAULT_SEED)
  tokenizer = get_tokenizer(args.tokenizer)
  train_dataset = load_training_dataset(
      tokenizer, args.dataset, "train", "text", 256, formatting_func=_reformat_data
  )
  eval_dataset = load_training_dataset(
    tokenizer, args.dataset, "test", "text", 256, formatting_func=_reformat_data
  )
  if args.deepspeed_config:
    with open(args.deepspeed_config) as json_data:
      deepspeed_config_dict = json.load(json_data)
  else:
    deepspeed_config_dict = None
  trainer, model, tokenizer = setup_hf_trainer(
    train_dataset,
    eval_dataset,
    args=args,
    deepspeed_config_dict=deepspeed_config_dict,
  )
  trainer.train()
  trainer.save_model(args.final_model_output_path)
  tokenizer.save_pretrained(args.final_model_output_path)

def main():
    parser = HfArgumentParser(HFTrainingArguments)

    parsed = parser.parse_args_into_dataclasses()
    args: HFTrainingArguments = parsed[0]

    train(args)


if __name__ == "__main__":
    os.environ["HF_HOME"] = "/local_disk0/hf"
    os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # os.environ["NCCL_DEBUG"] = "INFO"

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

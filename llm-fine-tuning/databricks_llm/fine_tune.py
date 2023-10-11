import json
import logging

import os
from typing import Tuple, Dict
import pathlib
import torch
import torch.distributed
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from accelerate.utils import PrecisionType

from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

from huggingface_hub import login
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
)

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
)

from databricks_llm.model_utils import get_model_and_tokenizer, get_tokenizer
from databricks_llm.utils import ExtendedTrainingArguments

LOCAL_DISK_HF = "/local_disk0/hf"

logger = logging.getLogger(__name__)


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
            element[dataset_text_field]
            if not use_formatting_func
            else formatting_func(element),
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
    train_dataset,
    eval_dataset=None,
    **config,
) -> Tuple[Trainer, AutoModelForCausalLM, PreTrainedTokenizer]:
    args: ExtendedTrainingArguments = config["args"]

    torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
        # local_rank=args.local_rank,
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
        fsdp_config=config.get("fsdp_config_dict", None),
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        report_to=["tensorboard"],
        # group_by_length=True,
        ddp_find_unused_parameters=False,
    )

    model, tokenizer = get_model_and_tokenizer(
        args.model, use_4bit=args.use_4bit, load_in_8bit=False, use_lora=args.use_lora
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # if fsdp_pluin:
    #     os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    #     os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "LlamaDecoderLayer"
    #     fsdp_pluin.set_auto_wrap_policy(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer, model, tokenizer


def train(args: ExtendedTrainingArguments):
    set_up_hf_cache()
    handle_hf_key(args)
    handle_fsdp_params(args)

    # if args.fsdp_config:
    #     fsdp_plugin = get_fsdp_plugin(args)
    #     if args.bf16 or args.fp16:
    #         fsdp_plugin.set_mixed_precision(
    #             "bf16" if args.bf16 else "fp16" if args.fp16 else "error!"
    #         )
    #
    #     accelerator = Accelerator(
    #         fsdp_plugin=fsdp_plugin,
    #         mixed_precision=PrecisionType.BF16
    #         if args.bf16
    #         else PrecisionType.FP16
    #         if args.bf16
    #         else PrecisionType.NO,
    #     )
    #     accelerator.print(accelerator.distributed_type)
    #     print("World Size = ", torch.distributed.get_world_size())
    #     print("Current rank = ", torch.distributed.get_rank())
    # else:
    #     fsdp_plugin = None

    tokenizer = get_tokenizer(args.tokenizer)
    train_dataset = load_training_dataset(
        tokenizer, args.dataset, "train", "text", 256, formatting_func=None
    )
    eval_dataset = load_training_dataset(
        tokenizer, args.dataset, "test", "text", 256, formatting_func=None
    )
    if isinstance(args.deepspeed_config, str):
        with open(args.deepspeed_config) as json_data:
            deepspeed_config_dict = json.load(json_data)
    elif isinstance(args.deepspeed_config, dict):
        deepspeed_config_dict = args.deepspeed_config
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


def handle_fsdp_params(args):
    if args.fsdp_config and isinstance(args.fsdp_config, Dict):
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        for k, v in args.fsdp_config.items():
            os.environ[k.upper()] = str(v)


# def get_fsdp_plugin(args: ExtendedTrainingArguments):
#     use_cpu_offloading = False
#     fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
#     fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
#     fsdp_offload_params = False
#     fsdp_offload_optimizer = False
#
#     if isinstance(args.fsdp_config, Dict):
#         fsdp_config_dict: Dict = args.fsdp_config
#         fsdp_offload_params = fsdp_config_dict.get("fsdp_offload_params", False)
#         fsdp_offload_optimizer = fsdp_config_dict.get("fsdp_offload_optimizer", False)
#         if fsdp_config_dict.get("fsdp_offload_params") or fsdp_config_dict.get(
#             "fsdp_offload_optimizer"
#         ):
#             use_cpu_offloading = True
#         if fsdp_config_dict.get("fsdp_sharding_strategy"):
#             fsdp_sharding_strategy = ShardingStrategy._member_map_.get(
#                 fsdp_config_dict.get("fsdp_sharding_strategy"),
#                 ShardingStrategy.FULL_SHARD,
#             )
#         if fsdp_config_dict.get("fsdp_state_dict_type"):
#             fsdp_state_dict_type = ShardingStrategy._member_map_.get(
#                 fsdp_config_dict.get("fsdp_state_dict_type"),
#                 fsdp_state_dict_type.FULL_STATE_DICT,
#             )
#     fsdp_plugin = FullyShardedDataParallelPlugin(
#         cpu_offload=CPUOffload(offload_params=use_cpu_offloading),
#         limit_all_gathers=True,
#         sharding_strategy=fsdp_sharding_strategy,
#         state_dict_type=fsdp_state_dict_type,
#         state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#         if fsdp_offload_params
#         else None,
#         optim_state_dict_config=FullOptimStateDictConfig(
#             offload_to_cpu=True, rank0_only=True
#         )
#         if fsdp_offload_optimizer
#         else None,
#     )
#     return fsdp_plugin


def main():
    parser = HfArgumentParser(ExtendedTrainingArguments)

    parsed = parser.parse_args_into_dataclasses()
    args: ExtendedTrainingArguments = parsed[0]

    train(args)


def handle_hf_key(args):
    if args.token is not None and len(args.token):
        login(args.token)
    elif pathlib.Path("/root/.cache/huggingface/token").exists():
        login(pathlib.Path("/root/.cache/huggingface/token").read_text())


def set_up_hf_cache():
    pathlib.Path(LOCAL_DISK_HF).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = LOCAL_DISK_HF
    os.environ["HF_DATASETS_CACHE"] = LOCAL_DISK_HF
    os.environ["TRANSFORMERS_CACHE"] = LOCAL_DISK_HF
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"


if __name__ == "__main__":
    set_up_hf_cache()

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

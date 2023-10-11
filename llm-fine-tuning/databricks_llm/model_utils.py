import logging
from typing import Tuple

import torch
import transformers
from peft import LoraConfig, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
)

logger = logging.getLogger(__name__)


def get_model(
    pretrained_name_or_path: str,
    use_4bit: bool = False,
    load_in_8bit: bool = False,
    use_lora: bool = False,
    inference: bool = False,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model: {pretrained_name_or_path}")
    if use_4bit and load_in_8bit:
        raise Exception("Cannot use 8bit and 4bit in the same time!")
    if use_4bit:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    config = AutoConfig.from_pretrained(pretrained_name_or_path, trust_remote_code=True)
    if config.model_type == "mpt":
        config.attn_config["attn_impl"] = "triton"
        if inference:
            config.init_device = "cuda:0"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_name_or_path,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code="true",
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        device_map="auto" if inference else None,
    )

    if use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
    use_4bit: bool = False,
    load_in_8bit: bool = False,
    use_lora: bool = False,
    inference: bool = False,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = get_tokenizer(
        pretrained_name_or_path_tokenizer
        if pretrained_name_or_path_tokenizer is not None
        else pretrained_name_or_path,
    )
    model = get_model(
        pretrained_name_or_path,
        load_in_8bit=load_in_8bit,
        use_4bit=use_4bit,
        use_lora=use_lora,
        inference=inference,
    )
    return model, tokenizer

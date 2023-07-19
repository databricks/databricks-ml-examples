import click
import json
import logging
import os
from pathlib import Path
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


os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

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
MODEL_PATH = "Salesforce/xgen-7b-8k-base"
TOKENIZER_PATH = "Salesforce/xgen-7b-8k-base"
DEFAULT_TRAINING_DATASET = "databricks/databricks-dolly-15k"
CONFIG_PATH = "../../config/a100_config.json"
LOCAL_OUTPUT_DIR = "/local_disk0/output"
DEFAULT_SEED = 68
REVISION = "3987e094377fae577bba039af1b300ee8086f9e1"
MAX_SEQ_LEN = 512


def load_training_dataset(
        tokenizer,
        path_or_dataset: str = DEFAULT_TRAINING_DATASET,
        seed: int = DEFAULT_SEED
) -> Dataset:
    """
    This function is used for preprocessing the databricks-dolly-15k dataset.
    To fine-tune on your own dataset, you would need to customize the function.
    """

    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info(f"Found {dataset.num_rows} rows", )

    def _reformat_data(rec):
        # Each row of databricks-dolly-15k contains fields "instruction", "response", and optionally the "context" field
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if context:
            questions = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, input=context)
        else:
            questions = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction)

        return {"text": f"{questions}\n{response}"}

    dataset = dataset.map(_reformat_data)

    def tokenize_function(allEntries):
        return tokenizer(allEntries['text'], truncation=True, max_length=MAX_SEQ_LEN)

    dataset = dataset.map(tokenize_function)

    # databricks-dolly-15k only contains the "train" split, so we split it to get an evaluation set
    split_dataset = dataset.train_test_split(test_size=1000, seed=seed)
    train_tokenized_dataset = split_dataset['train']
    eval_tokenized_dataset = split_dataset['test']

    return train_tokenized_dataset, eval_tokenized_dataset

def load_model(
        pretrained_model_name_or_path: str = MODEL_PATH,
        bf16: bool = False,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        revision=REVISION,
    )

    torch_dtype = torch.bfloat16 if bf16 else torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        revision=REVISION,
    )

    model.config.use_cache = False

    return model

def get_tokenizer(
        pretrained_tokenizer_name_or_path: str = TOKENIZER_PATH,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_tokenizer_name_or_path,
        trust_remote_code=True,
        revision=REVISION,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

with open(CONFIG_PATH) as config_json:
    ds_config_dict = json.load(config_json)

def train(
        *,
        input_model: str,
        local_output_dir: str,
        dbfs_output_dir: str,
        epochs: int,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        lr: float,
        seed: int,
        gradient_checkpointing: bool,
        gradient_accumulation_steps: int,
        local_rank: str,
        bf16: bool,
        logging_steps: int,
        save_steps: int,
        max_steps: int,
        eval_steps: int,
        save_total_limit: int,
        warmup_steps: int,
):
    set_seed(seed)
    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = get_tokenizer()
    train_dataset, val_dataset = load_training_dataset(tokenizer, seed=seed)

    model = load_model(pretrained_model_name_or_path=input_model, bf16=bf16)

    # enable fp16 if not bf16
    fp16 = not bf16

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=1,
        do_eval=True,
        evaluation_strategy="epoch",
        eval_steps=eval_steps,
        fp16=fp16,
        bf16=bf16,
        deepspeed=ds_config_dict,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_steps=max_steps,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
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


@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune", default=MODEL_PATH)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", default=LOCAL_OUTPUT_DIR)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=1, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=1, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=1, help="Batch size to use for evaluation.")
@click.option("--warmup-steps", type=int, default=20, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=100, help="How often to checkpoint the model")
@click.option("--max-steps", type=int, default=200, help="Maximum steps to run")
@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option("--gradient-accumulation-steps", type=int, default=8, help="Number of steps to accumulate gradients until stepping the optimizer")
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A10's and A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
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

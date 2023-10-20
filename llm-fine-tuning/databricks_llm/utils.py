import functools
import json
import pathlib
import os
from dataclasses import field, dataclass
import logging
from typing import Optional, Union, Tuple, Dict, Any
import shutil

import yaml
from huggingface_hub import login
from transformers import (
    IntervalStrategy,
    SchedulerType,
)

logger = logging.getLogger(__name__)
LOCAL_DISK_HF = "/local_disk0/hf"


@dataclass
class ExtendedTrainingArguments:
    local_rank: Optional[str] = field(default="-1")
    token: Optional[str] = field(default=None)
    number_of_tasks: Optional[int] = field(default=2)
    dataset: Optional[str] = field(default=None)
    model: Optional[str] = field(default=None)
    tokenizer: Optional[str] = field(default=None)

    use_lora: Optional[bool] = field(default=True)
    use_4bit: Optional[bool] = field(default=False)

    final_model_output_path: Optional[str] = field(default="/local_disk0/final_model")

    deepspeed_config: Union[Optional[str], Optional[Dict[str, Any]]] = field(
        default=None
    )
    fsdp_config: Union[Optional[str], Optional[Dict[str, Any]]] = field(default=None)

    output_dir: Optional[str] = field(default=None)
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


def _mount(it):
    for _ in it:
        import pandas as pd
        import subprocess
        import pathlib

        ls_cmd = "touch /dev/shm/test && ls /dev/shm"
        res = ""

        if not (pathlib.Path("/dev/shm").exists()) or not (
            pathlib.Path("/dev/shm").is_dir()
        ):
            cmd = (
                "mkdir /local_disk0/devshm && mount --bind /dev/shm /local_disk0/devshm && "
                + ls_cmd
            )

            s = subprocess.run(cmd)
            res = s.stdout
        yield pd.DataFrame(data={"res": [res]})


def get_spark():
    try:
        import IPython

        return IPython.get_ipython().user_ns["spark"]
    except:
        raise Exception(
            "Spark is not available! You are probably running this code outside of Databricks environment."
        )


def get_display():
    try:
        import IPython

        return IPython.get_ipython().user_ns["display"]
    except:
        raise Exception(
            "Spark is not available! You are probably running this code outside of Databricks environment."
        )


def _num_executors():
    return get_spark().sparkContext._jsc.sc().getExecutorMemoryStatus().keySet().size()


def _prepare_df_for_all_executors():
    num_executors = _num_executors()
    input_df = get_spark().range(
        start=0, end=num_executors, step=1, numPartitions=num_executors
    )
    return input_df


def check_mount_dev_shm():
    input_df = _prepare_df_for_all_executors()
    p_df = input_df.mapInPandas(
        func=_mount, schema="res string", barrier=True
    ).toPandas()
    get_display()(p_df)


def copy_source_code(dest: str):
    src = (pathlib.Path.cwd() / "..").resolve()
    dest = (pathlib.Path(dest)).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(str(dest), ignore_errors=True)
    shutil.copytree(
        str(src),
        str(dest),
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("notebooks", "old_notebooks"),
    )


def write_huggingface_token(it, token_file: str = ""):
    import pandas as pd

    for _ in it:
        with open("/root/.cache/huggingface/token", "w") as f:
            f.write(token_file)
        yield pd.DataFrame(data={"res": ["OK"]})


def remote_login(args: ExtendedTrainingArguments):
    token = get_huggingface_token(args)
    if token:
        input_df = _prepare_df_for_all_executors()
        _f = functools.partial(write_huggingface_token, token_file=token)
        p_df = input_df.mapInPandas(
            func=_f, schema="res string", barrier=True
        ).toPandas()
        get_display()(p_df)


def huggingface_login(args: ExtendedTrainingArguments):
    token = get_huggingface_token(args)
    if token:
        login(token)


def get_huggingface_token(args: ExtendedTrainingArguments) -> str | None:
    if args.token is not None and len(args.token):
        return args.token
    elif pathlib.Path("/root/.cache/huggingface/token").exists():
        return pathlib.Path("/root/.cache/huggingface/token").read_text()
    elif pathlib.Path(f"{LOCAL_DISK_HF}/huggingface/token").exists():
        pathlib.Path(f"{LOCAL_DISK_HF}/huggingface/token").read_text()
    else:
        return None


def set_up_huggingface_cache():
    pathlib.Path(LOCAL_DISK_HF).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = LOCAL_DISK_HF
    os.environ["HF_DATASETS_CACHE"] = LOCAL_DISK_HF
    os.environ["TRANSFORMERS_CACHE"] = LOCAL_DISK_HF
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"


def resolve_deepspeed_config(path: str) -> Dict[str, Any]:
    config_path = str((pathlib.Path.cwd() / ".." / "ds_configs" / path).resolve())
    with open(config_path) as file:
        return json.load(file)


def resolve_fsdp_config(path: str) -> Dict[str, Any]:
    config_path = str((pathlib.Path.cwd() / ".." / "fsdp_configs" / path).resolve())
    with open(config_path) as file:
        return yaml.safe_load(file)

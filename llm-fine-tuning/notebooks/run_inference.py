# Databricks notebook source
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from huggingface_hub import notebook_login, login

# notebook_login()


# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

import logging

import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

from databricks_llm.inference import generate_text, generate_text_for_df
from databricks_llm.model_utils import get_model_and_tokenizer
from databricks_llm.notebook_utils import get_dbutils

# COMMAND ----------

DEFAULT_INPUT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
]

# COMMAND ----------

get_dbutils().widgets.combobox(
    "pretrained_name_or_path",
    DEFAULT_INPUT_MODEL,
    SUPPORTED_INPUT_MODELS,
    "pretrained_name_or_path",
)

# COMMAND ----------

pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
print(pretrained_name_or_path)

# COMMAND ----------

questions = [
    " The coffee shop 'The Wrestlers' is located on the riverside, near 'Raja Indian Cuisine'. They serve English food and a price range of less than £20, and are not family-friendly. ",
    " Cotto is an inexpensive English restaurant near The Portland Arms in the city centre, and provides English coffee shop food. Customers recently rated the store 5 out of 5. ",
    " The Eagle coffee shops Chinese food, moderately priced, customer rating 1 out of 5, located city centre, kid friendly, located near Burger King. ",
    " The Punter is a child friendly establishment located by the riverside with a customer rating of 1 out of 5. ",
    " Taste of Cambridge, a coffee shop specializing in English eatery, is located in riverside near Crowne Plaza Hotel and is known to be very kid friendly. ",
    " The Punter is an expensive Chinese coffee shop located near Café Sicilia. ",
    " Clowns is a coffee shop that severs English food. Clowns is located in Riverside near Clare Hall. Clowns customer service ratings are low. ",
]

# COMMAND ----------

model, tokenizer = get_model_and_tokenizer(
    pretrained_name_or_path,
    pretrained_name_or_path_tokenizer="meta-llama/Llama-2-13b-chat-hf",
    inference=True,
)

# COMMAND ----------

# MAGIC %md # Generation using fine-tuned Llama v2  & Llama v2 Prompt Structure

# COMMAND ----------


def get_prompt_llama(query: str) -> str:
    return f"""<s>[INST] <<SYS>>Extract entities from the text below.<</SYS>> {query} [/INST] """


def post_process(s: str) -> str:
    _idx = s.find("[/INST]")
    if _idx > 0:
        s = s[_idx + len("[/INST]") :].strip()
    return s.replace("[inst}", "")


# COMMAND ----------

q_df = pd.DataFrame(data={"txt": questions}, columns=["txt"])

res_df = generate_text_for_df(
    model,
    tokenizer,
    q_df,
    "txt",
    "gen_txt",
    batch_size=20,
    gen_prompt_fn=get_prompt_llama,
    post_process_fn=post_process,
    max_new_tokens=64,
    temperature=0,
)
display(res_df)

# COMMAND ----------

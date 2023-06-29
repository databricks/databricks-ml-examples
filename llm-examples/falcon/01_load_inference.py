# Databricks notebook source
# MAGIC %md
# MAGIC # Falcon-7b-instruct Inference on Databricks
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input (batch inference requires slightly more memory). On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.

# COMMAND ----------

# MAGIC %pip install -q -U torch==2.0.1
# MAGIC %pip install -q einops==0.6.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The below snippets are adapted from [the model card of falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct). The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# Load model to text generation pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision="9f16e66a0235c4ba24e321e3be86dd347a7911a0", # it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of falcon-7b-instruct in https://huggingface.co/tiiuae/falcon-7b-instruct/commits/main
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# Define prompt template, the format below is from: http://fastml.com/how-to-train-your-own-chatgpt-alpaca-style-part-one/

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

# COMMAND ----------

# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on a single input

# COMMAND ----------

results = gen_text(["What is a large language model?"])
print(results[0])

# COMMAND ----------

# Use args such as temperature and max_new_tokens to control text generation
results = gen_text(["What is a large language model?"], temperature=0.5, max_new_tokens=100, use_template=True)
print(results[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference

# COMMAND ----------

# From databricks-dolly-15k
inputs = [
  "Think of some family rules to promote a healthy family relationship",
  "In the series A Song of Ice and Fire, who is the founder of House Karstark?",
  "which weighs more, cold or hot water?",
  "Write a short paragraph about why you should not have both a pet cat and a pet bird.",
  "Is beauty objective or subjective?",
  "What is SVM?",
  "What is the current capital of Japan?",
  "Name 10 colors",
  "How should I invest my money?",
  "What are some ways to improve the value of your home?",
  "What does fasting mean?",
  "What is cloud computing in simple terms?",
  "What is the meaning of life?",
  "What is Linux?",
  "Why do people like gardening?",
  "What makes for a good photograph?"
]

# COMMAND ----------

# Set batch size
results = gen_text(inputs, use_template=True, batch_size=8)

for output in results:
  print(output)
  print('\n')

# COMMAND ----------

# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving Llama-2-7b-chat-hf with a cluster driver proxy app
# MAGIC
# MAGIC This notebook enables you to run [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) on a Databricks cluster and expose the model to LangChain via [driver proxy](https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input. On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.
# MAGIC
# MAGIC requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main
model = "meta-llama/Llama-2-7b-chat-hf"
revision = "0ede8dd71e923db6258295621d817ca8714516d4"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision=revision,
    return_full_text=False, # don't return the prompt, only return the generated response
)

# COMMAND ----------

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>


{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# Define parameters to generate text
def gen_text_for_serving(prompt, **kwargs):
    prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    return pipeline(prompt, **kwargs)[0]['generated_text']

# COMMAND ----------

print(gen_text_for_serving("How to master Python in 3 days?"))

# COMMAND ----------

# See full list of configurable args: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
print(gen_text_for_serving("How to master Python in 3 days?", temperature=0.1, max_new_tokens=100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("llama-2-7b-chat")

@app.route('/', methods=['POST'])
def serve_falcon_7b_instruct():
  resp = gen_text_for_serving(**request.json)
  return jsonify(resp)

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Keep `app.run` running, and it could be used with Langchain ([documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app)), or by call the serving endpoint with:
# MAGIC ```python
# MAGIC import requests
# MAGIC import json
# MAGIC
# MAGIC def request_llamav2_7b(prompt, temperature=1.0, max_new_tokens=1024):
# MAGIC   token = ... # TODO: fill in with your Databricks personal access token that can access the cluster that runs this driver proxy notebook
# MAGIC   url = ...   # TODO: fill in with the driver_proxy_api output above
# MAGIC   
# MAGIC   headers = {
# MAGIC       "Content-Type": "application/json",
# MAGIC       "Authentication": f"Bearer {token}"
# MAGIC   }
# MAGIC   data = {
# MAGIC     "prompt": prompt,
# MAGIC     "temperature": temperature,
# MAGIC     "max_new_tokens": max_new_tokens,
# MAGIC   }
# MAGIC
# MAGIC   response = requests.post(url, headers=headers, data=json.dumps(data))
# MAGIC   return response.text
# MAGIC
# MAGIC
# MAGIC request_llamav2_7b("What is databricks?")
# MAGIC ```
# MAGIC Or you could try using ai_query([doucmentation](https://docs.databricks.com/sql/language-manual/functions/ai_query.html)) to call this driver proxy from Databricks SQL with:
# MAGIC ```
# MAGIC SELECT ai_query('cluster_ud:port', -- TODO: fill in the cluster_id and port number from output above.  
# MAGIC   named_struct('prompt', 'What is databricks?', 'temperature', CAST(0.1 AS DOUble)),
# MAGIC   'returnType', 'STRING')
# MAGIC ```
# MAGIC Note: The [AI Functions](https://docs.databricks.com/large-language-models/ai-functions.html) is in the public preview, to enable the feature for your workspace, please submit this [form](https://docs.google.com/forms/d/e/1FAIpQLScVyh5eRioqGwuUVxj9JOiKBAo0-FWi7L3f4QWsKeyldqEw8w/viewform).

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving falcon-40b-instruct with a cluster driver proxy app
# MAGIC
# MAGIC This notebook enables you to run [falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) on a Databricks cluster and expose the model to LangChain via [driver proxy](https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app).
# MAGIC
# MAGIC
# MAGIC [vllm](https://github.com/vllm-project/vllm/tree/main) is an open-source library that makes LLM inference fast with various optimizations.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Instance:
# MAGIC     - `g5.12xlarge` on aws
# MAGIC     - `Standard_NC24ads_A100_v4` on azure
# MAGIC     - `g2-standard-48` on gcp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# MAGIC %pip install -U  torch==2.1.0  torchvision==0.16.0  torchvision==0.15.2  transformers==4.35.0  accelerate==0.24.1  einops==0.7.0  sentencepiece==0.1.99 
# MAGIC %pip install -U vllm==0.2.0

# MAGIC dbutils.library.restartPython()

# COMMAND ----------


# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of `falcon-40b-instruct`. in https://huggingface.co/tiiuae/falcon-40b-instruct/commits/main
model = "tiiuae/falcon-40b-instruct"
revision = "ecb78d97ac356d098e79f0db222c9ce7c5d9ee5f"

# COMMAND ----------

from vllm import LLM
llm = LLM(model=model, revision=revision)

# COMMAND ----------

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
### Instruction:
{system_prompt}
{instruction}

### Response:\n
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# COMMAND ----------

# Define parameters to generate text
from vllm import SamplingParams

def gen_text_for_serving(prompt, **kwargs):
    prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512

    sampling_params = SamplingParams(**kwargs)

    outputs = llm.generate(prompt, sampling_params=sampling_params)
    texts = [out.outputs[0].text for out in outputs]

    return texts[0]


# COMMAND ----------

print(gen_text_for_serving("How to master Python in 3 days?"))

# COMMAND ----------

# See full list of configurable args: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
print(gen_text_for_serving("How to master Python in 3 days?", temperature=0.1, max_tokens=100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("falcon-40b-instruct")

@app.route('/', methods=['POST'])
def serve_falcon-40b-instruct():
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
# MAGIC def request_falcon-40b-instruct(prompt, temperature=1.0, max_new_tokens=1024):
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
# MAGIC request_falcon-40b-instruct("What is databricks?")
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
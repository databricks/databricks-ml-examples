# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving Mistral-7B-Instruct with a cluster driver proxy app
# MAGIC
# MAGIC The [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) Large Language Model (LLM) is a instruct fine-tuned version of the [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure

# COMMAND ----------

# MAGIC %pip install -U vllm transformers==4.34.0 accelerate==0.20.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

from vllm import LLM

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Mistral-7B-Instruct-v0. in https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/commits/main
model = "mistralai/Mistral-7B-Instruct-v0.1"
revision = "3dc28cf29d2edd31a0a7b8f0b21637059815b4d5"

llm = LLM(model=model, revision=revision)

# COMMAND ----------

from vllm import SamplingParams

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

app = Flask("mistral-7b-instruct")

@app.route('/', methods=['POST'])
def serve_mistral_7b_instruct():
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
# MAGIC def request_mistral_7b(prompt, temperature=1.0, max_new_tokens=1024):
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
# MAGIC request_mistral_7b("What is databricks?")
# MAGIC ```
# MAGIC Or you could try using ai_query([doucmentation](https://docs.databricks.com/sql/language-manual/functions/ai_query.html)) to call this driver proxy from Databricks SQL with:
# MAGIC ```
# MAGIC SELECT ai_query('cluster_id:port', -- TODO: fill in the cluster_id and port number from output above.  
# MAGIC   named_struct('prompt', 'What is databricks?', 'temperature', CAST(0.1 AS Double)),
# MAGIC   'returnType', 'STRING')
# MAGIC ```
# MAGIC Note: The [AI Functions](https://docs.databricks.com/large-language-models/ai-functions.html) is in the public preview, to enable the feature for your workspace, please submit this [form](https://docs.google.com/forms/d/e/1FAIpQLScVyh5eRioqGwuUVxj9JOiKBAo0-FWi7L3f4QWsKeyldqEw8w/viewform).

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------



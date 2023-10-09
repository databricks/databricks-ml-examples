# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving istral-7B-Instruct as chat completion via vllm with a cluster driver proxy app
# MAGIC
# MAGIC The [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) Large Language Model (LLM) is a instruct fine-tuned version of the [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC [vllm](https://github.com/vllm-project/vllm/tree/main) is an open-source library that makes LLM inference fast with various optimizations.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure

# COMMAND ----------

# MAGIC %pip install -U vllm==0.2.0 fschat==0.2.30 transformers==4.34.0 accelerate==0.20.3
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

from transformers import StoppingCriteria, StoppingCriteriaList
from vllm import SamplingParams

import fastchat
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \n\n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

def build_prompt(prompts) -> str:
    conv = get_conversation_template(model)
    conv = Conversation(
      name=conv.name,
      system_template=conv.system_template,
      system_message=conv.system_message,
      roles=conv.roles,
      messages=list(conv.messages),  # prevent in-place modification
      offset=conv.offset,
      sep_style=SeparatorStyle(conv.sep_style),
      sep=conv.sep,
      sep2=conv.sep2,
      stop_str=conv.stop_str,
      stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(prompts, str):
        prompt = prompts
    else:
        for message in prompts:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt


# Define parameters to generate text
def gen_text_for_serving(prompt, **kwargs):
    prompt = build_prompt(prompt)

    # Get the input params for the standard parameters for chat routes: https://mlflow.org/docs/latest/gateway/index.html#chat
    kwargs.setdefault("max_tokens", 512)
    kwargs.setdefault("temperature", 0.1)

    sampling_params = SamplingParams(**kwargs)
    outputs = llm.generate(prompt, sampling_params=sampling_params)

    output_response = []
    for request_output in outputs:
        response_messages = [{
                    "message": {
                        "role": "assistant",
                        "content": completion_output.text,
                    },
                    "metadata": {"finish_reason": completion_output.finish_reason},
                } for completion_output in request_output.outputs]
        input_length = len(request_output.prompt_token_ids)
        output_length = sum([len(completion_output.token_ids) for completion_output in request_output.outputs])
        metadata = {
            "input_tokens": input_length,
            "output_tokens": output_length,
            "total_tokens": input_length+input_length,
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "route_type": "llm/v1/chat",
        }
        output_response.append({"candidates": response_messages, "metadata":metadata})

    return output_response

# COMMAND ----------

# See all standard parameters from https://mlflow.org/docs/latest/gateway/index.html#chat
print(
    gen_text_for_serving(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is ML?"},
        ],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("mistral-7b-chat-completion")

@app.route('/', methods=['POST'])
def serve_mistral_7b_chat_completion():
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
# MAGIC
# MAGIC Or you could try using ai_query([doucmentation](https://docs.databricks.com/sql/language-manual/functions/ai_query.html)) to call this driver proxy from Databricks SQL with:
# MAGIC
# MAGIC Note: The [AI Functions](https://docs.databricks.com/large-language-models/ai-functions.html) is in the public preview, to enable the feature for your workspace, please submit this [form](https://docs.google.com/forms/d/e/1FAIpQLScVyh5eRioqGwuUVxj9JOiKBAo0-FWi7L3f4QWsKeyldqEw8w/viewform).

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------



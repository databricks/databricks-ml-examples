# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving Llama-2-7b-chat-hf as chat completion with a cluster driver proxy app
# MAGIC
# MAGIC This notebook enables you to run [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) on a Databricks cluster and expose the model to LangChain via [driver proxy](https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC Requirements:
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
revision = "08751db2aca9bf2f7f80d2e516117a53d7450235"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
)
model.eval()

# COMMAND ----------

from transformers import StoppingCriteria, StoppingCriteriaList


class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop[2:] == input_ids[0][-(len(stop) - 2) :])).item():
                return True

        return False


# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \n\n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def build_prompt(messages):
    """
    Assemble the message list to one prompt.
    message: A list of messages in a conversation from which to a new message (chat completion). Each chat message is a string dictionary containing `role` and `content`.
    """

    if messages[0]["role"] == "system":
        prompt = "<s>" + B_SYS + messages[0]["content"] + E_SYS
        messages = messages[1:]
    else:
        prompt = "<s>" + B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

    assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
        [msg["role"] == "assistant" for msg in messages[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    assert (
        messages[-1]["role"] == "user"
    ), f"Last message must be from user, got {messages[-1]['role']}"
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "user":
            prompt += B_INST + content + E_INST
        else:
            prompt += content
    return prompt


# Define parameters to generate text
def gen_text_for_serving(messages, **kwargs):
    prompt = build_prompt(messages)

    # Get the input params for the standard parameters for chat routes: https://mlflow.org/docs/latest/gateway/index.html#chat
    kwargs.setdefault("max_tokens", 512)
    kwargs.setdefault("candidate_count", 1)
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("stop", [])

    encoded_input = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    generation_config = transformers.GenerationConfig(
        max_new_tokens=kwargs["max_tokens"],
        do_sample=True,
        temperature=kwargs["temperature"],
        num_return_sequences=kwargs["candidate_count"],
    )

    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
        for stop_word in kwargs["stop"]
    ]
    stopping_criteria = StoppingCriteriaList(
        [ChatStoppingCriteria(stops=stop_words_ids)]
    )

    output = model.generate(
        encoded_input,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    )

    # Decode the prediction to text
    response_messages = []
    prompt_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
    for i in range(len(output)):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)

        # Removing the prompt from the generated text
        generated_response = tokenizer.decode(
            output[i][prompt_length:], skip_special_tokens=True
        )

        response_messages.append({
          "message": {
                "role": "assistant",
                "content": generated_response,
            }
        })

    return {"candidates": response_messages}

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

app = Flask("llama-2-7b-chat-completion")

@app.route('/', methods=['POST'])
def serve_llama2_7b_chat_completion():
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



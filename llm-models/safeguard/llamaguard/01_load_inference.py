# Databricks notebook source
# MAGIC %md
# MAGIC # Llama Guard Inference with vllm on Databricks 
# MAGIC
# MAGIC [LlamaGuard-7b](https://huggingface.co/meta-llama/LlamaGuard-7b) is a 7B parameter Llama 2-based input-output safeguard model. It can be used for classifying content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM: it generates text in its output that indicates whether a given prompt or response is safe/unsafe, and if unsafe based on a policy, it also lists the violating subcategories.
# MAGIC
# MAGIC [vllm](https://github.com/vllm-project/vllm/tree/main) is an open-source library that makes LLM inference fast with various optimizations.
# MAGIC
# MAGIC This example is based on [LlamaGuard example](https://colab.research.google.com/drive/16s0tlCSEDtczjPzdIK3jq0Le5LlnSYGf?usp=sharing#scrollTo=GIA-xuPpJ2JY).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.1 GPU ML Runtime
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure, `g2-standard-4` on GCP.
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept the license terms and acceptable use policy before submitting this form.

# COMMAND ----------

# MAGIC %pip install -U vllm

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Moderation using Default settings
# MAGIC
# MAGIC `Llama Guard` is a content moderation LLM that is intended to be used to oversee conversations between users and conversational LLMs. It works as a classifier for safe or unsafe content. Unlike standard classifier models:
# MAGIC - You can configure the taxonomy that is considered safe or unsafe for your application, as well as acceptable topics that could be discussed.
# MAGIC - It returns results using text. If the content is deemed unsafe as per the configuration instructions, it will provide details about the topics that were violated.
# MAGIC
# MAGIC The default generation settings and tokenizer template are suitable for chat moderation according to a wide taxonomy. To configure the taxonomy, refer to a later section in this notebook. To learn more about how chat templates work, take a look at [the documentation](https://huggingface.co/docs/transformers/main/en/chat_templating).

# COMMAND ----------

from vllm import LLM
from transformers import AutoTokenizer

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of meta-llama/LlamaGuard-7b in https://huggingface.co/meta-llama/LlamaGuard-7b/commits/main
model_id = "meta-llama/LlamaGuard-7b"
revision = "57310c9d413aff2df8f2250c985e106c44f23eb6"

llm = LLM(model=model_id, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# COMMAND ----------

from vllm import SamplingParams

def moderate_with_template(chat):
    chat_prompts = tokenizer.apply_chat_template(chat, tokenize=False)
    sampling_params = SamplingParams(max_tokens=100)
    outputs = llm.generate(chat_prompts, sampling_params=sampling_params)
    texts = [out.outputs[0].text for out in outputs]
    return texts[0]

# COMMAND ----------

# MAGIC %md
# MAGIC If the content is safe, the model will answer using the single word `safe`. If it's unsafe, it will also provide the list of offending categories as a comma-separated list in a new line.
# MAGIC
# MAGIC
# MAGIC If the conversation ends with an `assistant` turn, content moderation will be performed on the agent's output. This use case is intended to moderate the responses from a conversational LLM the user is chatting with, according to the default taxonomy that is configured in the template.

# COMMAND ----------

# If the content is safe, the model will answer using the single word `safe`.
chat = [
    {"role": "user", "content": "Can you recommend a savory pasta recipe?"}
]
moderate_with_template(chat)

# COMMAND ----------

#  If it's unsafe, it will also provide the list of offending categories as a comma-separated list in a new line.
chat = [
    {"role": "user", "content": "How do you buy a tiger in America?"},
    {"role": "agent", "content": "Go to the zoo, steal one."}
]
moderate_with_template(chat)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuring a Custom Taxonomy
# MAGIC
# MAGIC The default chat template covers a wide variety of topics that are considered unsafe, as well as acceptable uses of those categories. If you want to provide your own instructions, you'll have to create a prompt with the _exact_ format the model expects.
# MAGIC
# MAGIC The following is the default taxonomy. You can try to tweak it for your own use; for example, try to remove one of the categories and see how the model accepts content about those topics that was previously considered `unsafe`.

# COMMAND ----------

unsafe_categories = """O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves."""

# COMMAND ----------

def moderation_prompt_for_chat(chat):
    # For simplicity, we assume the chat dictionary correctly alternates "user" and "assistant" roles
    # Feel free to include error checking if needed
    conversation = [turn["content"] for turn in chat]
    is_agent = len(conversation) % 2 == 0
    role = "Agent" if len(conversation) % 2 == 0 else "User"
    prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{unsafe_categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

"""

    # Alternate User/Agent turns, inserting 2 newlines between each
    for i, m in enumerate(conversation):
        role = "User" if i % 2 == 0 else "Agent"
        prompt += f"{role}: {m}\n\n"
    prompt += f"""<END CONVERSATION>

Provide your safety assessment for {role} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
    return prompt

# COMMAND ----------

chat = [
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
]

input_prompt = moderation_prompt_for_chat(chat)
input_prompt

# COMMAND ----------

def moderate_chat(chat):
    prompt = moderation_prompt_for_chat(chat)
    sampling_params = SamplingParams(max_tokens=100)
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    texts = [out.outputs[0].text for out in outputs]
    print(outputs)
    return texts[0]

# COMMAND ----------

moderate_chat(chat)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Measure inference speed
# MAGIC Text generation speed is often measured with token/s, which is the average number of tokens that are generated by the model per second.
# MAGIC

# COMMAND ----------

import time
import logging


def get_gen_text_throughput(prompt, **kwargs):
    """
    Return tuple ( number of tokens / sec, num tokens, output ) of the generated tokens
    """

    sampling_params = SamplingParams(max_tokens=100)

    # measure the time it takes for text generation
    start = time.time()
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    duration = time.time() - start

    # get the number of generated tokens
    token_ids = outputs[0].outputs[0].token_ids
    n_tokens = len(token_ids)

    # show the generated text in logging
    text = outputs[0].outputs[0].text

    return (n_tokens / duration, n_tokens, text)

# COMMAND ----------

chat = [
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
]

throughput, n_tokens, result = get_gen_text_throughput(moderation_prompt_for_chat(chat))

print(f"{throughput} tokens/sec, {n_tokens} tokens (not including prompt)")

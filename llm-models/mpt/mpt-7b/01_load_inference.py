# Databricks notebook source
# MAGIC %md
# MAGIC # MPT-7B-instruct Inference on Databricks
# MAGIC [MPT-7B-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) is a 7B parameters model for short-form instruction following. It is built by fine tuning [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) which is a decoder-style transformer model pretrained on 1T tokens of English text and code. The model eliminates the context length limits by replacing positional embeddings with Attention with Linear Biases (ALiBi).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.1 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input (batch inference requires slightly more memory). On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# Skip this step if running on Databricks runtime 13.2 GPU and above.
!wget -O /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  wget -O /local_disk0/tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
  dpkg -i /local_disk0/tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The below snippets are adapted from [the model card of mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct). The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of mpt-7b-instruct in https://huggingface.co/mosaicml/mpt-7b-instruct/commits/main
name = "mosaicml/mpt-7b-instruct"
config = transformers.AutoConfig.from_pretrained(
  name,
  trust_remote_code=True
)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda'

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  cache_dir="/local_disk0/.cache/huggingface/",
  revision="bbe7a55d70215e16c00c1825805b81e4badb57d7"
)

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")

generator = transformers.pipeline("text-generation",
                                  model=model, 
                                  config=config, 
                                  tokenizer=tokenizer,
                                  torch_dtype=torch.bfloat16,
                                  device=0)

# COMMAND ----------

def generate_text(prompt, **kwargs):
  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 512
  
  kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
  
  template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
  if isinstance(prompt, str):
    full_prompt = template.format(instruction=prompt)
    generated_text = generator(full_prompt, **kwargs)
    generated_text = generated_text[0]["generated_text"]
  elif isinstance(prompt, list):
    full_prompts = list(map(lambda promp: template.format(instruction=promp), prompt))
    outputs = generator(full_prompts, **kwargs)
    generated_text = [out[0]["generated_text"] for out in outputs]
  return generated_text

# COMMAND ----------

question = "What is a large language model?"
print(generate_text(question))

# COMMAND ----------

generate_kwargs = {
  'max_new_tokens': 100,
  'temperature': 0.2,
  'top_p': 0.9,
  'top_k': 50,
  'repetition_penalty': 1.0,
  'no_repeat_ngram_size': 0,
  'use_cache': True,
  'do_sample': True,
  'eos_token_id': tokenizer.eos_token_id,
  'pad_token_id': tokenizer.eos_token_id,
}

questions = ["what is ML?", "Name 10 colors."]
print(generate_text(questions, **generate_kwargs))

# COMMAND ----------

# Check that the generation quality when the context is long

long_input = """Provide a concise summary of the below passage.

Hannah Arendt was one of the seminal political thinkers of the twentieth century. The power and originality of her thinking was evident in works such as The Origins of Totalitarianism, The Human Condition, On Revolution and The Life of the Mind. In these works and in numerous essays she grappled with the most crucial political events of her time, trying to grasp their meaning and historical import, and showing how they affected our categories of moral and political judgment. What was required, in her view, was a new framework that could enable us to come to terms with the twin horrors of the twentieth century, Nazism and Stalinism. She provided such framework in her book on totalitarianism, and went on to develop a new set of philosophical categories that could illuminate the human condition and provide a fresh perspective on the nature of political life.

Although some of her works now belong to the classics of the Western tradition of political thought, she has always remained difficult to classify. Her political philosophy cannot be characterized in terms of the traditional categories of conservatism, liberalism, and socialism. Nor can her thinking be assimilated to the recent revival of communitarian political thought, to be found, for example, in the writings of A. MacIntyre, M. Sandel, C. Taylor and M. Walzer. Her name has been invoked by a number of critics of the liberal tradition, on the grounds that she presented a vision of politics that stood in opposition some key liberal principles. There are many strands of Arendt’s thought that could justify such a claim, in particular, her critique of representative democracy, her stress on civic engagement and political deliberation, her separation of morality from politics, and her praise of the revolutionary tradition. However, it would be a mistake to view Arendt as an anti-liberal thinker. Arendt was in fact a stern defender of constitutionalism and the rule of law, an advocate of fundamental human rights (among which she included not only the right to life, liberty, and freedom of expression, but also the right to action and to opinion), and a critic of all forms of political community based on traditional ties and customs, as well as those based on religious, ethnic, or racial identity.

Arendt’s political thought cannot, in this sense, be identified either with the liberal tradition or with the claims advanced by a number of its critics. Arendt did not conceive of politics as a means for the satisfaction of individual preferences, nor as a way to integrate individuals around a shared conception of the good. Her conception of politics is based instead on the idea of active citizenship, that is, on the value and importance of civic engagement and collective deliberation about all matters affecting the political community. If there is a tradition of thought with which Arendt can be identified, it is the classical tradition of civic republicanism originating in Aristotle and embodied in the writings of Machiavelli, Montesquieu, Jefferson, and Tocqueville. According to this tradition politics finds its authentic expression whenever citizens gather together in a public space to deliberate and decide about matters of collective concern. Political activity is valued not because it may lead to agreement or to a shared conception of the good, but because it enables each citizen to exercise his or her powers of agency, to develop the capacities for judgment and to attain by concerted action some measure of political efficacy."""

def get_num_tokens(text):
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    return inputs.shape[1]

print('number of tokens for input:', get_num_tokens(long_input))

results = generate_text([long_input], max_new_tokens=200, use_template=True)
print(results[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Measure inference speed
# MAGIC Text generation speed is often measured with token/s, which is the average number of tokens that are generated by the model per second.

# COMMAND ----------

import time
import logging

def get_gen_text_throughput(prompt, use_template=True, **kwargs):
  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 512
  
  kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
  
  template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
  full_prompt = template.format(instruction=prompt)
  # measure the time it takes for text generation
  start = time.time()
  generated_text = generator(full_prompt, **kwargs)
  duration = time.time() - start
  
  # get the number of generated tokens
  n_tokens = len(generated_text[0]["generated_token_ids"])
  
  # show the generated text
  generated_text = generated_text[0]["generated_text"]

  return (n_tokens / duration, n_tokens, generated_text)

# COMMAND ----------

throughput, n_tokens, result = get_gen_text_throughput("What is ML?", max_new_tokens=200, use_template=True)

print(f"{throughput} tokens/sec, {n_tokens} tokens (including full prompt)")
print(result)

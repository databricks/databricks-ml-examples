# Databricks notebook source
# MAGIC %md
# MAGIC # `dbrx-instruct` Inference on Databricks
# MAGIC
# MAGIC The [dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct) Large Language Model (LLM) is a instruct fine-tuned version of the [dbrx-base](https://huggingface.co/databricks/dbrx-base) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC [vllm](https://github.com/vllm-project/vllm/tree/main) is an open-source library that makes LLM inference fast with various optimizations.
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.3 GPU ML Runtime
# MAGIC - Instance:
# MAGIC     - `p4d.24xlarge` on aws
# MAGIC     - `Standard_NC24ads_A100_v4` on azure
# MAGIC     - `a2-ultragpu-4g` on gcp


# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# MAGIC %pip install -U  torch==2.2.2  torchvision==0.17.2  transformers==4.38.2  accelerate==0.26.1  einops==0.7.0  flash-attn==2.5.2 
# MAGIC %pip install vllm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC Load and run inference on Databricks.

# COMMAND ----------
from transformers import AutoTokenizer

# Load the model

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of `dbrx-instruct`. in https://huggingface.co/databricks/dbrx-instruct/commits/main
model = "databricks/dbrx-instruct"
revision = "b1c4a3c6d864b8c86b88f2d20016afdae4aa0fc9"

from vllm import LLM
llm = LLM(model=model, revision=revision)

# COMMAND ----------

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

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
from vllm import SamplingParams
# Define the function to generate text
def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts
    # the default max length is pretty small (16), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512
    
    sampling_params = SamplingParams(**kwargs)
    outputs = llm.generate(full_prompts, sampling_params=sampling_params)
    texts = [out.outputs[0].text for out in outputs]
    
    return texts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on a single input

# COMMAND ----------

results = gen_text(["What is a large language model?"])
print(results[0])

# COMMAND ----------
# Use args such as temperature and max_tokens to control text generation
results = gen_text(["What is a large language model?"], temperature=0.5, max_tokens=100, use_template=True)
print(results[0])

# COMMAND ----------

# Check that the generation quality when the context is long
from transformers import AutoTokenizer
long_input = """Provide a concise summary of the below passage.

Hannah Arendt was one of the seminal political thinkers of the twentieth century. The power and originality of her thinking was evident in works such as The Origins of Totalitarianism, The Human Condition, On Revolution and The Life of the Mind. In these works and in numerous essays she grappled with the most crucial political events of her time, trying to grasp their meaning and historical import, and showing how they affected our categories of moral and political judgment. What was required, in her view, was a new framework that could enable us to come to terms with the twin horrors of the twentieth century, Nazism and Stalinism. She provided such framework in her book on totalitarianism, and went on to develop a new set of philosophical categories that could illuminate the human condition and provide a fresh perspective on the nature of political life.

Although some of her works now belong to the classics of the Western tradition of political thought, she has always remained difficult to classify. Her political philosophy cannot be characterized in terms of the traditional categories of conservatism, liberalism, and socialism. Nor can her thinking be assimilated to the recent revival of communitarian political thought, to be found, for example, in the writings of A. MacIntyre, M. Sandel, C. Taylor and M. Walzer. Her name has been invoked by a number of critics of the liberal tradition, on the grounds that she presented a vision of politics that stood in opposition some key liberal principles. There are many strands of Arendt’s thought that could justify such a claim, in particular, her critique of representative democracy, her stress on civic engagement and political deliberation, her separation of morality from politics, and her praise of the revolutionary tradition. However, it would be a mistake to view Arendt as an anti-liberal thinker. Arendt was in fact a stern defender of constitutionalism and the rule of law, an advocate of fundamental human rights (among which she included not only the right to life, liberty, and freedom of expression, but also the right to action and to opinion), and a critic of all forms of political community based on traditional ties and customs, as well as those based on religious, ethnic, or racial identity.

Arendt’s political thought cannot, in this sense, be identified either with the liberal tradition or with the claims advanced by a number of its critics. Arendt did not conceive of politics as a means for the satisfaction of individual preferences, nor as a way to integrate individuals around a shared conception of the good. Her conception of politics is based instead on the idea of active citizenship, that is, on the value and importance of civic engagement and collective deliberation about all matters affecting the political community. If there is a tradition of thought with which Arendt can be identified, it is the classical tradition of civic republicanism originating in Aristotle and embodied in the writings of Machiavelli, Montesquieu, Jefferson, and Tocqueville. According to this tradition politics finds its authentic expression whenever citizens gather together in a public space to deliberate and decide about matters of collective concern. Political activity is valued not because it may lead to agreement or to a shared conception of the good, but because it enables each citizen to exercise his or her powers of agency, to develop the capacities for judgment and to attain by concerted action some measure of political efficacy."""

def get_num_tokens(text):
    tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", padding_side="left")
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    return inputs.shape[1]

print('number of tokens for input:', get_num_tokens(long_input))

results = gen_text([long_input], use_template=True, max_tokens=150)
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

results = gen_text(inputs, use_template=True)

for i, output in enumerate(results):
  print(f"======Output No. {i+1}======")
  print(output)
  print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Measure inference speed
# MAGIC Text generation speed is often measured with token/s, which is the average number of tokens that are generated by the model per second.
# MAGIC

# COMMAND ----------

import time

def get_gen_text_throughput(prompt, use_template=True, **kwargs):
    """
    Return tuple ( number of tokens / sec, num tokens, output ) of the generated tokens
    """
    if use_template:
        full_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
    else:
        full_prompt = prompt
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512
    sampling_params = SamplingParams(**kwargs)
    
    num_input_tokens = get_num_tokens(full_prompt)

    # measure the time it takes for text generation
    start = time.time()
   
    outputs = llm.generate(full_prompt, sampling_params=sampling_params)
    duration = time.time() - start

    # get the number of generated tokens
    token_ids = outputs[0].outputs[0].token_ids
    n_tokens = len(token_ids)
    # show the generated text in logging
    text = outputs[0].outputs[0].text
    return (n_tokens / duration, n_tokens, text)

# COMMAND ----------

throughput, n_tokens, result = get_gen_text_throughput("What is ML?", use_template=False)

print(f"{throughput} tokens/sec, {n_tokens} tokens (not including prompt)")

# COMMAND ----------

# When the context is long or the generated text is long, it takes longer to generate each token in average
throughput, n_tokens, result = get_gen_text_throughput(long_input, max_tokens=200, use_template=True)

print(f"{throughput} tokens/sec, {n_tokens} tokens (not including prompt)")

# COMMAND ----------
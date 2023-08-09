# Databricks notebook source
# MAGIC %md
# MAGIC # Llama 2 chat Inference on Databricks
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input (batch inference requires slightly more memory). On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.
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

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision=revision,
    return_full_text=False
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

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

# Check that the generation quality when the context is long

long_input = """Provide a concise summary of the below passage.

Hannah Arendt was one of the seminal political thinkers of the twentieth century. The power and originality of her thinking was evident in works such as The Origins of Totalitarianism, The Human Condition, On Revolution and The Life of the Mind. In these works and in numerous essays she grappled with the most crucial political events of her time, trying to grasp their meaning and historical import, and showing how they affected our categories of moral and political judgment. What was required, in her view, was a new framework that could enable us to come to terms with the twin horrors of the twentieth century, Nazism and Stalinism. She provided such framework in her book on totalitarianism, and went on to develop a new set of philosophical categories that could illuminate the human condition and provide a fresh perspective on the nature of political life.

Although some of her works now belong to the classics of the Western tradition of political thought, she has always remained difficult to classify. Her political philosophy cannot be characterized in terms of the traditional categories of conservatism, liberalism, and socialism. Nor can her thinking be assimilated to the recent revival of communitarian political thought, to be found, for example, in the writings of A. MacIntyre, M. Sandel, C. Taylor and M. Walzer. Her name has been invoked by a number of critics of the liberal tradition, on the grounds that she presented a vision of politics that stood in opposition some key liberal principles. There are many strands of Arendt’s thought that could justify such a claim, in particular, her critique of representative democracy, her stress on civic engagement and political deliberation, her separation of morality from politics, and her praise of the revolutionary tradition. However, it would be a mistake to view Arendt as an anti-liberal thinker. Arendt was in fact a stern defender of constitutionalism and the rule of law, an advocate of fundamental human rights (among which she included not only the right to life, liberty, and freedom of expression, but also the right to action and to opinion), and a critic of all forms of political community based on traditional ties and customs, as well as those based on religious, ethnic, or racial identity.

Arendt’s political thought cannot, in this sense, be identified either with the liberal tradition or with the claims advanced by a number of its critics. Arendt did not conceive of politics as a means for the satisfaction of individual preferences, nor as a way to integrate individuals around a shared conception of the good. Her conception of politics is based instead on the idea of active citizenship, that is, on the value and importance of civic engagement and collective deliberation about all matters affecting the political community. If there is a tradition of thought with which Arendt can be identified, it is the classical tradition of civic republicanism originating in Aristotle and embodied in the writings of Machiavelli, Montesquieu, Jefferson, and Tocqueville. According to this tradition politics finds its authentic expression whenever citizens gather together in a public space to deliberate and decide about matters of collective concern. Political activity is valued not because it may lead to agreement or to a shared conception of the good, but because it enables each citizen to exercise his or her powers of agency, to develop the capacities for judgment and to attain by concerted action some measure of political efficacy."""

def get_num_tokens(text):
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    return inputs.shape[1]

print('number of tokens for input:', get_num_tokens(long_input))

results = gen_text([long_input], max_new_tokens=150, use_template=False)
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

# MAGIC %md
# MAGIC ## Measure inference speed
# MAGIC Text generation speed is often measured with token/s, which is the average number of tokens that are generated by the model per second.
# MAGIC

# COMMAND ----------

import time
import logging


def get_gen_text_throughput(prompt, use_template=True, **kwargs):
    """
    Return tuple ( number of tokens / sec, num tokens, output ) of the generated tokens
    """
    if use_template:
        full_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
    else:
        full_prompt = prompt

    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    kwargs.update(
        {
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_tensors": True,  # make the pipeline return token ids instead of decoded text to get the number of generated tokens
        }
    )

    num_input_tokens = get_num_tokens(full_prompt)

    # measure the time it takes for text generation
    start = time.time()
    outputs = pipeline(full_prompt, **kwargs)
    duration = time.time() - start

    # get the number of generated tokens
    n_tokens = len(outputs[0]["generated_token_ids"])

    # show the generated text in logging
    result = tokenizer.batch_decode(
        outputs[0]["generated_token_ids"][num_input_tokens:], skip_special_tokens=True
    )
    result = "".join(result)

    return (n_tokens / duration, n_tokens, result)

# COMMAND ----------

throughput, n_tokens, result = get_gen_text_throughput("What is ML?", use_template=False)

print(f"{throughput} tokens/sec, {n_tokens} tokens (including full prompt)")

# COMMAND ----------

# When the context is long or the generated text is long, it takes longer to generate each token in average
throughput, n_tokens, result = get_gen_text_throughput(long_input, max_new_tokens=200, use_template=True)

print(f"{throughput} tokens/sec, {n_tokens} tokens (including full prompt)")

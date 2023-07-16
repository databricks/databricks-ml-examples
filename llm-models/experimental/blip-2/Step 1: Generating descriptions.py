# Databricks notebook source
#Steps
#Generate clothing item list
#Generate style list for each. 
#Have a range of common clothing colors
#Generate a collected listed of prompts
#Generate Descriptions according to a prompt template such that there is no descriptions of people in the output

# COMMAND ----------

# MAGIC %pip install accelerate
# MAGIC %pip install -i https://test.pypi.org/simple/ bitsandbytes
# MAGIC %pip install einops
# MAGIC %pip install --upgrade torch
# MAGIC %pip install inflect

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate clothing item list

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)



# COMMAND ----------

sequences = pipeline(
   "Create a list of 100 clothing items. Each item should be singular and each item should be separated by commas: shirt, trousers,",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# COMMAND ----------

string = sequences[0]['generated_text'].split(':')[1]

# COMMAND ----------

items = [item.lstrip("0123456789.- ") for item in string.split(', ') if item]
items

# COMMAND ----------

items = list(set(items))
items

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate style list for each. 

# COMMAND ----------

sequences = pipeline(
   "Create a comma separated list of adjectives for clothing items:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# COMMAND ----------

styles = sequences[0]['generated_text'].split(':')[1]
items_style = [item.lstrip("0123456789-. ").strip(' ') for item in styles.split('\n') if item]
items_style = items_style[0].split(',')
items_style

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate a range of common clothing colors

# COMMAND ----------

sequences = pipeline(
   "Create a comma separated list of regular colors for clothing items:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# COMMAND ----------

colors = sequences[0]['generated_text'].split(':')[1]
colors= [item.lstrip("0123456789. ") for item in colors.split('\n') if item]
colors = colors[0].split(',')

# COMMAND ----------

colors

# COMMAND ----------

"Create concise description of {} {} {}: ".format('red', 'athletic', 'blazer')

# COMMAND ----------

# #Create description prompt template
# template = "Create detailed description of {} {} {}: ".format(color, style, item)
# sequences = pipeline(
#    template,
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

# COMMAND ----------

# color = 'red'
# style= 'casual'
# item = 'sweater'

# COMMAND ----------

def generate(prompt):
  template = prompt
  sequences = pipeline(
    template,
      max_length=75,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
  )
  return sequences[0]['generated_text'].split('\n')[1]

# COMMAND ----------

generate('Create concise description of red athletic blazer with no brand names:')

# COMMAND ----------

colors, items, items_style

# COMMAND ----------

#Getting rid of empty characters
colors_refined = [color for color in colors if len(color) >1]
items_refined = [item for item in items if len(item) >1]
styles_refined = [style for style in items_style if len(style)>1]
#It is commonly onserved the trailing item has missing characters
#colors_refined.pop(), items_refined.pop(), styles_refined.pop()
colors_refined, items_refined, styles_refined

# COMMAND ----------

#Refine the list of items such that anomalies are removed by manually inspecting
singular_items = list(set(['jeans','overalls','socks','jacket','cardigan','sweaters','blazers','jumpsuits','leggings','skirt','shirt','t-shirts','trousers','shorts','dress','tights']))
singular_items

# COMMAND ----------

#Refining colors further to remove extra periods
colors_refined = ['Black', ' white', ' red', ' blue', ' green']

# COMMAND ----------

prompts = []
for item in singular_items:
  for style in styles_refined:
    for color in colors_refined:
       prompts.append("Create a concise description of {} {} {}. Do not mention brand names:".format(color, style, item)) 
      #prompts.append(generate(color, style, item))

# COMMAND ----------

prompts

# COMMAND ----------

refined_prompts = [s.replace('  ', ' ') for s in prompts]

# COMMAND ----------

refined_prompts

# COMMAND ----------

generate('Create a concise description of yellow fashionable shirt-dress. Do not mention brand names:')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS instructdb;
# MAGIC use instructdb;

# COMMAND ----------

#Create prompts for captioning
captioning_prompts = []
for item in singular_items:
  for style in styles_refined:
    for color in colors_refined:
       captioning_prompts.append("Create a caption for this image of {} {} {} :".format(color, style, item)) 
      #prompts.append(generate(color, style, item))
refined_captioning_prompts = [s.replace('  ', ' ') for s in captioning_prompts]

# COMMAND ----------

import pandas as pd
falcon_captioning_prompts = pd.DataFrame(refined_captioning_prompts, columns=['prompt'])
spark.createDataFrame(falcon_captioning_prompts).write.saveAsTable('captioning_prompts')

# COMMAND ----------

import pandas as pd
falcon_prompts = pd.DataFrame(refined_prompts, columns=['prompt'])
spark.createDataFrame(falcon_prompts).write.saveAsTable('concise_prompts')

# COMMAND ----------

# MAGIC %sql
# MAGIC use instructdb;

# COMMAND ----------

sample_prompts = spark.sql("SELECT * FROM concise_prompts")
display(sample_prompts)

# COMMAND ----------

#You can easily create a Hugging Face dataset from a spark dataframe
from datasets import Dataset
ds = Dataset.from_spark(sample_prompts)
ds

# COMMAND ----------

from transformers.pipelines.pt_utils import KeyDataset

descriptions_refined = []
sequences = pipeline(
   KeyDataset(ds, "prompt"),
    max_length=75,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

# COMMAND ----------

for sequence in sequences:
  descriptions_refined.append(sequence[0]['generated_text'])

# COMMAND ----------

descriptions_refined

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(descriptions_refined, columns=['description'])
spark.createDataFrame(df).write.saveAsTable('fashion_description_concise')

#spark.createDataFrame(df).write.saveAsTable('fashion_description')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM fashion_description_concise

# COMMAND ----------

#*********Do not change code below here
import random
sample_prompts = random.sample(refined_prompts, 500)
sample_prompts

# COMMAND ----------

generate('Create a detailed description of Black functional Jeans. Do not mention brand names:')

# COMMAND ----------

descriptions = [generate(prompt) for prompt in refined_prompts]

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(descriptions, columns=['description'])
#spark.createDataFrame(df).write.saveAsTable('fashion_description_4bit')

spark.createDataFrame(df).write.saveAsTable('fashion_description')


# COMMAND ----------

display(df)

# COMMAND ----------

df.shape

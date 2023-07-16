# Databricks notebook source
# MAGIC %pip install diffusers

# COMMAND ----------

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", revision="fp16", torch_dtype=torch.float16).to('cuda')


# COMMAND ----------

df = spark.sql("SELECT * from instructdb.fashion_description_concise")
display(df)

# COMMAND ----------

from pyspark.sql.functions import split
from pyspark.sql.functions import monotonically_increasing_id


# Assuming your DataFrame is named 'df' and the column with strings is named 'col_with_colon'
df = df.withColumn('diffusion_prompt', split(df['description'], ':').getItem(1)).drop('description').withColumn("id", monotonically_increasing_id())

display(df)

# COMMAND ----------

#You can easily create a Hugging Face dataset from a spark dataframe
from datasets import Dataset
ds = Dataset.from_spark(df)
ds

# COMMAND ----------

diffusion_prompts= ds['diffusion_prompt']

# COMMAND ----------

fashion_dir = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_images/"

# COMMAND ----------

i=0
suffix = '.png'
for prompt in diffusion_prompts:
  image = pipe(prompt).images[0]
  save_loc = fashion_dir+str(i)+suffix
  image.save(save_loc)
  i+=1



# COMMAND ----------

prompt = "Black fitted jeans are a classic and versatile style. They provide a sleek and streamlined look, often made from a mix of cotton and elastane or other synthetic fibers. The dark color is versatile, easily teaming with a variety of different tops and accessories."

image = pipe(prompt).images[0]
display(image)

# COMMAND ----------

image.save('/tmp/test_image.png')

# COMMAND ----------

from PIL import Image
Image.open('/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_images/239.png')

# COMMAND ----------

#Create the metadata csv for the image dataset

# COMMAND ----------

display(df)

# COMMAND ----------

dfpd = df.toPandas()
dfpd['png'] = '.png'
dfpd['file_name'] = dfpd['id'].astype(str)+dfpd['png'] 
dfpd['text']  = dfpd['diffusion_prompt']
dfpd = dfpd[['file_name','text']]
display(dfpd)

# COMMAND ----------

dfpd.to_csv(fashion_dir+'metadata.csv', index=False)

# COMMAND ----------

fashion_dir+'metadata.csv'

# COMMAND ----------

# MAGIC %sh 
# MAGIC cat '/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/fashion_images/metadata.csv'

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir=fashion_dir)

# COMMAND ----------

dataset

# Databricks notebook source
# MAGIC %pip install accelerate
# MAGIC %pip install -i https://test.pypi.org/simple/ bitsandbytes

# COMMAND ----------

image0 = "https://www.instyle.com/thmb/CdfUHOY8QqJB9pQdL9UB8DvVG0s=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/springfashiontrends21-b1688fb912194ebd9f8238312af48572.jpg"
image1 = "https://media.voguebusiness.com/photos/642c3460706ee157689b66bd/2:3/w_2560%2Cc_limit/ai-fashion-week-voguebus-story.jpg"
image2 = 'https://bananarepublicfactory.gapfactory.com/webcontent/0028/438/043/cn28438043.jpg'
image3 = "https://bananarepublic.gap.com/webcontent/0053/183/233/cn53183233.jpg"
image4 = "https://cdn2.stylecraze.com/wp-content/uploads/2018/02/Fashion-For-Women-Over-50--Outfit-Ideas-And-Wardrobe-Tips.jpg"



# COMMAND ----------

# pip install accelerate bitsandbytes
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")




# COMMAND ----------

def describe_apparel(image_url, processor, model):
  img_url = image_url
  raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

  question = "Describe the apparel in this image in vivid detail:"
  inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

  out = model.generate(**inputs)
  print(processor.decode(out[0], skip_special_tokens=True)), display(raw_image)


# COMMAND ----------

describe_apparel(image4, processor, model)

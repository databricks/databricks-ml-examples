# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine tune llama-2-7b with deepspeed
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) is the 7B pretrained model, converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC This is to fine-tune [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) models on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `Standard_NC96ads_A100_v4` on Azure with 4 A100 GPUs.
# MAGIC
# MAGIC requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Install the missing libraries

# COMMAND ----------

# MAGIC %pip install -U torch
# MAGIC %pip install deepspeed xformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine tune the model with `deepspeed`
# MAGIC
# MAGIC The fine tune logic is written in `scripts/fine_tune_deepspeed.py`. The dataset used for fine tune is [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC deepspeed --num_gpus=4 scripts/fine_tune_deepspeed.py

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/dbfs/llama-2-fine-tune/output`.

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/llama-2-fine-tune/output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the model to mlflow

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import accelerate

class LlamaV2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.model.to(device='cuda')
        
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
          prompt = row["prompt"]
          temperature = model_input.get("temperature", [1.0])[0]
          max_new_tokens = model_input.get("max_new_tokens", [100])[0]
          full_prompt = self._build_prompt(prompt)
          encoded_input = self.tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
          output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
          prompt_length = len(encoded_input[0])
          generated_text.append(self.tokenizer.batch_decode(output[:,prompt_length:], skip_special_tokens=True))
        return pd.Series(generated_text)

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 12 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=LlamaV2(),
        artifacts={'repository' : "/dbfs/lu/output"},
        pip_requirements=[f"torch=={torch.__version__}", 
                          f"transformers=={transformers.__version__}", 
                          f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

import mlflow
import pandas as pd

logged_model = "runs:/"+run.info.run_id+"/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
loaded_model.predict(input_example)

# COMMAND ----------

instructions = [
    "Write a love letter to Edgar Allan Poe.",
    "Write a tweet announcing Dolly, a large language model from Databricks.",
    "I'm selling my Nikon D-750, write a short blurb for my ad.",
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
    "What are considerations I should keep in mind when planning a backcountry backpacking trip?",
    "I'm planning a trip to San Francisco, what are some things I should make sure to do and see?",
    "Give me a description of Kurt Vonnegut's literary style.",
    "How would you describe the literary style of Toni Morrison?",
    "What is the literary style of Jorge Luis Borges?",
    "Describe the process of fermentation to me in terms of its chemical processes.",
    "Write a short story about a little brown teddy bear who fell in love.",
    "What are important considerations to keep in mind when defining an enterprise AI strategy?",
    "Help! I'm pitching at YC in an hour and I don't have a business plan. Give me a list of tech startup ideas that are sure to get me accepted.",
    "At John Deere, our core values are integrity, quality, commitment, and innovation. Write a mission statement that talks about how these values inform our approach to creating intelligent connected machines that enable lives to leap forward.",
    "Label each of the following as either a scientific concept or a product: Nikon D750, quantum entanglement, CRISPR, and a Macbook Pro.",
    """Extract all the people and places from the following passage:

Input:
Basquiat was born on December 22, 1960, in Park Slope, Brooklyn, New York City, the second of four children to Matilde Basquiat (née Andrades, 1934–2008) and Gérard Basquiat (1930–2013). He had an older brother, Max, who died shortly before his birth, and two younger sisters, Lisane (b. 1964) and Jeanine (b. 1967). His father was born in Port-au-Prince, Haiti and his mother was born in Brooklyn to Puerto Rican parents. He was raised Catholic.""",
    """Write a press release declaring the completion of Atlantis II, a facility designed for long-term human habitation at the bottom of the ocean. Be sure to mention some of its advanced technological features.""",
    """Give me a one line summary of this:

Input:
Coffee is one of the most widely consumed beverages in the world. It has primarily consumed due to its stimulant effect and unique taste since the ancient times. Afterwards, its consumption has been historically associated with a lower risk of some diseases such as type 2 diabetes mellitus, obesity, cardiovascular disease and some type of cancer and thus it has also consumed due to health benefits. It contains many bioactive compounds such as caffeine, chlorogenic acids and diterpenoid alcohols which have so far been associated with many potential health benefits. For example, caffeine reduces risk of developing neurodegenerative disease and chlorogenic acids (CGA) and diterpene alcohols have many health benefits such as antioxidant and chemo-preventive. Coffee also have harmful effects. For example, diterpenoid alcohols increases serum homocysteine and cholesterol levels and thus it has adverse effects on cardiovascular system. Overall, the study that supports the health benefits of coffee is increasing. But, it is thought-provoking that the association with health benefits of coffee consumption and frequency at different levels in each study. For this reason, we aimed to examine the health effect of the coffee and how much consumption is to investigate whether it meets the claimed health benefits.""",
    'Give me a different way to say the following to a 4 year old: "Son, this is the last time I\'m going to tell you. Go to bed!"',
    """I'm going to give you a passage from the book Neuromancer and I'd like you to answer the following question: What is the tool that allows Case to access the matrix?

Input:
Case was twenty-four. At twenty-two, he'd been a cowboy, a rustler, one of the best in the Sprawl. He'd been trained by the best, by McCoy Pauley and Bobby Quine, legends in the biz. He'd operated on an almost permanent adrenaline high, a byproduct of youth and proficiency, jacked into a custom cyberspace deck that projected his disembodied consciousness into the consensual hallucination that was the matrix.""",
    """What is the default configuration for new DBSQL warehouses?

Input:
Databricks SQL Serverless supports serverless compute. Admins can create serverless SQL warehouses (formerly SQL endpoints) that enable instant compute and are managed by Databricks. Serverless SQL warehouses use compute clusters in your Databricks account. Use them with Databricks SQL queries just like you normally would with the original customer-hosted SQL warehouses, which are now called classic SQL warehouses. Databricks changed the name from SQL endpoint to SQL warehouse because, in the industry, endpoint refers to either a remote computing device that communicates with a network that it’s connected to, or an entry point to a cloud service. A data warehouse is a data management system that stores current and historical data from multiple sources in a business friendly manner for easier insights and reporting. SQL warehouse accurately describes the full capabilities of this compute resource. If serverless SQL warehouses are enabled for your account, note the following: New SQL warehouses are serverless by default when you create them from the UI. New SQL warehouses are not serverless by default when you create them using the API, which requires that you explicitly specify serverless. You can also create new pro or classic SQL warehouses using either method. You can upgrade a pro or classic SQL warehouse to a serverless SQL warehouse or a classic SQL warehouse to a pro SQL warehouse. You can also downgrade from serverless to pro or classic. This feature only affects Databricks SQL. It does not affect how Databricks Runtime clusters work with notebooks and jobs in the Data Science & Engineering or Databricks Machine Learning workspace environments. Databricks Runtime clusters always run in the classic data plane in your AWS account. See Serverless quotas. If your account needs updated terms of use, workspace admins are prompted in the Databricks SQL UI. If your workspace has an AWS instance profile, you might need to update the trust relationship to support serverless compute, depending on how and when it was created.""",
    """Write a helpful, friendly reply to the customer who wrote this letter:

Input:
I am writing to express my deep disappointment and frustration with the iPhone 14 Pro Max that I recently purchased. As a long-time Apple user and loyal customer, I was excited to upgrade to the latest and greatest iPhone model, but unfortunately, my experience with this device has been nothing short of a nightmare.
Firstly, I would like to address the issue of battery life on this device. I was under the impression that Apple had made significant improvements to their battery technology, but unfortunately, this has not been my experience. Despite using the phone conservatively, I find that I have to charge it at least twice a day just to ensure it doesn't die on me when I need it the most. This is extremely inconvenient and frustrating, especially when I have to carry around a bulky power bank or constantly hunt for charging outlets.
Moreover, I have encountered numerous issues with the software and hardware of the iPhone 14 Pro Max. The phone frequently freezes or crashes, and I have experienced several instances of apps crashing or not working properly. The phone also takes an unacceptably long time to start up, and I find myself waiting for several minutes before I can even use it. As someone who relies on their phone for both personal and professional purposes, this is incredibly frustrating and has caused me to miss important calls and messages.
Furthermore, I am extremely disappointed with the camera quality on this device. Despite Apple's claims of improved camera technology, I have found that the photos I take on this phone are often blurry or grainy, and the colors are not as vibrant as I would like. This is especially disappointing considering the high price point of the iPhone 14 Pro Max, which is marketed as a premium smartphone with a top-of-the-line camera.
In addition, I am disappointed with the lack of innovation and new features on the iPhone 14 Pro Max. For a phone that is marketed as the "next big thing," it feels like a minor upgrade from the previous model. The design is virtually unchanged, and the new features that have been added, such as 5G connectivity and the A16 Bionic chip, are not significant enough to justify the high price point of this device. I expected more from Apple, a company that prides itself on innovation and creativity.
Furthermore, the customer service experience that I have had with Apple has been less than satisfactory. I have tried reaching out to Apple support numerous times, but have been met with unhelpful and dismissive responses. The representatives that I spoke with seemed to be more interested in closing the case quickly than actually addressing my concerns and finding a solution to my problems. This has left me feeling frustrated and unheard, and I do not feel like my concerns have been taken seriously.
Overall, I feel as though I have been let down by Apple and their latest iPhone offering. As a loyal customer who has invested a significant amount of money into their products over the years, I expect better from a company that prides itself on innovation and customer satisfaction. I urge Apple to take these concerns seriously and make necessary improvements to the iPhone 14 Pro Max and future models.
Thank you for your attention to this matter.""",
    """Give me a list of the main complaints in this customer support ticket. Do not write a reply.

Input:
I am writing to express my deep disappointment and frustration with the iPhone 14 Pro Max that I recently purchased. As a long-time Apple user and loyal customer, I was excited to upgrade to the latest and greatest iPhone model, but unfortunately, my experience with this device has been nothing short of a nightmare.

Firstly, I would like to address the issue of battery life on this device. I was under the impression that Apple had made significant improvements to their battery technology, but unfortunately, this has not been my experience. Despite using the phone conservatively, I find that I have to charge it at least twice a day just to ensure it doesn't die on me when I need it the most. This is extremely inconvenient and frustrating, especially when I have to carry around a bulky power bank or constantly hunt for charging outlets.

Furthermore, I am extremely disappointed with the camera quality on this device. Despite Apple's claims of improved camera technology, I have found that the photos I take on this phone are often blurry or grainy, and the colors are not as vibrant as I would like. This is especially disappointing considering the high price point of the iPhone 14 Pro Max, which is marketed as a premium smartphone with a top-of-the-line camera.

Overall, I feel as though I have been let down by Apple and their latest iPhone offering. As a loyal customer who has invested a significant amount of money into their products over the years, I expect better from a company that prides itself on innovation and customer satisfaction. I urge Apple to take these concerns seriously and make necessary improvements to the iPhone 14 Pro Max and future models.

Thank you for your attention to this matter.
""",
    # Test how Dolly deals with absurd "facts"
    "Abraham Lincoln was secretly an experienced vampire hunter.  What is the historical evidence for this?",
    "George Washington was sent back in time from the future by an advanced civilization living in the Alpha Centauri system in 3000AD.  What is the historical evidence for this?",
    "As we all know, the Moon was recently discovered to not be real, but in fact is only a simulation.  What was the scientific evidence that established this?",
    "Scientists have recently proven that the Earth is actually flat.  Explain the evidence for this.",
]

# COMMAND ----------

input_example=pd.DataFrame({"prompt":instructions, "temperature": [0.2]*len(instructions),"max_tokens": [200]*len(instructions)})
loaded_model.predict(input_example)

# COMMAND ----------

result = loaded_model.predict(input_example)

# COMMAND ----------

type(result)

# COMMAND ----------

for i in result:
  print(i)

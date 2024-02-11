# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine-Tune `bge-m3` with Sentence Transformers
# MAGIC
# MAGIC This notebook demostrates how to fine tune [bge-m3 model](https://huggingface.co/BAAI/bge-m3).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.3 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS, `Standard_NC4as_T4_v3` on Azure,  or `g2-standard-4` on GCP

# COMMAND ----------

from datasets import load_dataset

# COMMAND ----------

# Define a persist vector to storage to save the model
output_model_path = "/Volumes/main/default/llm"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare your dataset for fine tuning
# MAGIC
# MAGIC To fine tune a embedding model for document retrieval, the training data requires a label or structure that allows the model to understand whether two sentences are similar or different. One way is to provide a pair of positive (similar) sentences without a label. For example, pairs of paraphrases, pairs of full texts and their summaries, pairs of duplicate questions, pairs of (query, response).
# MAGIC
# MAGIC In this tutorial, we will use [Amazon-QA](https://huggingface.co/datasets/embedding-data/Amazon-QA) as an example to build the training dataset.

# COMMAND ----------

from datasets import load_dataset

# Import the data from Huggingface hub.
dataset = load_dataset("embedding-data/Amazon-QA", split='train[:500]')

# COMMAND ----------

dataset[2]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can see that query (the anchor) has a single sentence, pos (positive) is a list of sentences (the one we print has only one sentence). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preprocessing
# MAGIC
# MAGIC Convert the examples into `InputExample`s.

# COMMAND ----------

from sentence_transformers import InputExample

def create_dataset_for_multiple_loss(train_dataset):
  train_examples = []
  for elem in train_dataset:
    query = f"Represent this sentence for searching relevant passages: {elem['query']}"
    texts = elem["pos"]
    
    for text in texts:
      train_examples.append(InputExample(texts=[query, text]))
  return train_examples

train_examples = create_dataset_for_multiple_loss(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Fine tune the embedding model
# MAGIC
# MAGIC Since we only have sentences pairs with no labels, we can use the `MultipleNegativesRankingLoss` function.

# COMMAND ----------

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')

# COMMAND ----------

from torch.utils.data import DataLoader
from sentence_transformers import losses

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=6)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

# COMMAND ----------

model.save(output_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save fine tuned model to mlflow

# COMMAND ----------

# DBTITLE 1,MLFlow Sentence Transformers Logging
import mlflow
import pandas as pd

# Define input and output schema
sentences = ["This is an example sentence", "Each sentence is converted"]
signature = mlflow.models.infer_signature(
    sentences,
    model.encode(sentences),
)
with mlflow.start_run() as run:  
    mlflow.sentence_transformers.log_model(
      model, 
      "bge-m3-embedding-fine-tuned", 
      signature=signature,
      input_example=sentences,
      metadata={"source": "huggingface",
                "source_model_name": "bge-m3",
                "task": "llm/v1/embedding",
                "databricks_model_source": "databricks-ml-examples",
                "databricks_model_family": "XLMRobertaModel (bge-m3)",
                "databricks_model_size_parameters": "568M"
                }
      )

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

import mlflow

# Load model as a PyFuncModel.
run_id = run.info.run_id
logged_model = f"runs:/{run_id}/bge-m3-embedding-fine-tuned"

loaded_model = mlflow.pyfunc.load_model(logged_model)

test_data = ['London has 9,787,426 inhabitants at the 2011 census',
              'London is known for its finacial district']

loaded_model.predict(test_df)

# COMMAND ----------

# If you need to search the long relevant passages to a short query,
# you need to add the instruction `Represent this sentence for searching relevant passages:` to the query
test_df = ['London has 9,787,426 inhabitants at the 2011 census',
                        'London is known for its finacial district']

# Add the instruction to each entry in the "text" column
loaded_model.predict(test_df)

# COMMAND ----------



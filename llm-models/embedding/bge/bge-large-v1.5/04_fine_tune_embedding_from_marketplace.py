# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine-Tune `bge-large-en-v1.5` with Sentence Transformers
# MAGIC
# MAGIC This notebook demostrates how to fine tune [bge-large-en model](https://huggingface.co/BAAI/bge-large-en-v1.5).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.1 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure

# COMMAND ----------

import mlflow

# Set mlflow registry to databricks-uc
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

catalog_name = "databricks_bge_v1_5_models" # Default catalog name when installing the model from Databricks Marketplace
version = 1

model_mlflow_path = f"models:/{catalog_name}.models.bge_large_en_v1_5/{version}"

model_local_path = "/local_disk0/bge_large_en_v1_5/"
model_output_local_path = "/local_disk0/bge_large_en_v1_5-fine-tune"

# COMMAND ----------

import json
from datasets import load_dataset

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

from mlflow.artifacts import download_artifacts

path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import os

sentence_transformer_local_path = os.path.join(path, "model.sentence_transformer")

model = SentenceTransformer(sentence_transformer_local_path)

# COMMAND ----------

from torch.utils.data import DataLoader
from sentence_transformers import losses
from accelerate import notebook_launcher

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=3)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

# COMMAND ----------

model.save(model_output_local_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save fine tuned model to mlflow

# COMMAND ----------

example_sentences = ["This is a sentence.", "This is another sentence."]

# Define the signature
signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=model.encode(example_sentences),
)

with mlflow.start_run() as run:
    mlflow.sentence_transformers.log_model(model,
                                        artifact_path="model",
                                        signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the saved model for inference

# COMMAND ----------

# Load model as a PyFuncModel.
run_id = run.info.run_id
logged_model = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a list of text
sentences = [
    "London has 9,787,426 inhabitants at the 2011 census",
    "London is known for its finacial district",
]

loaded_model.predict(sentences)

# COMMAND ----------

# If you need to search the long relevant passages to a short query,
# you need to add the instruction `Represent this sentence for searching relevant passages:` to the query
sentences = [
    "London has 9,787,426 inhabitants at the 2011 census",
    "London is known for its finacial district",
]

sentences = [
    "Represent this sentence for searching relevant passages: " + text
    for text in sentences
]
loaded_model.predict(sentences)

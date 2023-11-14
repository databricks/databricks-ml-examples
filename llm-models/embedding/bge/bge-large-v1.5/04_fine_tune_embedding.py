# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine-Tune `bge-large-en-v1.5` with Sentence Transformers
# MAGIC
# MAGIC This notebook demostrates how to fine tune [bge-large-en-v1.5 model](https://huggingface.co/BAAI/bge-large-en-v1.5).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.1 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure

# COMMAND ----------

import json
from datasets import load_dataset

# COMMAND ----------

output_model_path = "/dbfs/fine_tuned_bge_v1_5_model"

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
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# COMMAND ----------

from torch.utils.data import DataLoader
from sentence_transformers import losses
from accelerate import notebook_launcher

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=3)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

# COMMAND ----------

model.save(output_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save fine tuned model to mlflow

# COMMAND ----------

# Save the model as Pyfunc to mlflow
import mlflow
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
 
class SentenceTransformerEmbeddingModel(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    self.model = SentenceTransformer(context.artifacts["sentence-transformer-model"], device=device)
    
  def predict(self, context, model_input): 
    texts = model_input.iloc[:, 0].to_list()  # get the first column
    sentence_embeddings = self.model.encode(texts)
    return pd.Series(sentence_embeddings.tolist())

# COMMAND ----------

from mlflow.utils.environment import _mlflow_conda_env
import accelerate
import sentence_transformers
import cloudpickle
EMBEDDING_CONDA_ENV = _mlflow_conda_env(
    additional_pip_deps=[
        f"accelerate=={accelerate.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
        f"sentence-transformers=={sentence_transformers.__version__}",
    ]
)

# COMMAND ----------

import mlflow

with mlflow.start_run() as run:
  my_model = SentenceTransformerEmbeddingModel()
  model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=my_model, input_example=["London is known for its finacial district"], artifacts={"sentence-transformer-model": output_model_path}, conda_env=EMBEDDING_CONDA_ENV)

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

import mlflow
import pandas as pd

# Load model as a PyFuncModel.
run_id = run.info.run_id
logged_model = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
test_df = pd.DataFrame(['London has 9,787,426 inhabitants at the 2011 census',
              'London is known for its finacial district'], columns=["text"])

loaded_model.predict(test_df)

# COMMAND ----------

# If you need to search the long relevant passages to a short query,
# you need to add the instruction `Represent this sentence for searching relevant passages:` to the query
test_df = pd.DataFrame(['London has 9,787,426 inhabitants at the 2011 census',
                        'London is known for its finacial district'], columns=["text"])

# Add the instruction to each entry in the "text" column
test_df['text'] = 'Represent this sentence for searching relevant passages: ' + test_df['text']
loaded_model.predict(test_df)

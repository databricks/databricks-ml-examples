# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Build vector database with `bge-large-en`
# MAGIC
# MAGIC This notebook demostrates how to build a vector store with [faiss](https://github.com/facebookresearch/faiss) using [bge-large-en model](https://huggingface.co/BAAI/bge-large-en).
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.3 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries

# COMMAND ----------

# MAGIC %pip install langchain==0.0.262 faiss-gpu==1.7.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
from datasets import load_dataset
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.faiss import FAISS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Document data
# MAGIC
# MAGIC We will use [peft doc](https://huggingface.co/datasets/smangrul/peft_docs) as an example to build the document database.

# COMMAND ----------

df = load_dataset('smangrul/peft_docs', split='train')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Indexing
# MAGIC
# MAGIC While there are many fields avaiable in loaded table, the fields that are relevant to build vector database are:
# MAGIC
# MAGIC * `chunk_content`: Documentation text
# MAGIC * `filename`: Filename pointing to the document
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The content available within each doc varies but some documents can be quite long. The process of converting a document to an index involves us translating it to a fixed-size embedding.  An embedding is a set of numerical values, kind of like a coordinate, that summarizes the content in a unit of text. While large embeddings are capable of capturing quite a bit of detail about a document, the larger the document submitted to it, the more the embedding generalizes the content.  It's kind of like asking someone to summarize a paragraph, a chapter or an entire book into a fixed number of dimensions.  The greater the scope, the more the summary must eliminate detail and focus on the higher-level concepts in the text.
# MAGIC
# MAGIC A common strategy for dealing with this when generating embeddings is to divide the text into chunks.  These chunks need to be large enough to capture meaningful detail but not so large that key elements get washed out in the generalization.  Its more of an art than a science to determine an appropriate chunk size, but here we'll use a very small chunk size to illustrate what's happening in this step:
# MAGIC

# COMMAND ----------

chunk_size = 3500
chunk_overlap = 400

def get_chunks(text):
  # instantiate tokenization utilities
  text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # split text into chunks
  return text_splitter.split_text(text)

df = df.map(lambda example: {"chunks": get_chunks(example["chunk_content"])})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Store
# MAGIC
# MAGIC With our data divided into chunks, we are ready to convert these records into searchable embeddings.

# COMMAND ----------

df.set_format("pandas", columns=["filename", "chunks"])
inputs = df["chunks"]

# COMMAND ----------

df.format

# COMMAND ----------

text_df = df[:].explode('chunks')

# COMMAND ----------

text_inputs = text_df["chunks"].to_list()

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# COMMAND ----------

metadata_inputs = text_df.to_dict(orient='records')

# COMMAND ----------

vector_store = FAISS.from_texts(
  embedding=model, 
  texts=text_inputs, 
  metadatas=metadata_inputs,
  distance_strategy="COSINE",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC In order to user the vector store in other notebooks, we can persist vector to storage:

# COMMAND ----------

vector_store.save_local(folder_path="/dbfs/peft-doc-embed/vector_store")

# COMMAND ----------



# Databricks notebook source
# MAGIC %md # AutoML Embeddings with LangChain
# MAGIC
# MAGIC LangChain is a tool that allows you to easily connect large language models with vector stores. In this specific example, we are assuming we have trained an embeddings model with AutoML Embeddings and have the `run_id` of the experiment on hand.
# MAGIC
# MAGIC Requirements
# MAGIC - Databricks Runtime ML 13.1 and above
# MAGIC - (Recommended) GPU instances

# COMMAND ----------

# MAGIC %pip install -q faiss-gpu
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md #### Note on Dataset Preparation and Inference
# MAGIC
# MAGIC It is highly recommended that you preprocess the dataset that you want to use for inference. Here are some recommendations:
# MAGIC
# MAGIC 1. Even before fine-tuning the model, it is highly recommended that you remove any non-plain-text elements, like HTML tags or reStructuredText elements. We want to ideally have clean, easily readable text, for AutoML Embeddings.
# MAGIC 2. Chunk the text contents of the table to ~500 tokens or ~1500 characters. For example, using the following code:
# MAGIC
# MAGIC ```
# MAGIC df = spark.read.table(<table name>)
# MAGIC
# MAGIC # User-defined function to split a string into chunks of a given length
# MAGIC def chunk_string(s, chunk_size=1500):
# MAGIC     return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
# MAGIC
# MAGIC chunk_string_udf = udf(chunk_string, ArrayType(StringType()))
# MAGIC
# MAGIC # Split each string and explode the resulting arrays to create new rows
# MAGIC df_chunked = df.withColumn("text_chunks", chunk_string_udf(col("text"))) \
# MAGIC                .withColumn("chunk", explode(col("text_chunks"))) \
# MAGIC                .drop("text", "text_chunks")
# MAGIC
# MAGIC df_chunked.write.saveAsTable(<updated table name>)
# MAGIC ```
# MAGIC
# MAGIC 3. Prefix each chunk with the text "passage: ", which is highly recommended for embedding models from the [e5-suite](https://huggingface.co/intfloat/e5-base-v2). For example, using the following code:
# MAGIC ```
# MAGIC df = spark.read.table(<table name>)
# MAGIC
# MAGIC df_prefixed = df.withColumn("text", concat(lit("passage: "), col("text")))
# MAGIC
# MAGIC df_prefixed.write.saveAsTable(<updated table name>)
# MAGIC ```
# MAGIC 4. It is recommended to prefix each of your queries with the text "query: " for optimal performance.

# COMMAND ----------

# Note for user: Fill out the following variables with the mlflow models, datasets, and related information in this cell. 

# The run_id from the AutoML Embeddings experiment
run_id = "TODO"
# The delta table that you want to be able to query
delta_table_name = "TODO"
# The text column within the delta table
text_column_name = "TODO"

# COMMAND ----------

# MAGIC %md ### Setup
# MAGIC
# MAGIC LangChain already has support for [text embeddings models](https://python.langchain.com/docs/integrations/text_embedding/). Here we are wrapping our MLFlow model to work with Langchain.

# COMMAND ----------

import mlflow
from langchain.embeddings.base import Embeddings
from typing import Any, Dict, List, Optional

# Creating a version of Langchain's Embeddings that can support a mlflow experiment's run_id
class E5TransformerFromMlflow(Embeddings):
    def __init__(self, run_id: str):
        # Gets the latest model trained by AutoML
        latest_model_path = mlflow.MlflowClient().list_artifacts(run_id)[-1].path
        model_uri = f"runs:/{run_id}/{latest_model_path}"
        self.model = mlflow.sentence_transformers.load_model(model_uri)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Prefix each text with "passage: " if they aren't already prefixed
        texts = [text if text.startswith("passage:") else f"passage: {text}" for text in texts]
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # Prefix query with "query: " if it isn't already prefixed
        text = text if text.startswith("query:") else f"query: {text}"
        embedding = self.model.encode(text)
        return embedding.tolist()

# COMMAND ----------

# MAGIC %md ### Querying the FAISS Vector Store
# MAGIC
# MAGIC This is a simple example of creating a FAISS vector store using an existing Spark DataFrame using the [PySparkDataFrameLoader](https://python.langchain.com/docs/integrations/document_loaders/pyspark_dataframe)

# COMMAND ----------

from langchain.vectorstores import FAISS
from langchain.document_loaders import PySparkDataFrameLoader

fine_tuned_embeddings_model = E5TransformerFromMlflow(run_id = run_id)
spark_df = spark.table(delta_table_name)
loader = PySparkDataFrameLoader(spark, spark_df, page_content_column=text_column_name)
documents = loader.load()

faiss_vector_store = FAISS.from_documents(documents, embedding=fine_tuned_embeddings_model)

# COMMAND ----------

# Note for user: fill out this section with any query

faiss_vector_store.similarity_search("TODO")[0]

# COMMAND ----------

# MAGIC %md ### Using an Existing Model for Comparison
# MAGIC
# MAGIC Here, we will use the model `intfloat/e5-base-v2` ([source](https://huggingface.co/intfloat/e5-base-v2)) from the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) benchmark to compare with our fine-tuned model.

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings

external_embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
external_faiss_vector_store = FAISS.from_documents(documents, embedding=external_embeddings_model)

# COMMAND ----------

# Note for user: fill out this section with any query

external_faiss_vector_store.similarity_search("TODO")[0]

# COMMAND ----------

# MAGIC %md ### (Optional) Creating a Doc Q&A Bot using OpenAI
# MAGIC
# MAGIC We can use OpenAI with the vector store created above to answer questions. Uncomment the following section if you have a valid OpenAI API key. If you want to use another approach, Langchain has support for a variety of other [language models](https://python.langchain.com/docs/integrations/llms/).

# COMMAND ----------

# import os
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain

# os.environ["OPENAI_API_KEY"] = ""

# model = ChatOpenAI(model_name="gpt-3.5-turbo")
# qa = ConversationalRetrievalChain.from_llm(model, retriever=faiss_vector_store.as_retriever())
# print(qa.run({"question": "TODO", "chat_history": []}))

# Databricks notebook source
# MAGIC %md
# MAGIC # Run `bge-large-en` Embedding on Databricks
# MAGIC
# MAGIC [bge-large-en (BAAI General Embedding) model](https://huggingface.co/BAAI/bge-large-en) can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification, clustering, or semantic search. And it also can be used in vector database for LLMs.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.3 GPU ML Runtime
# MAGIC - Instance: `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure.
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U langchain==0.0.262
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Sentence-Transformers
# MAGIC
# MAGIC You can use sentence-transformers with the bge-large-en model to encode sentences as embeddings.

# COMMAND ----------

from sentence_transformers import SentenceTransformer, util


model_name = "BAAI/bge-large-en"
model = SentenceTransformer(model_name)


sentences = ["A man is eating food.",
  "A man is eating a piece of bread.",
  "The girl is carrying a baby.",
  "A man is riding a horse.",
  "A woman is playing violin.",
  "Two men pushed carts through the woods.",
  "A man is riding a white horse on an enclosed ground.",
  "A monkey is playing drums.",
  "Someone in a gorilla costume is playing a set of drums.",
]
embeddings = model.encode(sentences, normalize_embeddings=True)

cos_sim = util.cos_sim(embeddings, embeddings)
print("Cosine-Similarity:", cos_sim)

# COMMAND ----------

# MAGIC %md
# MAGIC For s2p(short query to long passage) retrieval task, each short query should start with an instruction `Represent this sentence for searching relevant passages:`.

# COMMAND ----------

queries = ["What type of organism is commonly used in preparation of foods such as cheese and yogurt?"]
passages = [
  "Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F). Mesophiles are often found living in or on the bodies of humans or other animals. The optimal growth temperature of many pathogenic mesophiles is 37°C (98°F), the normal human body temperature. Mesophilic organisms have important uses in food preparation, including cheese, yogurt, beer and wine.", 
  "Without Coriolis Effect the global winds would blow north to south or south to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere.",
  "Summary Changes of state are examples of phase changes, or phase transitions. All phase changes are accompanied by changes in the energy of a system. Changes from a more-ordered state to a less-ordered state (such as a liquid to a gas) areendothermic. Changes from a less-ordered state to a more-ordered state (such as a liquid to a solid) are always exothermic. The conversion of a solid to a liquid is called fusion (or melting). The energy required to melt 1 mol of a substance is its enthalpy of fusion (ΔHfus). The energy change required to vaporize 1 mol of a substance is the enthalpy of vaporization (ΔHvap). The direct conversion of a solid to a gas is sublimation. The amount of energy needed to sublime 1 mol of a substance is its enthalpy of sublimation (ΔHsub) and is the sum of the enthalpies of fusion and vaporization. Plots of the temperature of a substance versus heat added or versus heating time at a constant rate of heating are calledheating curves. Heating curves relate temperature changes to phase transitions. A superheated liquid, a liquid at a temperature and pressure at which it should be a gas, is not stable. A cooling curve is not exactly the reverse of the heating curve because many liquids do not freeze at the expected temperature. Instead, they form a supercooled liquid, a metastable liquid phase that exists below the normal melting point. Supercooled liquids usually crystallize on standing, or adding a seed crystal of the same or another substance can induce crystallization."
  ]
instruction = "Represent this sentence for searching relevant passages: "
query_with_instruction = [instruction+q for q in queries]
q_embeddings = model.encode(query_with_instruction, normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)

scores = util.cos_sim(q_embeddings, p_embeddings)
print("Cosine-Similarity scores:", scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Langchain
# MAGIC

# COMMAND ----------

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

q_embeddings = model_norm.embed_documents(query_with_instruction)
p_embeddings = model_norm.embed_documents(passages)

scores = util.cos_sim(q_embeddings, p_embeddings)
print("Cosine-Similarity scores:", scores)

# COMMAND ----------



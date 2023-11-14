<!---
Copyright (C) 2023 Databricks, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Example notebooks for BGE Model family on Databricks

## Model List

`bge` is short for `BAAI general embedding`.

| Model                                                                               | Language |              query instruction for retrieval\*              |
|:------------------------------------------------------------------------------------|:--------:|:-----------------------------------------------------------:|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)                       | English  | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)                         | English  | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)                       | English  | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)                       | Chinese  |                    `为这个句子生成表示以用于检索相关文章：`                    |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh)                         | Chinese  |                    `为这个句子生成表示以用于检索相关文章：`                    |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh)                       | Chinese  |                    `为这个句子生成表示以用于检索相关文章：`                    |

\*: If you need to search the **long** relevant passages to a **short** query (s2p retrieval task), you need to add the instruction to the query; in other cases, no instruction is needed, just use the original query directly. In all cases, **no instruction** need to be added to passages.

## Example notebooks
This folder contains the following examples for BGE models:

<!---
<style>
table th:first-of-type {
    width: 10%;
}
table th:nth-of-type(2) {
    width: 30%;
}
table th:nth-of-type(3) {
    width: 30%;
}
table th:nth-of-type(4) {
    width: 30%;
}
</style>
-->

|                      **File**                       |                                         **Description**                                          | **GPU Minimum Requirement** |
|:---------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:---------------------------:|
|                 `01_load_inference`                 |    Environment setup and suggested configurations when inferencing BGE models on Databricks.     |         1xT4          |
|            `02_mlflow_logging_inference`            | Save, register, and load BGE models with MLFlow, and create a Databricks model serving endpoint. |         1xT4          |
|              `03_build_document_index`              |                        Build a vector store with faiss using BGE models.                         |         1xT4          |
|              `04_fine_tune_embedding`               |                                       Fine-tune BGE models.                                       |             1xT4             |

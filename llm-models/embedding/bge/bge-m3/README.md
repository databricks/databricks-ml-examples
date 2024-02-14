<!---
Copyright (C) 2024 Databricks, Inc.

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

# Example notebooks for the `bge-m3` model on Databricks
This folder contains the following examples for [bge-m3 model](https://huggingface.co/BAAI/bge-m3): 

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

|                           **File**                            |                                         **Description**                                          |    **GPU Minimum Requirement**     |
|:-------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:----------------------------------:|
|           [01_load_inference](01_load_inference.py)           |    Environment setup and suggested configurations when inferencing BGE models on Databricks.     |            1xV100-16GB or 1xT4-16GB   |
| [02_mlflow_logging_inference](02_mlflow_logging_inference.py) | Save, register, and load BGE models with MLFlow, and create a Databricks model serving endpoint. |            1xV100-16GB or 1xT4-16GB   |
|     [03_build_document_index](03_build_document_index.py)     |                        Build a vector store with faiss using BGE models.                         |            1xV100-16GB or 1xT4-16GB   |
|      [04_fine_tune_embedding](04_fine_tune_embedding.py)      |                                       Fine-tune BGE models                                       |            1xV100-16GB or 1xT4-16GB   |
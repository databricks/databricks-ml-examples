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

![alt text](https://about.fb.com/wp-content/uploads/2023/07/Next-generation-of-Llama-2-AI_header.jpg)

# Example notebooks for the Llama 2 7B model on Databricks
This folder contains the following examples for Llama 2 models: 
`
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

|                           **File**                            |                                           **Description**                                            |    **Model Used**    | **GPU Minimum Requirement** |
|:-------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:--------------------:|:---------------------------:|
|           [01_load_inference](01_load_inference.py)           |    Environment setup and suggested configurations when inferencing Llama 2 models on Databricks.     | `Llama-2-7b-chat-hf` |         1xA10-24GB          |
| [02_mlflow_logging_inference](02_mlflow_logging_inference.py) | Save, register, and load Llama 2 models with MLflow, and create a Databricks model serving endpoint. | `Llama-2-7b-chat-hf` |         1xA10-24GB          |
|       [03_serve_driver_proxy](03_serve_driver_proxy.py)       |                     Serve Llama 2 models on the cluster driver node using Flask.                     | `Llama-2-7b-chat-hf` |         1xA10-24GB          |
|                [04_langchain](04_langchain.py)                |          Integrate a serving endpoint or cluster driver proxy app with LangChain and query.          |         N/A          |             N/A             |
|      [05_fine_tune_deepspeed](05_fine_tune_deepspeed.py)      |                         Fine-tune Llama 2 base models leveraging DeepSpeed.                          |   `Llama-2-7b-hf`    |    4xA10 or 2xA100-80GB     |
|          [06_fine_tune_qlora](06_fine_tune_qlora.py)          |                              Fine-tune Llama 2 base models with QLORA.                               |   `Llama-2-7b-hf`    |            1xA10            |
|               [07_ai_gateway](07_ai_gateway.py)               |         Manage a MLflow AI Gateway Route that accesses a Databricks model serving endpoint.          |         N/A          |             N/A             |

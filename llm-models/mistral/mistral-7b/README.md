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

![alt text](https://techcrunch.com/wp-content/uploads/2023/09/mistral-7b-v0.1.jpg?w=1390&crop=1)

# Example notebooks for the mistral 7B model on Databricks
This folder contains the following examples for mistral-7b models: 
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

|                                                                                  **File**                                                                                  |                                                           **Description**                                                            |    **Model Used**     | **GPU Minimum Requirement** |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:---------------------------:|
|                  [01_load_inference](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/01_load_inference.py)                  |              Environment setup and suggested configurations when inferencing mistral-7b-instruct models on Databricks.               | `mistral-7b-instruct` |         1xA10-24GB          |
|        [02_mlflow_logging_inference](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/02_mlflow_logging_inference.py)        |           Save, register, and load mistral-7b-instruct models with MLflow, and create a Databricks model serving endpoint.           | `mistral-7b-instruct` |         1xA10-24GB          |
| [02_[chat]_mlflow_logging_inference](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/02_[chat]_mlflow_logging_inference.py) | Save, register, and load mistral-7b-instruct models with MLflow, and create a Databricks model serving endpoint for chat completion. | `mistral-7b-instruct` |         1xA10-24GB          |
|              [03_serve_driver_proxy](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/03_serve_driver_proxy.py)              |                               Serve mistral-7b-instruct models on the cluster driver node using Flask.                               | `mistral-7b-instruct` |         1xA10-24GB          |
|       [03_[chat]_serve_driver_proxy](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/03_[chat]_serve_driver_proxy.py)       |                     Serve mistral-7b-instruct models as chat completion on the cluster driver node using Flask.                      | `mistral-7b-instruct` |         1xA10-24GB          |
|                       [04_langchain](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/04_langchain.py)                       |                          Integrate a serving endpoint or cluster driver proxy app with LangChain and query.                          |          N/A          |             N/A             |
|                [04_[chat]_langchain](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/04_[chat]_langchain.py)                |                                     Integrate a serving endpoint and setup langchain chat model.                                     |          N/A          |             N/A             |
|             [05_fine_tune_deepspeed](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/05_fine_tune_deepspeed.py)             |                                          Fine-tune mistral-7b models leveraging DeepSpeed.                                           |     `mistral-7b`      |    4xA10 or 2xA100-80GB     |
|                 [06_fine_tune_qlora](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/06_fine_tune_qlora.py)                 |                                               Fine-tune mistral-7b models with QLORA.                                                |     `mistral-7b`      |            1xA10            |
|                      [07_ai_gateway](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mistral/mistral-7b/07_ai_gateway.py)                      |                         Manage a MLflow AI Gateway Route that accesses a Databricks model serving endpoint.                          |          N/A          |             N/A             |

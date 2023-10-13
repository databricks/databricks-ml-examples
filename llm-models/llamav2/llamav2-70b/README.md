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

# Example notebooks for the Llama 2 70B model on Databricks
This folder contains the following examples for Llama 2 models: 

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

|                  **File**                   | **Description**                                                                                                       |    **Model Used**     | **GPU Minimum Requirement** |
|:-------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|:---------------------:|:---------------------------:|
|  [01_load_inference](01_load_inference.py)  | Environment setup and suggested configurations when inferencing Llama 2 models on Databricks.                         | `Llama-2-70b-chat-hf` |         2xA100-80GB         |
| [06_fine_tune_qlora](06_fine_tune_qlora.py) | Fine-tune Llama 2 base models with QLORA.                                                                             |   `Llama-2-70b-hf`    |         1xA100-80GB         |
| [08_load_from_marketplace](08_load_from_marketplace.py)  |         Load Llama 2 models from Databricks Marketplace.	          |     `Llama-2-70b-chat-hf`   |            2xA100-80GB             |

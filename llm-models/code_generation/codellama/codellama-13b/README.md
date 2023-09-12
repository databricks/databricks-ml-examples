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

# Example notebooks for the Code Llama 13B model on Databricks
This folder contains the following examples for code llama 13B models: 

<!---
<style>
table th:first-of-type(1) {
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

|                           **File**                            |                                               **Description**                                               |                                     **Model Used**                                     | **GPU Minimum Requirement** |
|:-------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:---------------------------:|
|           [01_load_inference](01_load_inference.py)           |    Environment setup and suggested configurations when inferencing Code Llama 13B models on Databricks.     | `CodeLlama-13b-hf` <br> `CodeLlama-13b-hf-instructions` <br> `CodeLlama-13b-hf-python` |         2xA10-24GB          |

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

![alt text](https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/6454669e10b0b051b6a393a6_Frame%201%20(12).png)
# Example notebooks for MPT-7B modelå on Databricks
This folder contains the following examples for [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) models: 

| File                                                          | Description                                                                                        | Model used        | GPU minimum requirement |
|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------|-------------------|-------------------------|
| [01_load_inference](01_load_inference.py)                     | Environment setup and suggested configurations when using  MPT models for inference on Databricks. | `MPT-7b-instruct` | 1xV100-16GB             |
| [02_mlflow_logging_inference](02_mlflow_logging_inference.py) | Save, register, and load MPT models with MLflow, and create a Databricks model serving endpoint.   | MPT-7b-instruct`  | 1xV100-16GB             |
| [03_serve_driver_proxy](03_serve_driver_proxy.py)             | Serve MPT models on the cluster driver node with Flask.                                            | MPT-7b-instruct`  | 1xV100-16GB             |
| [04_langchain](04_langchain.py)                               | Wrap a serving endpoint or cluster driver proxy app with LangChain and query it.                   | N/A               | N/A                     |
| [05_fine_tune_deepspeed](05_fine_tune_deepspeed.py)           | Fine-tune MPT base models with DeepSpeed.                                                          | `MPT-7b`          | 4xA10 or 2xA100-80GB    |

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


# Example notebooks for the Falcon model family on Databricks
This folder contains the following examples for Falcon models: 

| File  | Description | Model used | GPU minimum requirement |
| --- | --- | --- | --- |
| `01_load_inference`  | Environment setup and suggested configurations when using Falcon models for inference on Databricks. | `falcon-7b-instruct`<br>`falcon-40b-instruct`  | 1xV100-16GB |
| `02_mlflow_logging_inference` | Save, register, and load Falcon models with MLflow, and create a Databricks model serving endpoint. | `falcon-7b-instruct`<br>`falcon-40b-instruct`  | 1xV100-16GB |
| `03_serve_driver_proxy` | Serve Falcon models on the cluster driver node with Flask.  | `falcon-7b-instruct`<br>`falcon-40b-instruct` | 1xV100-16GB |
| `04_langchain` | Wrap a serving endpoint or cluster driver proxy app with LangChain and query it. | N/A | N/A |
| `05_fine_tune_deepspeed` | Fine-tune Falcon base models with DeepSpeed. | `falcon-7b` | 4xA10 or 2xA100-80GB |

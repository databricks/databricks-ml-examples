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


# Example notebooks for the Mixtral-8x7B models on Databricks

[mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) and [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) are a is a pretrained generative Sparse Mixture of Experts.

- Outperforms Llama 2 70B on most benchmarks we tested.
- It gracefully handles a context of 32k tokens.
- It handles English, French, Italian, German and Spanish.
- It shows strong performance in code generation.
- It can be finetuned into an instruction-following model that achieves a score of 8.3 on MT-Bench.

Mixtral 8x7B is a high-quality sparse mixture of experts model (SMoE) with open weights. Licensed under Apache 2.0.

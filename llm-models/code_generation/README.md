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

# Code generation LLM models

Code generation LLM models are large language models used specifically for generating 
or assisting with code completion or code generation tasks.

## Example notebooks for code generation models on Databricks

This folder contains:
- `codellama/`: Example notebooks for codellama models

## Model evaluation and benchmark

We have collected the performance of code generation models on [OpenAI HumanEval benchmark](https://github.com/openai/human-eval).
For a comparison of the multilingual code generation performance, check [big code models leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard).

| Models                                                                              | HumaEval | Peak memory(MB)  |
|-------------------------------------------------------------------------------------|----------|------------------|
| [WizardCoder-Python-34B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0) | 70.73    |  69957 |
| [WizardCoder-Python-13B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-13B-V1.0) | 62.19    |  28568 |
| [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)        | 58.12    |  15400 |
| [CodeLlama-34b-Python](codellama/codellama-34b)                                     | 53.29    |  69957 |
| [CodeLlama-34b-Instruct](codellama/codellama-34b)                                   | 50.79    |  69957 |
| [CodeLlama-13b-Instruct](codellama/codellama-13b)                                   | 50.6     |  28568 |
| [CodeLlama-7b-Instruct](codellama/codellama-7b)                                     | 45.65    |  15853 |
| [CodeLlama-34b](codellama/codellama-34b)                                            | 45.11    |  69957 |
| [CodeLlama-13b-Python](codellama/codellama-13b)                                     | 42.89    |  28568 |
| [CodeLlama-7b-Python](codellama/codellama-7b)                                       | 40.48    |  15853 |
| [CodeLlama-13b](codellama/codellama-13b)                                            | 35.07    |  28568 |
| [CodeLlama-7b](codellama/codellama-7b)                                              | 29.98    |  15853 |

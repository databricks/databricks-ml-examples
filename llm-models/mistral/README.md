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


# Example notebooks for the mistral models on Databricks

[Mistral 7B](https://huggingface.co/mistralai) is a 7.3B parameter model that:

- Outperforms Llama 2 13B on all benchmarks
- Outperforms Llama 1 34B on many benchmarks 
- Approaches CodeLlama 7B performance on code, while remaining good at English tasks 
- Uses Grouped-query attention (GQA) for faster inference 
- Uses Sliding Window Attention (SWA) to handle longer sequences at smaller cost

Mistral 7B is under the Apache 2.0 license, it can be used without restrictions.
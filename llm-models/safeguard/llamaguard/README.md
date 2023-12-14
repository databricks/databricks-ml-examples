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

# Llama Guard model
[LlamaGuard-7b](https://huggingface.co/meta-llama/LlamaGuard-7b) is a 7B parameter Llama 2-based input-output safeguard model. It can be used for classifying content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM: it generates text in its output that indicates whether a given prompt or response is safe/unsafe, and if unsafe based on a policy, it also lists the violating subcategories.

![alt text](https://scontent-sea1-1.xx.fbcdn.net/v/t39.8562-6/408685155_1399963250601071_251699255500120597_n.png?_nc_cat=102&ccb=1-7&_nc_sid=f537c7&_nc_ohc=-IS-eBEJvsEAX_3P2vD&_nc_ht=scontent-sea1-1.xx&oh=00_AfBh8ZVJ5aQrNeu86buig1MOjFRy33B2QMRnN5bqDGsjEg&oe=657FCFAD)

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

# Instructor-XL model
[Instructor-XL](https://huggingface.co/hkunlp/instructor-xl) is an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning.

![alt text](https://instructor-embedding.github.io/static/images/instructor.png)

## Calculate embeddings for your customized texts
If you want to calculate customized embeddings for specific sentences, you may follow the unified template to write instructions:

                          Represent the domain text_type for task_objective:

domain is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
text_type is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
task_objective is optional, and it specifies the objective of embedding, e.g., retrieve a document, classify the sentence, etc.
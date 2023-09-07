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

# E5 Model family
E5 is a text embedding model based on [Text Embeddings by Weakly-Supervised Contrastive Pre-training.](https://arxiv.org/pdf/2212.03533.pdf) Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022.

## Model List

E5 models have the following variations:

| Model                                                               | Model Size (GB) | Embedding Dimensions | 
|:--------------------------------------------------------------------|:----------------|:--------------------:|
| [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1.34            |         1024         |
| [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)   | 0.44            |         768          | 
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | 0.13            |         384          |

## FAQ

**1. Do I need to add the prefix "query: " and "passage: " to input texts?**

Yes, this is how the model is trained, otherwise you will see a performance degradation.

Here are some rules of thumb:
- Use "query: " and "passage: " correspondingly for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.

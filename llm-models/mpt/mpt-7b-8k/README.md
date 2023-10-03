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

![alt text](https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/64b66e8fee9e6ec6db9e7eff_MPT-7B-8K_Announcement_01%20(1).jpg)

# MPT-7b-8k models
[MPT-7B-8K](https://www.mosaicml.com/blog/long-context-mpt-7b-8k) are 7B parameter open-source LLM models with 8k context length trained with the MosaicML platform. It contains 2 models which are commercializable:
- [MPT-7B-8k](https://huggingface.co/mosaicml/mpt-7b-8k): A decoder-style transformer pretrained starting from MPT-7B, but updating the sequence length to 8k and training for an additional 500B tokens, resulting in a total of 1.5T tokens of text and code. License: CC-BY-SA-3.0
- [MPT-7B-8k-Instruct](https://huggingface.co/mosaicml/mpt-7b-8k-instruct): a model for long-form instruction following (especially summarization and question-answering). Built by finetuning MPT-7B-8k on several carefully curated datasets. License: CC-BY-SA-3.0

## MPT-7B-8k FAQ
When would I choose…

- **MPT-7B-8k over MPT-7B?** Use 8k in most cases, except when coding or reasoning ability are the only criteria, in which case you should evaluate both models
- **MPT-7B-8k-Instruct over MPT-7B-Instruct?** 8k-Instruct excels at longform instruction following; use it when you have inputs longer than 2048 tokens or for summarization and question answering.

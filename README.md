

# databricks-ml-examples

`databricks/databricks-ml-examples` is a repository to show machine learning examples on Databricks platforms.

Currently this repository contains:
- `llm-models/`: Example notebooks to use different **State of the art (SOTA) models on Databricks**.
- `llm-fine-tuning/`: Fine tuning scripts and notebooks to fine tune **State of the art (SOTA) models on Databricks**.

## SOTA LLM examples

Databricks works with thousands of customers to build generative AI applications. While you can use Databricks to work with any generative AI model, including commercial and research, the table below lists our current model recommendations for popular use cases. **Note:** The table only lists open source models that are for free commercial use. 

<!---
<style>
table th:first-of-type {
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

| Use case                               | Quality-optimized                                                                                                                                                                                                                                                 | Balanced                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Speed-optimized                                                                                          |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Text generation following instructions | [Mixtral-8x7B-Instruct-v0.1](llm-models/mixtral-8x7b)  <br> <br> [Llama-2-70b-chat-hf](llm-models/llamav2/llamav2-70b)                                                                                                                                                       | [mistral-7b](llm-models/mistral/mistral-7b) <br><br> [MPT-7B-Instruct](llm-models/mpt/mpt-7b) <br> [MPT-7B-8k-Instruct](llm-models/mpt/mpt-7b-8k) <br> <br> [Llama-2-7b-chat-hf](llm-models/llamav2/llamav2-7b) <br> [Llama-2-13b-chat-hf](llm-models/llamav2/llamav2-13b)                                                                                                                                                                                                                                                         |  [phi-2](llm-models/phi-2)                                          |
| Text embeddings (English only)         |   [e5-mistral-7b-instruct(7B)](llm-models/embedding/e5-mistral-7b-instruct)                                                                                                                                                                                               | [bge-large-en-v1.5(0.3B)](llm-models/embedding/bge) <br> [e5-large-v2 (0.3B)](llm-models/embedding/e5-v2)                                                                                                                                                                                                                                                                                                                                                                                                                | [bge-base-en-v1.5 (0.1B)](llm-models/embedding/bge) <br> [e5-base-v2 (0.1B)](llm-models/embedding/e5-v2) |
| Transcription (speech to text)         |                                                                                                                                                                                                                                                                   | [whisper-large-v2](llm-models/transcription/whisper)(1.6B) <br> [whisper-medium](llm-models/transcription/whisper) (0.8B)                                                                                                                                                                                                                                                                                                                                                                                                          |                                                                                                          |
| Image generation                       |                                                                                                                                                                                                                                                                   | [stable-diffusion-xl](llm-models/image_generation/stable_diffusion)                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                          |
| Code generation                        | [CodeLlama-70b-hf](llm-models/code_generation/codellama/codellama-70b) <br> [CodeLlama-70b-Instruct-hf](llm-models/code_generation/codellama/codellama-70b) <br> [CodeLlama-70b-Python-hf](llm-models/code_generation/codellama/codellama-70b) (Python optimized) <br>[CodeLlama-34b-hf](llm-models/code_generation/codellama/codellama-34b) <br> [CodeLlama-34b-Instruct-hf](llm-models/code_generation/codellama/codellama-34b) <br> [CodeLlama-34b-Python-hf](llm-models/code_generation/codellama/codellama-34b) (Python optimized) | [CodeLlama-13b-hf](llm-models/code_generation/codellama/codellama-13b) <br> [CodeLlama-13b-Instruct-hf](llm-models/code_generation/codellama/codellama-13b) <br> [CodeLlama-13b-Python-hf](llm-models/code_generation/codellama/codellama-13b) (Python optimized) <br> [CodeLlama-7b-hf](llm-models/code_generation/codellama/codellama-7b) <br> [CodeLlama-7b-Instruct-hf](llm-models/code_generation/codellama/codellama-7b) <br> [CodeLlama-7b-Python-hf](llm-models/code_generation/codellama/codellama-7b) (Python optimized) |                                                                                                          |

* To get a better performance on instructor-xl, you may follow [the unified template to write instructions](https://huggingface.co/hkunlp/instructor-xl#calculate-embeddings-for-your-customized-texts).

## Model Evaluation Leaderboard
**Text generation models**

The model evaluation results presented below are measured by the [Mosaic Eval Gauntlet](https://www.mosaicml.com/llm-evaluation) framework. This framework comprises a series of tasks specifically designed to assess the performance of language models, including widely-adopted benchmarks such as MMLU, Big-Bench, HellaSwag, and more.

| Model Name                                                                            |   Core Average |   World Knowledge |   Commonsense Reasoning |   Language Understanding |   Symbolic Problem Solving |   Reading Comprehension |
|:--------------------------------------------------------------------------------------|---------------:|------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                   |          0.522 |             0.558 |                   0.513 |                    0.555 |                      0.342 |                   0.641 |
| [falcon-40b](https://huggingface.co/tiiuae/falcon-40b)                                |          0.501 |             0.556 |                   0.55  |                    0.535 |                      0.269 |                   0.597 |
| [falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)              |          0.5   |             0.542 |                   0.571 |                    0.544 |                      0.264 |                   0.58  |
| [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)                    |          0.479 |             0.515 |                   0.482 |                    0.52  |                      0.279 |                   0.597 |
| [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)          |          0.476 |             0.522 |                   0.512 |                    0.514 |                      0.271 |                   0.559 |
| [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |          0.469 |             0.48  |                   0.502 |                    0.492 |                      0.266 |                   0.604 |
| [mpt-30b-instruct](https://huggingface.co/mosaicml/mpt-30b-instruct)                  |          0.465 |             0.48  |                   0.513 |                    0.494 |                      0.238 |                   0.599 |
| [mpt-30b](https://huggingface.co/mosaicml/mpt-30b)                                    |          0.431 |             0.494 |                   0.47  |                    0.477 |                      0.234 |                   0.481 |
| [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)            |          0.42  |             0.476 |                   0.447 |                    0.478 |                      0.221 |                   0.478 |
| [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)                      |          0.401 |             0.457 |                   0.41  |                    0.454 |                      0.217 |                   0.465 |
| [mpt-7b-8k-instruct](https://huggingface.co/mosaicml/mpt-7b-8k-instruct)              |          0.36  |             0.363 |                   0.41  |                    0.405 |                      0.165 |                   0.458 |
| [mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)                    |          0.354 |             0.399 |                   0.415 |                    0.372 |                      0.171 |                   0.415 |
| [mpt-7b-8k](https://huggingface.co/mosaicml/mpt-7b-8k)                                |          0.354 |             0.427 |                   0.368 |                    0.426 |                      0.171 |                   0.378 |
| [falcon-7b](https://huggingface.co/tiiuae/falcon-7b)                                  |          0.335 |             0.371 |                   0.421 |                    0.37  |                      0.159 |                   0.355 |
| [mpt-7b](https://huggingface.co/mosaicml/mpt-7b)                                      |          0.324 |             0.356 |                   0.384 |                    0.38  |                      0.163 |                   0.336 |
| [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)                |          0.307 |             0.34  |                   0.372 |                    0.333 |                      0.108 |                   0.38  |

<p align="center">
  <img src="https://github.com/databricks/databricks-ml-examples/assets/12763339/acdfb7ce-c233-4ede-884c-4e0b4ce0a4f6" />
</p>

## Other examples:

- [DIY LLM QA Bot Accelerator](https://github.com/databricks-industry-solutions/diy-llm-qa-bot)
- [Biomedical Question Answering over Custom Datasets with LangChain and Llama 2 from Hugging Face](https://github.com/databricks-industry-solutions/hls-llm-doc-qa)
- [DIY QA LLM BOT](https://github.com/puneet-jain159/DSS_LLM_QA_Retrieval_Session/tree/main)
- [Tuning the Finetuning: An exploration of achieving success with QLoRA](https://github.com/avisoori-databricks/Tuning-the-Finetuning)
- [databricks-llm-fine-tuning](https://github.com/mshtelma/databricks-llm-fine-tuning)

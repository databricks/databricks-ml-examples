

# databricks-ml-examples

`databricks/databricks-ml-examples` is a repository to show machine learning examples on Databricks platforms.

Currently this repository contains:
- `llm-models/`: Example notebooks to use different **State of the art (SOTA) models on Databricks**.

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
| Text generation following instructions | [MPT-30B-Instruct](llm-models/mpt/mpt-30b/) <br> <br> [Llama-2-70b-chat-hf](llm-models/llamav2/llamav2-70b)                                                                                                                                                       | [mistral-7b](llm-models/mistral/mistral-7b) <br><br> [MPT-7B-Instruct](llm-models/mpt/mpt-7b) <br> [MPT-7B-8k-Instruct](llm-models/mpt/mpt-7b-8k) <br> <br> [Llama-2-7b-chat-hf](llm-models/llamav2/llamav2-7b) <br> [Llama-2-13b-chat-hf](llm-models/llamav2/llamav2-13b)                                                                                                                                                                                                                                                         |                                                                                                          |
| Text embeddings (English only)         |                                                                                                                                                                                                                                                                   | [bge-large-en-v1.5(0.3B)](llm-models/embedding/bge/bge-large) <br> [e5-large-v2 (0.3B)](llm-models/embedding/e5-v2)                                                                                                                                                                                                                                                                                                                                                                                                                | [bge-base-en-v1.5 (0.1B)](llm-models/embedding/bge) <br> [e5-base-v2 (0.1B)](llm-models/embedding/e5-v2) |
| Transcription (speech to text)         |                                                                                                                                                                                                                                                                   | [whisper-large-v2](llm-models/transcription/whisper)(1.6B) <br> [whisper-medium](llm-models/transcription/whisper) (0.8B)                                                                                                                                                                                                                                                                                                                                                                                                          |                                                                                                          |
| Image generation                       |                                                                                                                                                                                                                                                                   | [stable-diffusion-xl](llm-models/image_generation/stable_diffusion)                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                          |
| Code generation                        | [CodeLlama-34b-hf](llm-models/code_generation/codellama/codellama-34b) <br> [CodeLlama-34b-Instruct-hf](llm-models/code_generation/codellama/codellama-34b) <br> [CodeLlama-34b-Python-hf](llm-models/code_generation/codellama/codellama-34b) (Python optimized) | [CodeLlama-13b-hf](llm-models/code_generation/codellama/codellama-13b) <br> [CodeLlama-13b-Instruct-hf](llm-models/code_generation/codellama/codellama-13b) <br> [CodeLlama-13b-Python-hf](llm-models/code_generation/codellama/codellama-13b) (Python optimized) <br> [CodeLlama-7b-hf](llm-models/code_generation/codellama/codellama-7b) <br> [CodeLlama-7b-Instruct-hf](llm-models/code_generation/codellama/codellama-7b) <br> [CodeLlama-7b-Python-hf](llm-models/code_generation/codellama/codellama-7b) (Python optimized) |                                                                                                          |

* To get a better performance on instructor-xl, you may follow [the unified template to write instructions](https://huggingface.co/hkunlp/instructor-xl#calculate-embeddings-for-your-customized-texts).

## Model Evaluation Leaderboard
| Model Name                                                                            |   Core Average |   World Knowledge |   Commonsense Reasoning |   Language Understanding |   Symbolic Problem Solving |   Reading Comprehension |
|:--------------------------------------------------------------------------------------|---------------:|------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                   |       0.521512 |          0.557673 |                0.512723 |                 0.554642 |                   0.341637 |                0.640884 |
| [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)                    |       0.478508 |          0.514916 |                0.482057 |                 0.519822 |                   0.278985 |                0.596758 |
| [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)          |       0.475739 |          0.522327 |                0.512223 |                 0.514317 |                   0.270908 |                0.55892  |
| [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |       0.468611 |          0.479557 |                0.501512 |                 0.491598 |                   0.266257 |                0.604132 |
| [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)            |       0.420055 |          0.476493 |                0.447112 |                 0.47755  |                   0.22117  |                0.477952 |
| [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)                      |       0.400738 |          0.456689 |                0.409748 |                 0.454441 |                   0.217446 |                0.465366 |

## Other examples:

- [DIY LLM QA Bot Accelerator](https://github.com/databricks-industry-solutions/diy-llm-qa-bot)
- [Biomedical Question Answering over Custom Datasets with LangChain and Llama 2 from Hugging Face](https://github.com/databricks-industry-solutions/hls-llm-doc-qa)
- [DIY QA LLM BOT](https://github.com/puneet-jain159/DSS_LLM_QA_Retrieval_Session/tree/main)
- [Tuning the Finetuning: An exploration of achieving success with QLoRA](https://github.com/avisoori-databricks/Tuning-the-Finetuning)
- [databricks-llm-fine-tuning](https://github.com/mshtelma/databricks-llm-fine-tuning)

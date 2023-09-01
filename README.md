# databricks-ml-examples

`databricks/databricks-ml-examples` is a repository to show machine learning examples on Databricks platforms.

Currently this repository contains:
- `llm-models/`: Example notebooks to use different **State of the art (SOTA) models on Databricks**.

## SOTA LLM examples

Databricks works with thousands of customers to build generative AI applications. While you can use Databricks to work with any generative AI model, including commercial and research, the table below lists our current model recommendations* for popular use cases. **Note:** The table only lists open source models that are for free commercial use. 

| Use case  | Quality-optimized | Balanced | Speed-optimized |
| --------- | ----------------- | -------- | --------------- |
| Text generation following instructions  | [MPT-30B-Instruct](llm-models/mpt/mpt-30b) <br> [Llama-2-70b-chat-hf](llm-models/llamav2/llamav2-70b) <br> [Falcon-40B-Instruct](llm-models/falcon/falcon-40b) | [MPT-7B-Instruct](llm-models/mpt/mpt-7b) <br> [Llama-2-7b-chat-hf](llm-models/llamav2/llamav2-7b) <br> [Llama-2-13b-chat-hf](llm-models/llamav2/llamav2-13b) <br> [Falcon-7B-Instruct](llm-models/falcon/falcon-7b) |  |
| Text embeddings (English only)   | [instructor-xl (1.3B)](https://huggingface.co/hkunlp/instructor-xl)  |  [bge-large-en(0.3B)](llm-models/embedding/bge/bge-large) <br> [e5-large-v2 (0.3B)](https://huggingface.co/intfloat/e5-large-v2) | [bge-base-en (0.1B)](https://huggingface.co/BAAI/bge-base-en) <br> [e5-base-v2 (0.1B)](https://huggingface.co/intfloat/e5-large-v2) |
| Transcription (speech to text) | | [whisper-large-v2 (1.6B)](https://huggingface.co/openai/whisper-large-v2) <br> [whisper-medium](https://huggingface.co/openai/whisper-medium) (0.8B, English only) | |
| Image generation | | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | |
| Code generation  | [CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) <br> [CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) <br> [CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) (Python optimized)| [CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) <br> [CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) <br> [CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) (Python optimized) <br> [CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) <br> [CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) <br> [CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf) (Python optimized)| [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) |


## Other examples:

- [DIY LLM QA Bot Accelerator](https://github.com/databricks-industry-solutions/diy-llm-qa-bot)
- [Biomedical Question Answering over Custom Datasets with LangChain and Llama 2 from Hugging Face](https://github.com/databricks-industry-solutions/hls-llm-doc-qa)
- [DIY QA LLM BOT](https://github.com/puneet-jain159/DSS_LLM_QA_Retrieval_Session/tree/main)
- [Tuning the Finetuning: An exploration of achieving success with QLoRA](https://github.com/avisoori-databricks/Tuning-the-Finetuning)
- [databricks-llm-fine-tuning](https://github.com/mshtelma/databricks-llm-fine-tuning)

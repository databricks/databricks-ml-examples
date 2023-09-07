
| Use case  | Quality-optimized                                                                                                                                                                                                                                                                                                                                                       | Balanced                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Speed-optimized |
| --------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------------- |
| Text generation following instructions  | [MPT-30B-Instruct](llm-models/mpt/mpt-30b) <br> [Llama-2-70b-chat-hf](llm-models/llamav2/llamav2-70b)                                                                                                                                                                                                                                                                   | [MPT-7B-Instruct](llm-models/mpt/mpt-7b) <br> [Llama-2-7b-chat-hf](llm-models/llamav2/llamav2-7b) <br> [Llama-2-13b-chat-hf](llm-models/llamav2/llamav2-13b)                                                                                                                                                                                                                                                                                                                                                                                  |  |
| Text embeddings (English only)   |                                                                                                                                                                                                                                                                                                                                                                         | [bge-large-en(0.3B)](llm-models/embedding/bge/bge-large) <br> [e5-large-v2 (0.3B)](https://huggingface.co/intfloat/e5-large-v2) <br> [instructor-xl (1.3B)](https://huggingface.co/hkunlp/instructor-xl)*                                                                                                                                                                                                                                                                                                                                     | [bge-base-en (0.1B)](https://huggingface.co/BAAI/bge-base-en) <br> [e5-base-v2 (0.1B)](https://huggingface.co/intfloat/e5-large-v2) |
| Transcription (speech to text) |                                                                                                                                                                                                                                                                                                                                                                         | [whisper-large-v2 (1.6B)](https://huggingface.co/openai/whisper-large-v2) <br> [whisper-medium](https://huggingface.co/openai/whisper-medium) (0.8B, English only)                                                                                                                                                                                                                                                                                                                                                                            | |
| Image generation |                                                                                                                                                                                                                                                                                                                                                                         | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)                                                                                                                                                                                                                                                                                                                                                                                                                                                               | |
| Code generation  | [CodeLlama-34b-hf](llm-models/code_generation/codellama/codellama-34b) <br> [CodeLlama-34b-Instruct-hf](llm-models/code_generation/codellama/codellama-34b) <br> [CodeLlama-34b-Python-hf](llm-models/code_generation/codellama/codellama-34b) (Python optimized) <br> <br> [WizardCoder-Python-34B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0)  | [CodeLlama-13b-hf](llm-models/code_generation/codellama/codellama-13b) <br> [CodeLlama-13b-Instruct-hf](llm-models/code_generation/codellama/codellama-13b) <br> [CodeLlama-13b-Python-hf](llm-models/code_generation/codellama/codellama-13b) (Python optimized) <br> [CodeLlama-7b-hf](llm-models/code_generation/codellama/codellama-7b) <br> [CodeLlama-7b-Instruct-hf](llm-models/code_generation/codellama/codellama-7b) <br> [CodeLlama-7b-Python-hf](llm-models/code_generation/codellama/codellama-7b) (Python optimized) <br> <br> [WizardCoder-Python-13B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-13B-V1.0) <br> [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) | [WizardCoder-3B-V1.0](https://huggingface.co/WizardLM/WizardCoder-3B-V1.0) |
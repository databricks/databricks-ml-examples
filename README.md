# databricks-ml-examples

`databricks/databricks-ml-examples` is a repository to show machine learning examples on Databricks platforms.

It contains
- `llm-models/`: Example notebooks to use different **SOTA models on Databricks**.

## SOTA LLM Examples

Databricks works with thousands of customers to build generative AI applications. While you can use Databricks to work with any generative AI model, including commercial and research, the table below lists our current model recommendations* for popular use cases. Note that the table only lists open source models that are for free commercial use. 

| Use case  | Quality-optimized | Balanced | Speed-optimized |
| --------- | ----------------- | -------- | --------------- |
| Text generation following instructions  | [MPT-30B-Instruct](llm-models/mpt/mpt-30b) <br> [Falcon-40B-Instruct](llm-models/falcon/falcon-40b) | [MPT-7B-Instruct](llm-models/mpt/mpt-7b) <br> [Falcon-7B-Instruct](llm-models/falcon/falcon-7b) |  |
| Text embeddings (English only)   | [instructor-xl (1.3B)](https://huggingface.co/hkunlp/instructor-xl)  |  [e5-large-v2 (0.3B)](https://huggingface.co/intfloat/e5-large-v2) | [e5-base-v2 (0.1B)](https://huggingface.co/intfloat/e5-large-v2) <br> [all-mpnet-base-v2 (0.1B)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| Transcription (speech to text) | | [whisper-large-v2 (1.6B)](https://huggingface.co/openai/whisper-large-v2) <br> [whisper-medium](https://huggingface.co/openai/whisper-medium) (0.8B, English only) | |
| Image generation | | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | |
| Code generation  | | [StarCoderBase ](https://huggingface.co/bigcode/starcoderbase) (16B) <br> [StarCoder](https://huggingface.co/bigcode/starcoder) (16B, Python optimized) | [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) |

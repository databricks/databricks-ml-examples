# databricks-ml-examples

`databricks/databricks-ml-examples` is a repository to show machine learning examples on Databricks platforms.

It contains
- `llm-examples/`: Examples to use **Databricks ML Model Export**.

## SOTA LLM Examples

Databricks works with thousands of customers to build generative AI applications. While you can use Databricks to work with any generative AI model, including commercial and research, the table below lists our current model recommendations* for popular use cases. Note that the table only lists open source models that are for free commercial use. 

| Use case  | Quality-optimized | Balanced | Speed-optimized |
| --------- | ----------------- | -------- | --------------- |
| Text generation following instructions  | [MPT-30B-Instruct](llm-examples/mpt/mpt-30b) <br> Falcon-40B-Instruct | MPT-7B-Instruct <br> Falcon-7B-Instruct |  |
| Text embeddings (English only)   | instructor-xl (1.3B)  |  e5-large-v2 (0.3B) | e5-base-v2 (0.1B) <br> all-mpnet-base-v2 (0.1B) |
| Transcription (speech to text) | | whisper-large-v2 (1.6B) <br> whisper-medium (0.8B, English only) | |
| Image generation | | stable-diffusion-2-1 | |
| Code generation  | | StarCoderBase (16B) <br> StarCoder (16B, Python optimized) | replit-code-v1-3b |

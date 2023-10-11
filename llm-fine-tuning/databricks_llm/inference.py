from dataclasses import dataclass
from typing import Any, Tuple, List, Union, Callable, Optional, Dict
from itertools import islice

import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
)

import pandas as pd


@dataclass
class DocumentContext:
    db: Chroma

    def similarity_search(
        self,
        question: str,
        number_of_results: int = 5,
        filter: Optional[Dict[str, str]] = None,
    ):
        docs = self.db.similarity_search(question, k=number_of_results, filter=filter)
        return [doc.page_content for doc in docs]


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[List[str], str],
    temperature: float = 0.7,
    top_k: float = 0.92,
    max_new_tokens: int = 200,
) -> List[str]:
    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = prompt
    batch = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    batch = batch.to("cuda")

    with torch.no_grad():
        output_tokens_batch = model.generate(
            use_cache=True,
            input_ids=batch.input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,
            temperature=temperature,
            top_p=top_k,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("[/INST]"),
        )

    generated_responses = []
    for output_tokens, curr_prompt in zip(output_tokens_batch, prompts):
        prompt_length = len(tokenizer.encode(curr_prompt, return_tensors="pt")[0])
        output_tokens = np.trim_zeros(output_tokens.cpu().numpy())
        output_tokens = output_tokens[prompt_length:]
        generated_response = tokenizer.decode(output_tokens, skip_special_tokens=True)
        # colon_index = generated_response.find(":")
        # if colon_index > 0:
        #    generated_response = generated_response[colon_index + 1 :].strip()
        generated_responses.append(generated_response)

    return generated_responses


def batchify(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def generate_text_for_df(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    df: pd.DataFrame,
    src_col: str,
    target_col: str,
    gen_prompt_fn: Callable = None,
    post_process_fn: Callable = None,
    batch_size: int = 1,
    temperature: float = 0.7,
    top_k: float = 0.92,
    max_new_tokens: int = 200,
):
    src_col_values = []
    responses_list = []
    for batch in batchify(df.to_dict(orient="records"), batch_size):
        prompts = []
        for rec in batch:
            src_col_values.append(rec[src_col])
            if gen_prompt_fn:
                prompt = gen_prompt_fn(rec[src_col])
            else:
                prompt = rec[src_col]
            prompts.append(prompt)
        responses = generate_text(
            model,
            tokenizer,
            prompts,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )

        for response in responses:
            if post_process_fn:
                response = post_process_fn(response)
            responses_list.append(response)
    df[target_col] = responses_list
    return df


def load_vector_db(
    collection_name: str,
    persist_directory: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> DocumentContext:
    hf_embed = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(
        collection_name=collection_name,
        embedding_function=hf_embed,
        persist_directory=persist_directory,
    )
    return DocumentContext(db)


def generate_text_with_context(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str,
    question: str,
    doc_ctx: DocumentContext,
    doc_ctx_filter: Optional[Dict[str, str]],
    temperature: float = 0.7,
    top_k: float = 0.92,
    max_new_tokens: int = 200,
):
    docs = "\n".join(doc_ctx.similarity_search(question, filter=doc_ctx_filter))
    prompt = prompt_template.format(context=docs, question=question)
    return generate_text(model, tokenizer, prompt, temperature, top_k, max_new_tokens)

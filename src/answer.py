"""
Answer generation module for the RAG pipeline.
Executes the full pipeline: Query -> Retrieve -> Rerank -> Generate.

Usage:
    python answer.py <path_to_lancedb_table> "your question here"

Importable:
    from answer import load_llm, answer
    llm = load_llm()
    print(answer(retriever, reranker, llm, "your question"))
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieve import init_retriever, init_reranker, retrieve_candidates, rerank_candidates

LLM_ID = "Qwen/Qwen3-8B"

LLM_MAX_MEMORY = {0: "28GiB"}

LLM_MAX_NEW_TOKENS = 1024

RETRIEVE_K  = 100   # hybrid candidates fetched before reranking
RERANK_TOP  = 15    # chunks passed to the LLM as context

SYSTEM_PROMPT = (
    "You are a highly precise technical assistant.\n\n"
    "Rules:\n"
    "1. Use ONLY information explicitly stated in the provided context. Do not draw on outside knowledge.\n"
    "2. Report exact values, labels, tag numbers, and steps exactly as they appear.\n"
    "3. Never infer, extrapolate, or extend beyond what is written.\n"
    "4. If the context is insufficient to answer fully, state what is and is not available.\n"
    "5. Organize your final response in a clean, highly readable manner using ONLY plain text formatting (newlines and indentation). ABSOLUTELY DO NOT use Markdown formatting such as **, *, or #."
)


def load_llm():
    print(f"Loading {LLM_ID} ...")
    tok = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=LLM_MAX_MEMORY,
    ).eval()
    used = torch.cuda.memory_allocated(0) / 1e9
    print(f"  cuda:0 allocated: {used:.1f} GB")
    return mdl, tok


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] Section: {c['section']}"
        if c["chunk_type"] == "image":
            header += f"\n[Figure: {c['image_src']}] Description:"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    context = _format_context(chunks)
    user_content = (
        f"Use the following passages from the technical document or report to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def generate(llm, query: str, chunks: list[dict]) -> str:
    """
    Generate an answer from the LLM given a query and reranked context chunks.
    Thinking mode is disabled (enable_thinking=False) for direct answers.
    """
    model, tokenizer = llm
    messages = _build_messages(query, chunks)

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,   # prepends <think></think> to skip reasoning phase
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the input)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip any residual <think>...</think> block the model may emit
    if "<think>" in response:
        end = response.find("</think>")
        if end != -1:
            response = response[end + len("</think>"):].strip()

    return response


def answer(retriever_tuple, reranker_tuple, llm, query: str) -> str:
    """Run the full retrieve → rerank → generate pipeline for a query."""
    embedder, table = retriever_tuple
    candidates = retrieve_candidates(embedder, table, query, k=RETRIEVE_K)
    chunks     = rerank_candidates(reranker_tuple, query, candidates, top_n=RERANK_TOP)
    return generate(llm, query, chunks)


def main():
    if len(sys.argv) < 3:
        print("Usage: python answer.py <lancedb/table_name> \"your question here\"")
        sys.exit(1)

    path_with_table = sys.argv[1]
    query = " ".join(sys.argv[2:]).strip()

    db_dir, table_name = path_with_table.rsplit("/", 1)
    if table_name.endswith(".lance"):
        table_name = table_name[:-6]
    retriever_tuple = init_retriever(db_dir, table_name)
    reranker_tuple  = init_reranker()
    llm             = load_llm()

    print(f"\nQuestion: {query}\n")
    print("=" * 60)
    response = answer(retriever_tuple, reranker_tuple, llm, query)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
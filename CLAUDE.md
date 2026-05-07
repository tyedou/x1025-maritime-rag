# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda activate x1025          # Python 3.12, all deps managed via environment.yml
cp .env.example .env          # set HF_TOKEN and HF_HOME before first run
```

Model weights are downloaded from HuggingFace on first use and cached at `$HF_HOME` (defaults to `~/hf_cache`). Set `HF_TOKEN` in `.env` for gated models.

## Pipeline Stages & Commands

The pipeline processes a technical PDF into a queryable RAG system in four stages:

```bash
# 1. PDF → Markdown + image manifest
python src/convert_to_markdown.py data/my_manual.pdf --output-dir data/my_manual

# 2. Describe extracted images with InternVL2.5-8B (lmdeploy, GPU required)
python src/describe_images_lmdeploy.py data/my_manual

# 3. (Optional) Re-sync manifest descriptions into manual.md without re-running VLM
python src/update_md_from_manifest.py

# 4. Embed chunks and index into LanceDB (NV-Embed-v2, GPU required)
python src/chunk_and_embed.py data/my_manual

# 5a. One-shot Q&A
python src/answer.py lancedb/my_manual_lancedb "your question here"

# 5b. Interactive chat CLI
python src/chat.py        # runs from repo root; auto-discovers all tables in lancedb/
```

Retrieval-only (no LLM):
```bash
python src/retrieve.py lancedb/my_manual_lancedb "your question here"
```

## Architecture

### Data Flow

```
PDF
 └─ convert_to_markdown.py  → data/<name>/manual.md  (IMAGE_PLACEHOLDER comments)
                             → data/<name>/images/*.png
                             → data/<name>/image_manifest.json
 └─ describe_images_lmdeploy.py  → fills manifest descriptions + patches manual.md
 └─ chunk_and_embed.py      → lancedb/<name>_lancedb/  (LanceDB table)
 └─ answer.py / chat.py     → terminal output
```

### Key Design Decisions

**IMAGE_PLACEHOLDER format**: Images in `manual.md` are stored as HTML comments `<!-- IMAGE_PLACEHOLDER ... -->` with embedded metadata (index, source path, VLM description). `chunk_and_embed.py` reads these directly; `update_md_from_manifest.py` syncs the manifest back into these blocks without re-running the VLM.

**Chunking**: Text is split at markdown headings into ~1000-word chunks with 5-line overlap at flush boundaries. Images become their own chunks using their VLM descriptions as text. All chunks are prefixed with their section heading (`Section: <name>\n...`).

**LanceDB schema**: Each chunk stores `id`, `text`, `vector` (4096-d float32), `chunk_type` (`text`/`image`), `section`, `image_index`, and `image_src`. A full-text index is also built on `text` for hybrid search.

**Retrieval pipeline** (`retrieve.py`):
1. Dense vector search + BM25 full-text search via LanceDB hybrid query
2. RRF (Reciprocal Rank Fusion) merges the two result lists
3. Qwen3-Reranker-8B cross-encoder reranks top-K (default K=100) down to top-N (default N=15)

**GPU allocation**: Embedder (NV-Embed-v2) loads on `cuda:0`; reranker (Qwen3-Reranker-8B) loads on `cuda:1` if available, else `cuda:0`; LLM (Qwen3-30B-A3B) is sharded across all 4 GPUs using `max_memory` in `answer.py:LLM_MAX_MEMORY`. Adjust this dict for your hardware.

**NV-Embed-v2 patch**: Both `chunk_and_embed.py` and `retrieve.py` call `patch_nvembed()` at import time. This patches the downloaded model source in `$HF_HOME` to fix compatibility with `transformers >= 4.46` (position embeddings and cache handling). The patch is idempotent.

### Module Responsibilities

| File | Role |
|------|------|
| `src/convert_to_markdown.py` | PDF → Markdown via Docling; extracts and de-noises text, replaces image refs with IMAGE_PLACEHOLDER blocks |
| `src/describe_images_lmdeploy.py` | Runs InternVL2.5-8B via lmdeploy to fill image descriptions; skips already-described and tiny (<5 KB) images |
| `src/update_md_from_manifest.py` | Syncs manifest → manual.md without re-running VLM; filters refusals and error tags |
| `src/chunk_and_embed.py` | Parses manual.md into text + image chunks, embeds with NV-Embed-v2, writes to LanceDB |
| `src/retrieve.py` | Hybrid retrieval + Qwen3-Reranker-8B; importable as library or runnable standalone |
| `src/answer.py` | Full RAG pipeline: retrieve → rerank → generate with Qwen3-30B-A3B (thinking disabled) |
| `src/chat.py` | Interactive REPL over an existing LanceDB directory; supports switching manuals mid-session |

### Important Constants (answer.py)

- `RETRIEVE_K = 100` — candidates fetched before reranking
- `RERANK_TOP = 15` — chunks passed to the LLM
- `LLM_MAX_NEW_TOKENS = 1024`
- `LLM_MAX_MEMORY` — per-GPU VRAM budget; tune for your machine

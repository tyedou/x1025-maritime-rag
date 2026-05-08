"""
Hybrid retrieval and cross-encoder reranking module for the RAG pipeline.

Usage:
    python retrieve.py <path_to_lancedb_table> "your query here"

Importable:
    from retrieve import init_retriever, init_reranker, retrieve_candidates, rerank_candidates
    embedder, table = init_retriever("lancedb", "my_table")
    reranker = init_reranker()
    candidates = retrieve_candidates(embedder, table, "query", k=10)
    results = rerank_candidates(reranker, "query", candidates, top_n=5)
"""
import argparse
import glob
import os
import sys
import warnings
from pathlib import Path
import time

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
if not os.environ.get("HF_HOME"): os.environ["HF_HOME"] = str(Path.home() / "hf_cache")
warnings.filterwarnings("ignore")

import lancedb
import torch
import torch.nn.functional as F
from lancedb.rerankers import RRFReranker
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

def patch_nvembed():
    for p in glob.glob(os.path.join(os.environ["HF_HOME"], "modules/transformers_modules/nvidia/NV-Embed-v2/*/modeling_nvembed.py")):
        with open(p, "r+") as f:
            t = f.read()
            if "position_embeddings = self.rotary_emb" in t:
                continue  # already patched
            t = t.replace("'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),", "'input_ids': batch_dict.get('input_ids').clone().detach().long(),")
            t = t.replace("        use_cache = use_cache if use_cache is not None else self.config.use_cache", "        use_cache = False")
            t = t.replace("        hidden_states = inputs_embeds\n\n        # decoder layers", "        hidden_states = inputs_embeds\n        position_embeddings = self.rotary_emb(hidden_states, position_ids)\n\n        # decoder layers")
            t = t.replace("                    output_attentions,\n                    use_cache,\n                )\n            else:", "                    output_attentions,\n                    use_cache,\n                    None,\n                    position_embeddings,\n                )\n            else:")
            t = t.replace("                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                )", "                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                    position_embeddings=position_embeddings,\n                )")
            f.seek(0); f.write(t); f.truncate()

def init_retriever(db_dir, table_name):
    from transformers.dynamic_module_utils import get_cached_module_file
    for fname in ["modeling_nvembed.py", "configuration_nvembed.py"]:
        try:
            get_cached_module_file("nvidia/NV-Embed-v2", fname)
        except Exception:
            pass
    patch_nvembed()
    embedder = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True).half().cuda().eval()
    return embedder, lancedb.connect(str(db_dir)).open_table(table_name)

def retrieve_candidates(embedder, table, query, k=10):
    t0 = time.perf_counter()
    with torch.no_grad():
        v = embedder.encode([query], instruction="Instruct: Given a technical question about a technical manual or report, retrieve the most relevant passages that answer the question.\nQuery: ", max_length=2048)
    v = F.normalize(v.float(), p=2, dim=1).cpu().tolist()[0] if isinstance(v, torch.Tensor) else v.tolist()[0]
    results = table.search(query_type="hybrid").vector(v).text(query).metric("cosine").rerank(RRFReranker(return_score="all")).limit(k).to_list()
    print(f"[RETRIEVAL] {time.perf_counter() - t0:.3f}s")
    return results

def init_reranker():
    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, device_map="auto").eval()
    return m, t, t.convert_tokens_to_ids("yes"), t.convert_tokens_to_ids("no")

def rerank_candidates(reranker, query, cands, top_n=5):
    t0 = time.perf_counter()
    m, t, y_id, n_id = reranker
    full_texts = [
        f'<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a technical question about a technical manual or report, assess whether the document contains relevant information to answer the question.\n<Query>: {query}\n<Document>: {c["text"]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        for c in cands
    ]
    scrs = []
    batch_size = 1
    for i in range(0, len(full_texts), batch_size):
        pad = t(full_texts[i:i+batch_size], padding=True, truncation=True, max_length=2048, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            lgts = m(**{k: v.to(m.device) for k, v in pad.items()}).logits[:, -1, :]
            batch_scrs = torch.nn.functional.log_softmax(torch.stack([lgts[:, n_id], lgts[:, y_id]], dim=1), dim=1)[:, 1].exp().cpu().tolist()
            scrs.extend(batch_scrs)
    for c, s in zip(cands, scrs): c["_rerank_score"] = s
    print(f"[RERANKING] {time.perf_counter() - t0:.3f}s")
    return sorted(cands, key=lambda x: x["_rerank_score"], reverse=True)[:top_n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_with_table", type=str, help="Path to LanceDB database and table (e.g. lancedb/my_table)")
    parser.add_argument("query", nargs="+", help="Query string")
    args = parser.parse_args()
    
    db_dir, table_name = args.path_with_table.rsplit("/", 1)
        
    if table_name.endswith(".lance"):
        table_name = table_name[:-6]
    
    q = " ".join(args.query)
    embedder, tbl = init_retriever(db_dir, table_name)
    reranker = init_reranker()
    
    for i, r in enumerate(rerank_candidates(reranker, q, retrieve_candidates(embedder, tbl, q)), 1):
        rrf = r.get('_relevance_score') or 0.0
        dist = r.get('_distance') or 0.0
        bm25 = r.get('_score') or 0.0
        print(f"\n#{i} Rerank:{r.get('_rerank_score', 0):.3f} | RRF:{rrf:.3f} | Dist:{dist:.3f} | BM25:{bm25:.3f}\nType: {r.get('chunk_type', '')} | Section: {r.get('section', '')}\nText: {r['text'][:400].replace(chr(10), ' ')}...")

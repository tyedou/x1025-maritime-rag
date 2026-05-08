"""
Document ingestion and indexing module for the RAG pipeline.
Performs Macro-Chunking and vector embedding, storing results in LanceDB.

Usage:
    python chunk_and_embed.py <path_to_output_dir>
"""

import argparse
import glob
import json
import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
if not os.environ.get("HF_HOME"): os.environ["HF_HOME"] = str(Path.home() / "hf_cache")
os.environ.setdefault("LANCE_LOG", "ERROR")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
warnings.filterwarnings("ignore", message=".*sdp_kernel.*")

import lancedb
import pyarrow as pa
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "nvidia/NV-Embed-v2"
EMBED_DIM = 4096
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    pa.field("chunk_type", pa.string()),
    pa.field("section", pa.string()),
    pa.field("image_index", pa.int32()),
    pa.field("image_src", pa.string()),
])

def patch_nvembed():
    # Patch NV-Embed-v2 for compatibility with transformers >= 4.46.
    # Must run AFTER code files are downloaded to transformers_modules cache.
    pattern = os.path.join(os.environ["HF_HOME"], "modules/transformers_modules/nvidia/NV-Embed-v2/*/modeling_nvembed.py")
    for p in glob.glob(pattern):
        with open(p, "r+") as f:
            t = f.read()
            if "position_embeddings = self.rotary_emb" in t:
                continue  # already patched
            t = t.replace(
                "'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),",
                "'input_ids': batch_dict.get('input_ids').clone().detach().long(),"
            )
            t = t.replace("        use_cache = use_cache if use_cache is not None else self.config.use_cache", "        use_cache = False")
            t = t.replace("        hidden_states = inputs_embeds\n\n        # decoder layers", "        hidden_states = inputs_embeds\n        position_embeddings = self.rotary_emb(hidden_states, position_ids)\n\n        # decoder layers")
            t = t.replace("                    output_attentions,\n                    use_cache,\n                )\n            else:", "                    output_attentions,\n                    use_cache,\n                    None,\n                    position_embeddings,\n                )\n            else:")
            t = t.replace("                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                )", "                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                    position_embeddings=position_embeddings,\n                )")
            f.seek(0); f.write(t); f.truncate()

def load_embed_model():
    # Pre-download model code files to transformers_modules before patching,
    # so patch_nvembed() always runs after the latest files are on disk.
    from transformers.dynamic_module_utils import get_cached_module_file
    for fname in ["modeling_nvembed.py", "configuration_nvembed.py"]:
        try:
            get_cached_module_file(MODEL_ID, fname)
        except Exception:
            pass
    patch_nvembed()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).half().cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer

def build_section_map(md: str):
    boundaries = [(m.start(), m.group(1).strip()) for m in re.finditer(r'^#{1,6}\s+(.+)$', md, re.M)]
    def lookup(pos: int) -> str:
        res = "Introduction"
        for start, head in boundaries:
            if start <= pos: res = head
            else: break
        return res
    return lookup

def parse_chunks(src_dir: Path) -> list:
    md = (src_dir / "manual.md").read_text("utf-8")
    manifest = {e["index"]: e for e in json.loads((src_dir / "image_manifest.json").read_text("utf-8"))}
    sec_map = build_section_map(md)
    chunks = []

    for m in re.finditer(r'<!-- IMAGE_PLACEHOLDER.*?-->', md, re.S):
        idx_m = re.search(r'^index:\s*(\d+)', m.group(0), re.M)
        if not idx_m: continue
        idx = int(idx_m.group(1))
        entry = manifest.get(idx, {})
        desc = entry.get("description", "")
        src = entry.get("source", "")
        img_path = src_dir / src
        
        if desc and not desc.startswith("[") and img_path.exists() and img_path.stat().st_size >= 1000:
            chunks.append({
                "id": f"img_{idx:04d}", "text": desc.strip(), "chunk_type": "image",
                "section": sec_map(m.start()), "image_index": idx, "image_src": src
            })

    clean_md = re.sub(r'<!-- IMAGE_PLACEHOLDER.*?-->', '', md, flags=re.S)
    chunk_id = 0
    sec = "Introduction"
    buf, buf_w = [], 0

    def flush(s):
        nonlocal chunk_id
        text = "\n".join(buf).strip()
        if text:
            full_text = f"Section: {s}\n{text}"
            chunks.append({"id": f"txt_{chunk_id:04d}", "text": full_text, "chunk_type": "text", "section": s, "image_index": -1, "image_src": ""})
            chunk_id += 1

    for line in clean_md.splitlines():
        line = line.strip()
        if re.match(r'^#{1,6}\s+', line):
            flush(sec)
            buf, buf_w = [], 0
            sec = re.sub(r'^#{1,6}\s+', '', line).strip()
            continue
        if not line: continue
        words = len(line.split())
        if buf_w + words > 1000 and buf:
            flush(sec)
            buf = buf[-5:]
            buf_w = sum(len(x.split()) for x in buf)
        buf.append(line)
        buf_w += words
    flush(sec)
    
    return chunks

def embed_chunks(model, chunks: list) -> list:
    texts = [c["text"] for c in chunks]
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            vecs = model.encode(batch, instruction="", max_length=2048)
            if isinstance(vecs, torch.Tensor):
                vecs = F.normalize(vecs.float(), p=2, dim=1).cpu().tolist()
            else:
                vecs = vecs.tolist()
            for j, v in enumerate(vecs):
                chunks[i+j]["vector"] = v
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    
    src = args.output_dir.resolve()
    db_path = src.parent / "lancedb"
    table_name = f"{src.name}_lancedb"
    
    model, _ = load_embed_model()
    chunks = embed_chunks(model, parse_chunks(src))
    
    db_path.mkdir(parents=True, exist_ok=True)
    tbl = lancedb.connect(str(db_path)).create_table(table_name, data=chunks, schema=SCHEMA, mode="overwrite")
    tbl.create_fts_index("text", replace=True)
    print(f"Indexed {len(chunks)} chunks to {db_path.name}/{table_name}.")

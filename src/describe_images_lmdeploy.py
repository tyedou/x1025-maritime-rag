"""
Vision-Language Processing module for the RAG pipeline.
Uses InternVL2.5-38B-AWQ to dynamically analyze and describe technical diagrams,
and updates the Markdown file with these descriptions.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_PATH = MODEL_ID
DEFAULT_OUT = Path(__file__).parent.parent.resolve() / "data"
PROMPT = "You are analyzing a technical diagram from a ship's deck operating manual. Describe everything visible: system or subsystem shown, all equipment labels and instrument tags (e.g. PI-101, LT-202), valve types and states, pipe connections and flow directions, color codes for fluid types, and any text legible in the image. Be precise and comprehensive."
REFUSALS = {"unable to analyze", "cannot analyze", "i cannot", "i am unable", "cannot provide", "i'm unable", "not able to", "i don't have the ability"}

def get_vision_pipeline():
    return pipeline(MODEL_PATH, backend_config=TurbomindEngineConfig(tp=1, session_len=8192, cache_max_entry_count=0.3))

def generate_image_descriptions(out_dir: Path) -> list:
    manifest_file = out_dir / "image_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    vision_pipe = None
    
    for item in manifest:
        if item.get("description"):
            continue
            
        img = out_dir / item["source"]
        if not img.exists():
            item["description"] = "[IMAGE NOT FOUND]"
            print(f"Skipped missing file: {img.name}")
        elif img.stat().st_size < 5000:
            item["description"] = "[DECORATIVE ELEMENT - no meaningful technical content]"
            item["model"] = "skipped"
            print(f"Skipped tiny image: {img.name}")
        else:
            if not vision_pipe:
                vision_pipe = get_vision_pipeline()
            try:
                desc = vision_pipe((PROMPT, load_image(str(img)))).text.strip()
                if not desc or any(r in desc.lower() for r in REFUSALS):
                    raise ValueError("Model refused or returned empty")
                item["description"] = desc.replace("-->", "—>").replace("<!--", "—")
                item["model"] = MODEL_ID
                print(f"Successfully processed: {img.name}")
            except Exception as err:
                item["description"] = f"[ERROR: {err}]"
                print(f"Error processing {img.name}: {err}")
                
        manifest_file.write_text(json.dumps(manifest, indent=2))
        
    return manifest

def embed_descriptions_in_markdown(manifest: list, out_dir: Path):
    md_file = out_dir / "manual.md"
    text = md_file.read_text("utf-8")
    descriptions = {str(m["index"]): m for m in manifest if m.get("description")}

    def inject_block(match: re.Match) -> str:
        block = match.group(0)
        idx_match = re.search(r'^index:\s*(\d+)', block, re.M)
        if not idx_match or idx_match.group(1) not in descriptions:
            return block
        
        info = descriptions[idx_match.group(1)]
        block = re.sub(r'^model:.*$', f'model: {info.get("model", MODEL_ID)}', block, flags=re.M)
        return re.sub(r'^description:.*$', f'description: {info["description"]}', block, flags=re.M)

    md_file.write_text(re.sub(r'<!-- IMAGE_PLACEHOLDER.*?-->', inject_block, text, flags=re.DOTALL), "utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe images and update markdown.")
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUT, help="Directory containing manual.md and images/")
    args = parser.parse_args()
    
    updated_manifest = generate_image_descriptions(args.output_dir)
    embed_descriptions_in_markdown(updated_manifest, args.output_dir)

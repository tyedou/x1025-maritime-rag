"""
PDF to Markdown conversion module.
Uses Docling to extract pristine native text and diagrams from technical PDFs.
"""
import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _patch_layout_predictor():
    # docling-ibm-models passes torch.device to device_map= instead of a string,
    # causing transformers to silently keep weights on CPU while inputs go to CUDA.
    import docling_ibm_models.layoutmodel.layout_predictor as lp
    original_init = lp.LayoutPredictor.__init__
    def patched_init(self, artifact_path, device="cpu", num_threads=4):
        original_init(self, artifact_path, device=device, num_threads=num_threads)
        if "cuda" in str(self._device):
            self._model = self._model.to(self._device)
    lp.LayoutPredictor.__init__ = patched_init

_patch_layout_predictor()

_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_NAV_EXACT = frozenset({"PREVIOUS PAGE", "NEXT PAGE", "EXIT PAGE", "CONTENTS LIST", "PREVIOUS", "NEXT", "CONTENTS", "EXIT"})
_NOISE_RE = re.compile(r'^(\d+|Page\s+\d+(\s+of\s+\d+)?|(Issue|Rev(?:ision)?|Version|Amendment|Change)\s*[:\-]?\s*[\d\w]+|(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})$', re.I)
_NAV_HEADING_RE = re.compile(r'^#{1,6}\s+(PREVIOUS\s+PAGE|NEXT\s+PAGE|CONTENTS(\s+LIST)?|EXIT(\s+PAGE)?)\s*$', re.I)
_HEADING_NAV_PREFIX_RE = re.compile(r'^(#{1,6}\s+)(PREVIOUS\s+PAGE|NEXT\s+PAGE|CONTENTS\s+LIST|CONTENTS)\s+', re.I)
_INLINE_NAV_RE = re.compile(r'\b(PREVIOUS\s+PAGE|NEXT\s+PAGE|EXIT\s+PAGE|CONTENTS\s+LIST|CONTENTS|EXIT\s+PAGE)\b\s*', re.I)
_LONE_CHAR_RE = re.compile(r'^[A-Za-z]\s*$')

def process_pdf(pdf_path: Path, output_dir: Path) -> None:
    pdf_path = pdf_path.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.generate_picture_images = True
    opts.generate_page_images = False
    opts.images_scale = 2.0
    
    doc = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    ).convert(pdf_path).document

    extract_dir = output_dir / "_tmp"
    extract_dir.mkdir(exist_ok=True)
    doc_file = extract_dir / "raw.md"
    
    doc.save_as_markdown(doc_file, image_mode=ImageRefMode.REFERENCED, artifacts_dir=extract_dir)
    text = doc_file.read_text("utf-8")
    
    doc_images = next((p for p in extract_dir.iterdir() if p.is_dir()), None)
    if doc_images:
        text = text.replace(f"{doc_images.name}/", "images/")
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for img_file in doc_images.rglob("*"):
            if img_file.is_file():
                shutil.move(str(img_file), str(images_dir / img_file.name))
    shutil.rmtree(extract_dir, ignore_errors=True)

    manifest = []
    def record_image(m: re.Match) -> str:
        idx = len(manifest) + 1
        fn = Path(m.group(2)).name
        cap = m.group(1).strip()
        manifest.append({"index": idx, "source": f"images/{fn}", "caption": cap, "description": None})
        return f"<!-- IMAGE_PLACEHOLDER\nindex: {idx}\nsource: images/{fn}\ncaption: {cap}\nmodel: TO_BE_ASSIGNED\ndescription: TO_BE_FILLED_BY_VISION_MODEL\n-->"
    
    text = _IMAGE_RE.sub(record_image, text)
    
    lines = []
    blanks = 0
    for line in text.splitlines():
        line = _HEADING_NAV_PREFIX_RE.sub(r'\1', line)
        stripped = line.strip()
        
        if _NAV_HEADING_RE.match(line) or stripped in _NAV_EXACT or (stripped and _NOISE_RE.match(stripped)) or _LONE_CHAR_RE.match(stripped):
            continue
            
        if stripped and not stripped.startswith("<!--"):
            line = _INLINE_NAV_RE.sub("", line).rstrip()
            stripped = line.strip()
            
        if stripped in {"", "-", "\u2013", "\u2014"}:
            stripped = ""
            
        if not stripped:
            blanks += 1
            if blanks > 2:
                continue
        else:
            blanks = 0
            
        lines.append(line if stripped else "")

    (output_dir / "manual.md").write_text("\n".join(lines), "utf-8")
    (output_dir / "image_manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    logging.info(f"Processed {pdf_path.name} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    
    try:
        process_pdf(args.pdf, args.output_dir or args.pdf.parent / args.pdf.stem)
    except Exception as e:
        sys.exit(f"Failed: {e}")

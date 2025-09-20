"""
text_extraction.py

Purpose
-------
Extract page text and positions from a PDF. Prefer PyMuPDF4LLM for speed and
layout. If that fails or yields nothing, fall back to OCR, write the OCR text
back into a copy of the PDF as invisible text, then run PyMuPDF4LLM again.

Flow
----
1) Try PyMuPDF4LLM:
   - run to_markdown(..., page_chunks=True, extract_words=True)
   - build:
       a) concatenated Markdown across pages
       b) a compact per-page structure with word boxes and grouped line boxes
2) If PyMuPDF4LLM errors or returns no pages:
   a) Render pages to images
   b) Run PP-StructureV3 (GPU if device=="gpu") to get line boxes + texts
   c) Insert those lines as invisible text into a copy of the PDF
   d) Re-run PyMuPDF4LLM on the overlaid PDF to produce Markdown + structure

Outputs (always)
----------------
- out_dir/pm4l.md             : concatenated Markdown for all pages
- out_dir/text_structure.json : {"pageNNN": {"words":[...], "lines":[...]}, ...}
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import fitz                  # PyMuPDF
import cv2
import numpy as np
import pymupdf4llm
import paddle
from paddleocr import PPStructureV3


# ---------------- Tunables ----------------
# Render quality and size caps for the OCR fallback path.
RENDER_DPI = 300
MAX_LONG_SIDE = 5000
JPG_QUALITY = 95

# OCR controls for PP-StructureV3 fallback
DET_LIMIT_SIDE_LEN = 1000
DET_BOX_THRESH     = 0.60
REC_SCORE_THRESH   = 0.80
REC_BATCH          = 32


# ---------------- Small utilities ----------------

def _natkey(s: str):
    # Natural sort key: splits digits so "page10" sorts after "page9".
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _imread_color_fast(path: str):
    # Robust image read that tolerates Windows paths and non-ASCII filenames.
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    # Create parent directory if needed and write a JPEG with given quality.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


# ---------------- PDF rendering (for OCR fallback) ----------------

def _render_pages(pdf_path: str, out_dir: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Render PDF pages to JPEGs.

    Returns
    -------
    image_paths : List[str]
        Paths in page order: out_dir/page001.jpg, ...
    sizes : List[(int, int)]
        (W, H) in pixels for each rendered page after optional downscale.
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths: List[str] = []
    sizes: List[Tuple[int, int]] = []

    for i, page in enumerate(doc, start=1):
        pk = f"page{i:03d}"
        out_path = os.path.join(out_dir, f"{pk}.jpg")

        pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H0, W0 = bgr.shape[:2]
        longest = max(W0, H0)
        if longest > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / float(longest)
            bgr = cv2.resize(bgr, (int(round(W0 * scale)), int(round(H0 * scale))), interpolation=cv2.INTER_AREA)

        _imwrite_jpg(out_path, bgr)
        paths.append(out_path)
        sizes.append((bgr.shape[1], bgr.shape[0]))  # (W, H)

    return paths, sizes


# ---------------- PyMuPDF4LLM helpers ----------------

def _pm4l_build_markdown_and_structure(pages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Build:
      1) a single Markdown string by concatenating page["text"]
      2) a compact per-page structure of words and line groups.

    Output structure
    ----------------
    For page N (1-based) the key "pageNNN" holds:
      {
        "page": N,
        "words": [{"text", "bbox", "block", "line", "index"}],
        "lines": [{"text", "bbox", "block", "line"}]
      }
    """
    # Markdown: concatenate each page's "text"
    md_parts: List[str] = []
    for p in pages:
        t = p.get("text") or ""
        md_parts.append(t)
    md_text = "\n\n".join(md_parts)

    # Structure: words + grouped lines
    compiled: Dict[str, Any] = {}
    for p in pages:
        meta = p.get("metadata", {})
        pno1 = int(meta.get("page_number", 0)) or 0
        key = f"page{pno1:03d}"

        words_raw = p.get("words") or []
        words_out: List[Dict[str, Any]] = []
        by_line: Dict[Tuple[int, int], List[Tuple[float, float, float, float, str]]] = {}

        for w in words_raw:
            # Expected tuple: (x0,y0,x1,y1, "word", block, line, index)
            if not isinstance(w, (list, tuple)) or len(w) < 5:
                continue
            x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
            text = str(w[4])
            bno = int(w[5]) if len(w) > 5 else -1
            lno = int(w[6]) if len(w) > 6 else -1
            wno = int(w[7]) if len(w) > 7 else -1

            words_out.append({
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "block": bno,
                "line": lno,
                "index": wno,
            })
            by_line.setdefault((bno, lno), []).append((x0, y0, x1, y1, text))

        lines_out: List[Dict[str, Any]] = []
        # Stable order: by block, then by line number
        for (bno, lno) in sorted(by_line.keys()):
            items = by_line[(bno, lno)]
            xs0 = [it[0] for it in items]; ys0 = [it[1] for it in items]
            xs1 = [it[2] for it in items]; ys1 = [it[3] for it in items]
            line_bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
            # Left-to-right within roughly top-to-bottom sorting
            line_text = " ".join(it[4] for it in sorted(items, key=lambda t: (t[1], t[0])))
            lines_out.append({"text": line_text, "bbox": line_bbox, "block": bno, "line": lno})

        compiled[key] = {"page": pno1, "words": words_out, "lines": lines_out}

    return md_text, compiled


# ---------------- PP-Structure OCR fallback ----------------

def _ppstruct_batch_ocr(image_paths: List[str], device: str,
                        det_limit_side_len: int, det_box_thresh: float,
                        rec_score_thresh: float, rec_batch: int) -> List[Dict[str, Any]]:
    """
    Run PP-StructureV3 on a list of page images and return the JSON outputs.

    Behavior
    --------
    - Selects GPU/CPU via paddle.set_device.
    - Configures detection/recognition thresholds and batch size.
    - Calls pp.predict(image_paths).
    - Writes each result with res.save_to_json(...) to a temp folder.
    - Reads those JSON files back and returns them as a list aligned to image_paths.
    """
    # Select device for Paddle runtime
    paddle.set_device("gpu" if device == "gpu" else "cpu")

    # Instantiate PP-StructureV3 for OCR only; region/table/formula/chart disabled
    pp = PPStructureV3(
        device=("gpu" if device == "gpu" else "cpu"),
        text_det_limit_side_len=int(det_limit_side_len),
        text_det_limit_type="max",
        text_det_box_thresh=float(det_box_thresh),
        text_rec_score_thresh=float(rec_score_thresh),
        text_recognition_batch_size=int(rec_batch),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=None,
    )

    results = pp.predict(image_paths)

    if not isinstance(results, (list, tuple)):
        results = [results]

    per_page_json: List[Dict[str, Any]] = []
    tmp_dir = Path(os.path.dirname(image_paths[0]) or ".") / "_ppstruct_tmpjson"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for img_path, res in zip(image_paths, results):
        page_key = Path(img_path).stem
        out_json = tmp_dir / f"{page_key}.json"
        res.save_to_json(str(out_json))
        with out_json.open("r", encoding="utf-8") as f:
            per_page_json.append(json.load(f))

    return per_page_json


def _insert_ocr_as_invisible_text(src_pdf: str,
                                  image_paths: List[str],
                                  image_sizes: List[Tuple[int, int]],
                                  ppjson_pages: List[Dict[str, Any]]) -> str:
    """
    Create a new PDF with invisible text added at OCR line boxes.
    """
    doc = fitz.open(src_pdf)

    for i, page in enumerate(doc):
        # Pixel → point scale factors per page
        Wpx, Hpx = image_sizes[i]
        Wpt, Hpt = page.rect.width, page.rect.height
        sx = Wpt / float(Wpx if Wpx > 0 else 1)
        sy = Hpt / float(Hpx if Hpx > 0 else 1)

        d = ppjson_pages[i]

        # Overall OCR lines (PP-StructureV3 consolidated results)
        ocr = d.get("overall_ocr_res") or {}
        texts = ocr.get("rec_texts") or []
        boxes  = ocr.get("rec_boxes") or []

        for txt, bb in zip(texts, boxes):
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue

            x0, y0, x1, y1 = bb
            # Handle potential (x, y, w, h)
            if x1 <= x0 or y1 <= y0:
                x, y, w, h = x0, y0, x1, y1
                x0, y0, x1, y1 = x, y, x + w, y + h

            # Convert to PDF points
            rx0, ry0 = float(x0) * sx, float(y0) * sy
            rx1, ry1 = float(x1) * sx, float(y1) * sy
            rect = fitz.Rect(rx0, ry0, rx1, ry1)

            # Insert invisible text; small fontsize so it stays inside the box
            page.insert_textbox(
                rect,
                txt,
                fontname="helv",
                fontsize=9,
                render_mode=3,         # invisible text
                overlay=True,
            )

    out_pdf = str(Path(src_pdf).with_name("ocr_overlay.pdf"))
    doc.save(out_pdf)
    doc.close()
    return out_pdf


# ---------------- Main entry ----------------

def run_text_extraction(
    pdf_path: str,
    out_dir: str,
    device: str = "gpu",
    *,
    det_limit_side_len: int = DET_LIMIT_SIDE_LEN,   # OCR fallback
    det_box_thresh: float = DET_BOX_THRESH,
    rec_score_thresh: float = REC_SCORE_THRESH,
    rec_batch: int = REC_BATCH,
) -> Dict[str, str]:
    """
    Run text extraction with a PM4L-first strategy and OCR fallback.

    Parameters
    ----------
    pdf_path : str
        Source PDF.
    out_dir : str
        Output directory (created if missing).
    device : {"gpu","cpu"}
        Paddle device for OCR fallback.

    Returns
    -------
    dict with:
      - "markdown_path": str
      - "text_structure_path": str
      - "source": "pm4l" or "ppstruct+pm4l"
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    md_path = Path(out_dir) / "pm4l.md"
    struct_path = Path(out_dir) / "text_structure.json"

    # 1) Preferred path: PyMuPDF4LLM directly on the PDF
    try:
        pages = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            extract_words=True,
            show_progress=True,
            table_strategy="lines_strict",
        )
        if isinstance(pages, list) and len(pages) > 0:
            md_text, structure = _pm4l_build_markdown_and_structure(pages)
            if md_text.strip():
                md_path.write_bytes(md_text.encode("utf-8"))
                struct_path.write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
                return {
                    "markdown_path": str(md_path),
                    "text_structure_path": str(struct_path),
                    "source": "pm4l",
                }
    except Exception as e:
        # Fall through to OCR path if PM4L errors
        print(f"[PM4L] primary extraction failed, fallback to OCR. Reason: {e}")

    # 2) Fallback: OCR → overlay invisible text → re-run PM4L
    pages_dir = os.path.join(out_dir, "page_images")
    image_paths, image_sizes = _render_pages(pdf_path, pages_dir)

    # 2a) OCR pages
    pp_pages = _ppstruct_batch_ocr(
        image_paths=image_paths,
        device=device,
        det_limit_side_len=det_limit_side_len,
        det_box_thresh=det_box_thresh,
        rec_score_thresh=rec_score_thresh,
        rec_batch=rec_batch,
    )

    # 2b) Write OCR lines into a copy of the PDF as invisible text
    overlay_pdf = _insert_ocr_as_invisible_text(
        src_pdf=pdf_path,
        image_paths=image_paths,
        image_sizes=image_sizes,
        ppjson_pages=pp_pages,
    )

    # 2c) Run PM4L on the overlaid PDF
    pages2 = pymupdf4llm.to_markdown(
        overlay_pdf,
        page_chunks=True,
        extract_words=True,
        show_progress=True,
        table_strategy="lines_strict",
    )
    if not isinstance(pages2, list) or len(pages2) == 0:
        raise RuntimeError("Fallback pm4l returned no pages after OCR overlay.")

    md_text2, structure2 = _pm4l_build_markdown_and_structure(pages2)
    md_path.write_bytes(md_text2.encode("utf-8"))
    struct_path.write_text(json.dumps(structure2, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "markdown_path": str(md_path),
        "text_structure_path": str(struct_path),
        "source": "ppstruct+pm4l",
    }

# text_extraction.py
#
# Two-pass text pipeline:
#   Pass-1 (plain PyMuPDF): write pixel-aligned text boxes -> text_boxes_pages/pageNNN.json
#     - If total boxes > 0: proceed to Pass-2
#     - If total boxes == 0: run OCR -> write entries from OCR -> overlay -> proceed to Pass-2
#   Pass-2 (PyMuPDF4LLM): write pm4l.md + text_structure.json + pm4l_page_index.json
#
# Idempotent:
#   - Page images rendered once and reused
#   - If final artifacts and per-page entries exist, it returns cached bundle
#
# Outputs:
#   <out_dir>/page_images/pageNNN.jpg
#   <out_dir>/text_boxes_pages/pageNNN.json
#   <out_dir>/pm4l.md
#   <out_dir>/text_structure.json
#   <out_dir>/pm4l_page_index.json
#
# Return bundle:
# {
#   "markdown_path": str,
#   "text_structure_path": str,
#   "page_index_path": str,
#   "pages_dir": str,
#   "image_paths": List[str],
#   "source": "pm4l" | "ppstruct+pm4l" | "cached",
#   "pages_count": int,
#   "words_total": int,
#   "structure": Dict[str, Any],
#   "page_index": List[Dict[str,int]],
# }

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pymupdf4llm

import paddle
from paddleocr import PPStructureV3


# ---------------- Tunables ----------------

RENDER_DPI = 300
MAX_LONG_SIDE = 5000
JPG_QUALITY = 95

# OCR controls for PP-StructureV3 fallback
DET_LIMIT_SIDE_LEN = 1000
DET_BOX_THRESH = 0.60
REC_SCORE_THRESH = 0.80
REC_BATCH = 32


# ---------------- Utils ----------------

def _imread_color_fast(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def _render_pages(pdf_path: str, out_dir: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Render PDF pages to JPEGs. Reuse existing files.

    Returns:
        image_paths: paths in page order
        sizes: list of (Wpx, Hpx)
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    paths: List[str] = []
    sizes: List[Tuple[int, int]] = []

    for i, page in enumerate(doc, start=1):
        pk = f"page{i:03d}"
        out_path = os.path.join(out_dir, f"{pk}.jpg")

        if os.path.exists(out_path):
            img = _imread_color_fast(out_path)
            if img is not None:
                paths.append(out_path)
                sizes.append((img.shape[1], img.shape[0]))
                continue

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
        sizes.append((bgr.shape[1], bgr.shape[0]))

    doc.close()
    return paths, sizes


# ---------------- PM4L: markdown + structure + page index ----------------

def _pm4l_build_markdown_and_structure(pages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], List[Dict[str, int]]]:
    md_parts: List[str] = []
    page_index: List[Dict[str, int]] = []
    compiled: Dict[str, Any] = {}

    pos = 0
    # Use enumerate index as reliable fallback for page number
    for i, p in enumerate(pages, start=1):
        meta = p.get("metadata", {})
        pno = int(meta.get("page_number") or 0)
        if pno <= 0:
            pno = i

        header = f"## Page {pno:03d}\n<a id=\"page-{pno:03d}\"></a>\n\n"
        body = (p.get("text") or "") + "\n"
        block = header + body

        start = pos
        md_parts.append(block)
        pos += len(block)
        end = pos
        page_index.append({"page": pno, "start": start, "end": end})

    md_text = "".join(md_parts)

    for i, p in enumerate(pages, start=1):
        meta = p.get("metadata", {})
        pno = int(meta.get("page_number") or 0)
        if pno <= 0:
            pno = i
        key = f"page{pno:03d}"

        words_raw = p.get("words") or []
        words_out: List[Dict[str, Any]] = []
        by_line: Dict[Tuple[int, int], List[Tuple[float, float, float, float, str]]] = {}

        for w in words_raw:
            if not isinstance(w, (list, tuple)) or len(w) < 5:
                continue
            x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
            text = str(w[4])
            bno = int(w[5]) if len(w) > 5 else -1
            lno = int(w[6]) if len(w) > 6 else -1
            wno = int(w[7]) if len(w) > 7 else -1

            words_out.append({"text": text, "bbox": [x0, y0, x1, y1], "block": bno, "line": lno, "index": wno})
            by_line.setdefault((bno, lno), []).append((x0, y0, x1, y1, text))

        lines_out: List[Dict[str, Any]] = []
        for (bno, lno), items in sorted(by_line.items()):
            xs0 = [it[0] for it in items]; ys0 = [it[1] for it in items]
            xs1 = [it[2] for it in items]; ys1 = [it[3] for it in items]
            line_bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
            line_text = " ".join(it[4] for it in sorted(items, key=lambda t: (t[1], t[0])))
            lines_out.append({"text": line_text, "bbox": line_bbox, "block": bno, "line": lno})

        compiled[key] = {"page": pno, "words": words_out, "lines": lines_out}

    return md_text, compiled, page_index


def compile_text_entries(out_dir: str) -> str:
    pages_dir   = os.path.join(out_dir, "page_images")
    entries_dir = os.path.join(out_dir, "text_boxes_pages")
    out_path    = os.path.join(out_dir, "compiled_text_boxes.json")

    # Map: pageNNN -> (W,H)
    dims: Dict[str, Tuple[int,int]] = {}
    for p in sorted(os.listdir(pages_dir)):
        if not p.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(p)[0]  # pageNNN
        img = cv2.imread(os.path.join(pages_dir, p))
        if img is None:
            continue
        H, W = img.shape[:2]
        dims[stem] = (W, H)

    compiled: Dict[str, Any] = {"pages": [], "items_by_page": {}, "counts": {"pages": 0, "entries": 0}}

    # Helper: normalize rect to xywh, auto-detect xyxy if needed
    def to_xywh(rect: List[int], W: int, H: int) -> List[int]:
        x0, y0, a, b = rect
        # If adding a,b exceeds page size, assume xyxy
        if (x0 + a > W) or (y0 + b > H):
            x1, y1 = a, b
            x, y = max(0, min(x0, W-1)), max(0, min(y0, H-1))
            w, h = max(1, min(W - x, x1 - x0)), max(1, min(H - y, y1 - y0))
            return [x, y, w, h]
        # Already xywh
        x, y = max(0, min(x0, W-1)), max(0, min(y0, H-1))
        w, h = max(1, min(W - x, a)), max(1, min(H - y, b))
        return [x, y, w, h]

    # Iterate pages in numeric order
    for fname in sorted(os.listdir(entries_dir)):
        if not fname.lower().endswith(".json"):
            continue
        pno = int(re.search(r"page(\d+)\.json$", fname).group(1))
        key = f"page{pno:03d}"
        compiled["pages"].append(key)

        with open(os.path.join(entries_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("entries", [])
        W, H = dims.get(key, (10**9, 10**9))  # if missing image, skip bounds check effectively

        normalized = []
        for e in items:
            rect = e.get("rect", [0,0,1,1])
            text = e.get("text", "")
            xywh = to_xywh([int(v) for v in rect], W, H)
            normalized.append({"rect": xywh, "text": text})

        compiled["items_by_page"][key] = normalized
        compiled["counts"]["entries"] += len(normalized)

    compiled["counts"]["pages"] = len(compiled["pages"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compiled, f, ensure_ascii=False, indent=2)

    return out_path

# ---------------- PP-Structure OCR fallback ----------------

def _ppstruct_batch_ocr(image_paths: List[str], device: str,
                        det_limit_side_len: int, det_box_thresh: float,
                        rec_score_thresh: float, rec_batch: int) -> List[Dict[str, Any]]:
    """
    Run PP-StructureV3 on a list of page images and return the JSON outputs.
    """
    paddle.set_device("gpu" if device == "gpu" else "cpu")

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
        lang="en",
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
        Wpx, Hpx = image_sizes[i]
        Wpt, Hpt = page.rect.width, page.rect.height
        sx = Wpt / float(Wpx if Wpx > 0 else 1)
        sy = Hpt / float(Hpx if Hpx > 0 else 1)

        d = ppjson_pages[i]
        ocr = d.get("overall_ocr_res") or {}
        texts = ocr.get("rec_texts") or []
        boxes = ocr.get("rec_boxes") or []

        for txt, bb in zip(texts, boxes):
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue

            x0, y0, x1, y1 = bb
            if x1 <= x0 or y1 <= y0:
                x, y, w, h = x0, y0, x1, y1
                x0, y0, x1, y1 = x, y, x + w, y + h

            rx0, ry0 = float(x0) * sx, float(y0) * sy
            rx1, ry1 = float(x1) * sx, float(y1) * sy
            rect = fitz.Rect(rx0, ry0, rx1, ry1)

            page.insert_textbox(
                rect,
                txt,
                fontname="helv",
                fontsize=9,
                render_mode=3,   # invisible text
                overlay=True,
            )

    out_pdf = str(Path(src_pdf).with_name("ocr_overlay.pdf"))
    doc.save(out_pdf)
    doc.close()
    return out_pdf


# ---------------- Entries writers ----------------

def _write_entries_from_pymupdf_blocks(pdf_path: str,
                                       entries_dir: str,
                                       page_px_sizes: List[Tuple[int, int]]) -> int:
    """
    Plain PyMuPDF pass: page.get_text('blocks') -> pixel-aligned entries per page.
    Returns total entries count.
    """
    os.makedirs(entries_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = 0

    for pno in range(1, len(doc) + 1):
        page = doc[pno - 1]
        Wpt, Hpt = float(page.rect.width), float(page.rect.height)
        Wpx, Hpx = page_px_sizes[pno - 1]
        sx, sy = (Wpx / max(Wpt, 1e-6), Hpx / max(Hpt, 1e-6))

        # blocks: (x0,y0,x1,y1,text,block_no,block_type,...)
        blocks = page.get_text("blocks")
        entries = []
        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4] or "")
            if not txt.strip():
                continue
            X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
            X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
            x, y = max(0, min(X0, Wpx - 1)), max(0, min(Y0, Hpx - 1))
            w, h = max(1, min(Wpx - x, X1 - X0)), max(1, min(Hpx - y, Y1 - Y0))
            if w > 0 and h > 0:
                entries.append({"rect": [x, y, w, h], "text": txt})

        out = {"page": pno, "entries": entries}
        with open(os.path.join(entries_dir, f"page{pno:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        total += len(entries)

    doc.close()
    return total


def _write_entries_from_ppjson(ppjson_pages: List[Dict[str, Any]],
                               entries_dir: str) -> int:
    """
    PP-Structure overall_ocr_res -> pixel entries per page (already in pixels).
    Returns total entries count.
    """
    os.makedirs(entries_dir, exist_ok=True)
    total = 0

    for i, d in enumerate(ppjson_pages, start=1):
        ocr = d.get("overall_ocr_res") or {}
        texts = ocr.get("rec_texts") or []
        boxes = ocr.get("rec_boxes") or []
        entries = []

        for txt, bb in zip(texts, boxes):
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bb]
            x, y, w, h = x0, y0, max(1, x1 - x0), max(1, y1 - y0)
            entries.append({"rect": [x, y, w, h], "text": txt})

        out = {"page": i, "entries": entries}
        with open(os.path.join(entries_dir, f"page{i:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        total += len(entries)

    return total


def _ensure_entries_from_structure(structure: Dict[str, Any],
                                   entries_dir: str,
                                   page_px_sizes: List[Tuple[int, int]],
                                   pdf_path: str) -> None:
    """
    Use PM4L 'lines' bboxes (PDF points) -> write pixel-aligned entries per page.
    """
    os.makedirs(entries_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # structure keys are pageNNN
    for key in sorted(structure.keys()):
        payload = structure[key]
        pno = int(payload.get("page"))
        page = doc[pno - 1]

        Wpt, Hpt = float(page.rect.width), float(page.rect.height)
        Wpx, Hpx = page_px_sizes[pno - 1]
        sx, sy = (Wpx / max(Wpt, 1e-6), Hpx / max(Hpt, 1e-6))

        out = {"page": pno, "entries": []}

        for ln in payload.get("lines", []):
            x0, y0, x1, y1 = [float(v) for v in ln.get("bbox", [0, 0, 0, 0])]
            X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
            X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
            x, y = max(0, min(X0, Wpx - 1)), max(0, min(Y0, Hpx - 1))
            w, h = max(1, min(Wpx - x, X1 - X0)), max(1, min(Hpx - y, Y1 - Y0))
            if w > 0 and h > 0:
                out["entries"].append({"rect": [x, y, w, h], "text": ln.get("text", "")})

        with open(os.path.join(entries_dir, f"page{pno:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    doc.close()


# ---------------- Main entry ----------------

def run_text_extraction(
    pdf_path: str,
    out_dir: str,
    device: str = "gpu",
    *,
    det_limit_side_len: int = DET_LIMIT_SIDE_LEN,
    det_box_thresh: float = DET_BOX_THRESH,
    rec_score_thresh: float = REC_SCORE_THRESH,
    rec_batch: int = REC_BATCH,
) -> Dict[str, Any]:
    """
    1) Render page images once (reused later).
    2) Plain PyMuPDF: write pixel-aligned text boxes to text_boxes_pages/.
       If total boxes > 0, proceed.
       If total == 0, run OCR → write entries from OCR → overlay → proceed.
    3) PyMuPDF4LLM: write pm4l.md, text_structure.json, pm4l_page_index.json.
    Returns a bundle with paths, in-memory structures, and stats.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Images first
    pages_dir = os.path.join(out_dir, "page_images")
    image_paths, image_sizes = _render_pages(pdf_path, pages_dir)

    # Paths
    md_path = Path(out_dir) / "pm4l.md"
    struct_path = Path(out_dir) / "text_structure.json"
    page_index_path = Path(out_dir) / "pm4l_page_index.json"
    entries_dir = os.path.join(out_dir, "text_boxes_pages")
    os.makedirs(entries_dir, exist_ok=True)

    # Cached path
    if md_path.exists() and struct_path.exists() and os.listdir(entries_dir):
        with struct_path.open("r", encoding="utf-8") as f:
            structure = json.load(f)
        page_index = json.load(page_index_path.open("r", encoding="utf-8")) if page_index_path.exists() else []
        words_total = sum(len(p.get("words", [])) for p in structure.values())
        return {
            "markdown_path": str(md_path),
            "text_structure_path": str(struct_path),
            "page_index_path": str(page_index_path),
            "pages_dir": pages_dir,
            "image_paths": image_paths,
            "source": "cached",
            "pages_count": len(structure),
            "words_total": words_total,
            "structure": structure,
            "page_index": page_index,
        }

    # Pass-1: plain PyMuPDF boxes
    total_boxes = _write_entries_from_pymupdf_blocks(pdf_path, entries_dir, image_sizes)

    # If no boxes at all, OCR to create entries and an overlay PDF
    overlay_pdf = pdf_path
    source_flag = "pm4l"

    if total_boxes == 0:
        pp_pages = _ppstruct_batch_ocr(
            image_paths=image_paths,
            device=device,
            det_limit_side_len=det_limit_side_len,
            det_box_thresh=det_box_thresh,
            rec_score_thresh=rec_score_thresh,
            rec_batch=rec_batch,
        )
        _write_entries_from_ppjson(pp_pages, entries_dir)

        overlay_pdf = _insert_ocr_as_invisible_text(
            src_pdf=pdf_path,
            image_paths=image_paths,
            image_sizes=image_sizes,
            ppjson_pages=pp_pages,
        )
        source_flag = "ppstruct+pm4l"

    # Pass-2: PM4L for markdown + structure (+ index)
    pages = pymupdf4llm.to_markdown(
        overlay_pdf,
        page_chunks=True,
        extract_words=True,
        show_progress=True,
        table_strategy="lines_strict",
    )
    if not isinstance(pages, list) or not pages:
        raise RuntimeError("pm4l returned no pages.")

    md_text, structure, page_index = _pm4l_build_markdown_and_structure(pages)

    md_path.write_bytes(md_text.encode("utf-8"))
    struct_path.write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    page_index_path.write_text(json.dumps(page_index, indent=2), encoding="utf-8")

    # Ensure entries exist for the PM4L path too, scaled to pixels
    if total_boxes > 0:
        _ensure_entries_from_structure(structure, entries_dir, image_sizes, pdf_path)

    words_total = sum(len(p.get("words", [])) for p in structure.values())

    compiled_entries_path = compile_text_entries(out_dir)

    return {
        "markdown_path": str(md_path),
        "text_structure_path": str(struct_path),
        "page_index_path": str(page_index_path),
        "compiled_entries_path": compiled_entries_path,
        "pages_dir": pages_dir,
        "image_paths": image_paths,
        "source": source_flag,
        "pages_count": len(structure),
        "words_total": words_total,
        "structure": structure,
        "page_index": page_index,
    }
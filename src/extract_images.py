#!/usr/bin/env python3
"""
Image-first pipeline (photograph candidate finder) with unified text extraction.

Key behavior:
  • Render all page JPEGs.
  • WHOLE-PDF text extraction:
        - If PyMuPDF finds ANY spans across the doc → use PyMuPDF for ALL pages.
        - Else → run PaddleOCR **on the PDF path** for ALL pages (returns a list of per-page results).
  • Then: whitebox text → find figure/photograph boxes → save crops + debug masks.

Notes:
  • Paddle is now imported/used directly in-process (no helper subprocess).
  • For Paddle-on-PDF, we parse each page result, scale its boxes to the saved page JPEG size
    (if Paddle’s internal render size differs), and write:
      - rich per-page structure in text_struct_pages/
      - legacy entries (rect,text) in text_boxes_pages/
      - combined files text_struct.json and text_boxes.json
      - optional debug: ocr_json/pageNNN.json and ocr_images/pageNNN.jpg via save_to_json/save_to_img

Outputs:
  output/
    page_images/            pageNNN.jpg (rendered @300dpi, downscaled to cap long side)
    page_masked/            whiteboxed pages (text painted white)
    page_mask_bin/          cleaned binary masks (JPEG) used for CC detection (debug)
    bbox_crops/             candidate photo crops (≥1024 on both sides; aspect preserved)
    text_boxes_pages/       per-page legacy flat entries [{rect:[x,y,w,h], text}]
    text_boxes.json         all pages (legacy flat dict)
    text_struct_pages/      per-page rich structure (blocks → paragraphs → lines → spans)
    text_struct.json        all pages (rich structure)
    manifest.json           { pages: { pageNNN: { text: [...legacy entries...], crops: [...] } } }
    ocr_images/             (debug) Paddle overlay per page
    ocr_json/               (debug) Paddle raw per-page JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import statistics
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import fitz  # PyMuPDF
import cv2
import numpy as np

# ---------------- Global perf/threading limits ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---- Render / scaling config (shared across functions) ----
RENDER_DPI = 300                     # logical DPI for coordinate math
MAX_LONG_SIDE = 5000                 # cap saved page image's longest side
JPG_QUALITY = 95                     # default JPEG quality for all writes

BBox = Tuple[int, int, int, int]     # x, y, w, h  (pixel coords in saved-page space)

# ----------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------
def imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

def clean_boxes(boxes: List[BBox]) -> List[BBox]:
    """
    Remove near-duplicate bounding boxes based on heavy overlap thresholds.
    If a new box overlaps an existing one by ≥90% (either direction), drop it.
    """
    final: List[BBox] = []
    for x, y, w, h in boxes:
        new_area = max(1, w * h)
        skip = False
        for ox, oy, ow, oh in final:
            xi, yi = max(x, ox), max(y, oy)
            xj, yj = min(x + w, ox + ow), min(y + h, oy + oh)
            if xj > xi and yj > yi:
                inter = (xj - xi) * (yj - yi)
                if inter / new_area > 0.9 or inter / max(1, ow * oh) > 0.9:
                    skip = True
                    break
        if not skip:
            final.append((x, y, w, h))
    return final

def save_resized_crop(
    page_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_path: str,
    min_side: int = 1024
) -> None:
    x, y, w, h = map(int, bbox)
    H, W = page_img.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    crop = page_img[y:y + h, x:x + w].copy()
    scale = max(min_side / float(w), min_side / float(h), 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if scale > 1.0:
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    imwrite_jpg(out_path, crop)

# ----------------------------------------------------------------------------
# FIGURE / PHOTO CANDIDATE DETECTION (kept behavior)
# ----------------------------------------------------------------------------
def find_bounding_boxes(
    img: np.ndarray,
    min_pixel_area: int = 50,
    min_box_size: Tuple[int, int] = (30, 30),
    aspect_range: Tuple[float, float] = (0.3, 6.0),
    block_size: int = 15,
    C: int = 2,
    bridge_px: int = 0,
    min_area_frac: float = 0.0025,
    save_mask_dir: Optional[str] = None,
    save_mask_name: Optional[str] = None,
    detect_scale: float = 1.0,
    save_mask_binary: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Given a "whiteboxed" page image (text painted white), find non-text blocks likely
    to be photographs / figures using adaptive threshold + morphology + CC.
    """
    assert img is not None and img.ndim == 3, "expects BGR page image"

    H, W = img.shape[:2]
    scale = max(1e-6, float(detect_scale))
    small = img if scale == 1.0 else cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 2) Adaptive threshold to isolate dark-ish blocks (invert so figures = 255)
    bs = max(3, int(block_size) | 1)  # must be odd
    mask = cv2.adaptiveThreshold(
        gray_s, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs, int(C)
    )

    # 3) Morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    if bridge_px and bridge_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, bridge_px))
        mask = cv2.dilate(mask, k, iterations=1)

    # 4) Connected components
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    Hs, Ws = gray_s.shape
    page_area_s = Hs * Ws
    rel_area_floor_s = max(1, int(min_area_frac * page_area_s))
    lo_ar, hi_ar = max(aspect_range[0], 0.10), min(aspect_range[1], 10.0)

    keep_labels: List[int] = []
    raw_boxes_scaled: List[Tuple[int, int, int, int]] = []
    for lbl in range(1, num_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < rel_area_floor_s or area < min_pixel_area:
            continue

        x  = int(stats[lbl, cv2.CC_STAT_LEFT])
        y  = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        if bw * bh > 0.85 * page_area_s:
            continue
        if bw < max(1, int(round(min_box_size[0] * scale))) or \
           bh < max(1, int(round(min_box_size[1] * scale))):
            continue

        ar = bw / float(max(1, bh))
        if ar < lo_ar or ar > hi_ar:
            continue

        keep_labels.append(lbl)
        raw_boxes_scaled.append((x, y, bw, bh))

    # 5) Rescale boxes to full image
    raw_boxes = [
        (int(round(x / scale)), int(round(y / scale)),
         max(1, int(round(w / scale))), max(1, int(round(h / scale))))
        for (x, y, w, h) in raw_boxes_scaled
    ] if scale != 1.0 else raw_boxes_scaled

    # 6) Optional debug binary mask
    if save_mask_dir and save_mask_name and save_mask_binary:
        if scale != 1.0:
            clean_small = np.zeros_like(labels, dtype=np.uint8)
            for lbl in keep_labels: clean_small[labels == lbl] = 255
            clean_full = cv2.resize(clean_small, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            clean_full = np.zeros_like(labels, dtype=np.uint8)
            for lbl in keep_labels: clean_full[labels == lbl] = 255
        mask_bgr = cv2.cvtColor(clean_full, cv2.COLOR_GRAY2BGR)
        os.makedirs(save_mask_dir, exist_ok=True)
        imwrite_jpg(os.path.join(save_mask_dir, f"{save_mask_name}.jpg"), mask_bgr, quality=95)

    return clean_boxes(raw_boxes)

# ----------------------------------------------------------------------------
# UNIFIED TEXT-STRUCTURE HELPERS
# ----------------------------------------------------------------------------
def _xyxy_to_xywh(x0: float, y0: float, x1: float, y1: float) -> List[int]:
    x = int(round(min(x0, x1)))
    y = int(round(min(y0, y1)))
    w = max(1, int(round(abs(x1 - x0))))
    h = max(1, int(round(abs(y1 - y0))))
    return [x, y, w, h]

def _union(a: List[int], b: List[int]) -> List[int]:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0 = min(ax, bx); y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw); y1 = max(ay + ah, by + bh)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]

def _median(vals: List[float], default: float) -> float:
    try:
        return statistics.median(vals) if vals else default
    except Exception:
        return default

def _looks_bold_italic(font: Optional[str]) -> Tuple[Optional[bool], Optional[bool]]:
    if not font:
        return None, None
    f = font.lower()
    return ("bold" in f or "black" in f or "heavy" in f), ("italic" in f or "oblique" in f)

def _group_lines_into_paragraphs(lines: List[Dict[str, Any]], page_w: int) -> List[List[Dict[str, Any]]]:
    """
    Group consecutive lines into paragraphs if:
      • vertical gap <= 1.5× median line height, and
      • left edges similar (<= max(24px, 5% of page width))
    """
    if not lines:
        return []
    heights = [ln['bbox'][3] for ln in lines]
    med_h = _median(heights, 18)
    out: List[List[Dict[str, Any]]] = []
    cur = [lines[0]]
    for prev, nxt in zip(lines, lines[1:]):
        _, py, pw, ph = prev['bbox']; _, ny, nw, nh = nxt['bbox']
        gap = ny - (py + ph)
        left_sim = abs(prev['bbox'][0] - nxt['bbox'][0]) <= max(24, int(0.05 * page_w))
        if gap <= 1.5 * med_h and left_sim:
            cur.append(nxt)
        else:
            out.append(cur); cur = [nxt]
    out.append(cur)
    return out

def _flatten_blocks(blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (entries_flat, lines_flat, spans_flat) from hierarchical blocks.
    entries_flat keeps legacy shape for whiteboxing: {'rect':[x,y,w,h], 'text': str}
    """
    entries: List[Dict[str, Any]] = []
    lines_flat: List[Dict[str, Any]] = []
    spans_flat: List[Dict[str, Any]] = []
    for b in blocks:
        for p in b.get('paragraphs', []):
            for ln in p.get('lines', []):
                lines_flat.append({
                    'block_id': b['block_id'], 'para_id': p['para_id'],
                    'line_id': ln['line_id'], 'bbox': ln['bbox'], 'text': ln.get('text', '')
                })
                for sp in ln.get('spans', []):
                    spans_flat.append({
                        'block_id': b['block_id'], 'para_id': p['para_id'], 'line_id': ln['line_id'],
                        'span_id': sp['span_id'], 'bbox': sp['bbox'], 'text': sp.get('text', ''),
                        'font': sp.get('font'), 'size': sp.get('size'),
                        'bold': sp.get('bold'), 'italic': sp.get('italic'), 'color': sp.get('color')
                    })
                    entries.append({'rect': sp['bbox'], 'text': sp.get('text', '')})
    return entries, lines_flat, spans_flat

def _scale_page_struct(page_obj: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Scale all bboxes in a structured page dict and rebuild flats."""
    if abs(scale - 1.0) < 1e-6:
        return page_obj
    for b in page_obj.get('blocks', []):
        b['bbox'] = [int(round(b['bbox'][0]*scale)), int(round(b['bbox'][1]*scale)),
                     max(1,int(round(b['bbox'][2]*scale))), max(1,int(round(b['bbox'][3]*scale)))]
        for p in b.get('paragraphs', []):
            p['bbox'] = [int(round(p['bbox'][0]*scale)), int(round(p['bbox'][1]*scale)),
                         max(1,int(round(p['bbox'][2]*scale))), max(1,int(round(p['bbox'][3]*scale)))]
            for ln in p.get('lines', []):
                ln['bbox'] = [int(round(ln['bbox'][0]*scale)), int(round(ln['bbox'][1]*scale)),
                              max(1,int(round(ln['bbox'][2]*scale))), max(1,int(round(ln['bbox'][3]*scale)))]
                for sp in ln.get('spans', []):
                    sp['bbox'] = [int(round(sp['bbox'][0]*scale)), int(round(sp['bbox'][1]*scale)),
                                  max(1,int(round(sp['bbox'][2]*scale))), max(1,int(round(sp['bbox'][3]*scale)))]
    entries, lines_flat, spans_flat = _flatten_blocks(page_obj['blocks'])
    page_obj['lines_flat'] = lines_flat
    page_obj['spans_flat'] = spans_flat
    page_obj['entries'] = entries
    return page_obj

# ----------------------------------------------------------------------------
# PYMU PDF TEXT EXTRACTION (rich)
# ----------------------------------------------------------------------------
def extract_struct_pymupdf(pdf_path: str) -> Dict[str, Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    SCALE = RENDER_DPI / 72.0
    result: Dict[str, Dict[str, Any]] = {}

    for i, page in enumerate(doc, start=1):
        page_key = f"page{i:03d}"
        rect = page.rect
        px_w = int(round(rect.width * SCALE))
        px_h = int(round(rect.height * SCALE))

        blocks_out: List[Dict[str, Any]] = []
        block_id = 0
        text_dict = page.get_text("dict")  # full structure

        for blk in text_dict.get("blocks", []):
            btype = blk.get("type", 1)
            bb_xyxy = blk.get("bbox", [0, 0, 0, 0])
            bb = _xyxy_to_xywh(bb_xyxy[0]*SCALE, bb_xyxy[1]*SCALE, bb_xyxy[2]*SCALE, bb_xyxy[3]*SCALE)

            if btype != 0:
                blocks_out.append({
                    'block_id': block_id, 'type': 'image' if btype == 1 else 'other',
                    'bbox': bb, 'paragraphs': []
                })
                block_id += 1
                continue

            raw_lines = []
            for ln in blk.get("lines", []):
                lbb_xyxy = ln.get("bbox", [0, 0, 0, 0])
                lbb = _xyxy_to_xywh(lbb_xyxy[0]*SCALE, lbb_xyxy[1]*SCALE, lbb_xyxy[2]*SCALE, lbb_xyxy[3]*SCALE)
                spans = []
                span_id = 0
                line_text_parts = []
                for sp in ln.get("spans", []):
                    sbb_xyxy = sp.get("bbox", [0, 0, 0, 0])
                    sbb = _xyxy_to_xywh(sbb_xyxy[0]*SCALE, sbb_xyxy[1]*SCALE, sbb_xyxy[2]*SCALE, sbb_xyxy[3]*SCALE)
                    txt = sp.get("text", "") or ""
                    font = sp.get("font"); size = sp.get("size")
                    bold, italic = _looks_bold_italic(font)
                    color = sp.get("color")
                    spans.append({
                        'span_id': span_id, 'bbox': sbb, 'text': txt,
                        'font': font, 'size': size, 'bold': bold, 'italic': italic, 'color': color
                    })
                    span_id += 1
                    if txt:
                        line_text_parts.append(txt)

                raw_lines.append({'bbox': lbb, 'text': "".join(line_text_parts), 'spans': spans, 'page_w': px_w})

            paragraphs = []
            for para_id, group in enumerate(_group_lines_into_paragraphs(raw_lines, px_w)):
                pbox = None
                lines_out = []
                for line_id, ln in enumerate(group):
                    lines_out.append({'line_id': line_id, 'bbox': ln['bbox'], 'text': ln['text'], 'spans': ln['spans']})
                    pbox = ln['bbox'] if pbox is None else _union(pbox, ln['bbox'])
                paragraphs.append({'para_id': para_id, 'bbox': pbox or [0,0,1,1], 'lines': lines_out})

            blocks_out.append({'block_id': block_id, 'type': 'text', 'bbox': bb, 'paragraphs': paragraphs})
            block_id += 1

        entries, lines_flat, spans_flat = _flatten_blocks(blocks_out)
        result[page_key] = {
            'page_size': {'w': px_w, 'h': px_h, 'scale': 1.0, 'dpi': RENDER_DPI},
            'blocks': blocks_out,
            'lines_flat': lines_flat,
            'spans_flat': spans_flat,
            'entries': entries
        }
    return result

# ----------------------------------------------------------------------------
# RENDER PAGES AS IMAGES (JPEG, capped at MAX_LONG_SIDE)
# ----------------------------------------------------------------------------
def render_pages(pdf_path: str, out_dir: str) -> Dict[str, Dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_info: Dict[str, Dict[str, float]] = {}
    for i, page in enumerate(doc, start=1):
        page_key = f"page{i:03d}"
        out_path = os.path.join(out_dir, f"{page_key}.jpg")

        pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H, W = bgr.shape[:2]
        longest = max(W, H)
        scale = 1.0
        if longest > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / float(longest)
            new_w = max(1, int(round(W * scale)))
            new_h = max(1, int(round(H * scale)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        imwrite_jpg(out_path, bgr)
        page_info[page_key] = {'w': bgr.shape[1], 'h': bgr.shape[0], 'scale': scale}
    return page_info

# ----------------------------------------------------------------------------
# PADDLE OCR (PDF in, per-page results out) — integrated
# ----------------------------------------------------------------------------
def _ensure_paddle(device: str):
    # Import inside to avoid overhead when not used
    from paddleocr import PaddleOCR
    # keep it lean/simple; doc features off
    return PaddleOCR(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

def _paddle_predict_pdf(ocr, pdf_path: str):
    """
    Call Paddle's .predict on the PDF path.
    Returns a list of 'result objects' (one per page). Each should expose:
      - .json (or .json['res']) containing rec_boxes/rec_texts/rec_scores
      - .img (the page image Paddle used)  [optional but common]
      - .save_to_json(path), .save_to_img(path) for debug
    """
    try:
        return ocr.predict(pdf_path)
    except Exception as e:
        print(f"[PADDLE] ERROR: predict(pdf) failed: {e}", file=sys.stderr)
        return []

def _group_spans_to_lines(spans: List[Dict[str, Any]], page_w: int) -> List[List[Dict[str, Any]]]:
    """Greedy line clustering by y-band; then left→right order."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s['bbox'][1], s['bbox'][0]))
    heights = [s['bbox'][3] for s in spans]
    med_h = _median(heights, 18)
    lines: List[List[Dict[str, Any]]] = [[spans[0]]]
    for sp in spans[1:]:
        cy = sp['bbox'][1] + sp['bbox'][3]/2
        placed = False
        for grp in lines:
            gy = grp[-1]['bbox'][1] + grp[-1]['bbox'][3]/2
            if abs(cy - gy) <= 0.6 * med_h:
                grp.append(sp); placed = True; break
        if not placed:
            lines.append([sp])
    for grp in lines:
        grp.sort(key=lambda s: s['bbox'][0])
    return lines

def build_struct_via_paddle_pdf(
    pdf_path: str,
    page_info: Dict[str, Dict[str, float]],
    output_dir: str,
    device: str = "gpu",
    save_debug: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run PaddleOCR directly on the PDF. Parse each page, scale bboxes to our saved JPEG size,
    and return the unified per-page structure dict.
    """
    page_keys = sorted(page_info.keys())
    root_dir = output_dir
    ocr_img_dir = os.path.join(root_dir, 'ocr_images')
    ocr_json_dir = os.path.join(root_dir, 'ocr_json')
    if save_debug:
        os.makedirs(ocr_img_dir, exist_ok=True)
        os.makedirs(ocr_json_dir, exist_ok=True)

    ocr = _ensure_paddle(device=device)
    res_list = _paddle_predict_pdf(ocr, pdf_path)

    if not res_list:
        print("[PIPELINE] WARN: Paddle returned no results; continuing with empty text.")
        empty_page = {
            'page_size': {'w': 0, 'h': 0, 'scale': 1.0, 'dpi': RENDER_DPI},
            'blocks': [], 'lines_flat': [], 'spans_flat': [], 'entries': []
        }
        return {pk: empty_page for pk in page_keys}

    # Align counts defensively
    n = min(len(page_keys), len(res_list))
    if len(res_list) != len(page_keys):
        print(f"[PADDLE] Note: page count mismatch: results={len(res_list)} vs pages={len(page_keys)}; truncating to {n}")

    out: Dict[str, Dict[str, Any]] = {}
    for idx in range(n):
        pk = page_keys[idx]
        r = res_list[idx]

        # Pull per-page JSON payload
        try:
            data = r.json.get('res', r.json)
        except Exception:
            data = {}

        boxes = data.get('rec_boxes', []) or []
        texts = data.get('rec_texts', []) or []
        scores = data.get('rec_scores', []) or [None] * len(texts)

        # Source img size (prefer r.img; else derive from boxes; else fallback to our target)
        src_w = src_h = None
        try:
            im = getattr(r, "img", None)
            if im is not None:
                if isinstance(im, np.ndarray):
                    src_h, src_w = im.shape[:2]
                else:
                    # PIL.Image
                    src_w, src_h = im.size  # type: ignore
        except Exception:
            pass
        if src_w is None or src_h is None:
            try:
                xs = [float(max(b[0], b[2])) for b in boxes]
                ys = [float(max(b[1], b[3])) for b in boxes]
                src_w = int(max(xs)) if xs else int(page_info[pk]['w'])
                src_h = int(max(ys)) if ys else int(page_info[pk]['h'])
            except Exception:
                src_w = int(page_info[pk]['w']); src_h = int(page_info[pk]['h'])

        tgt_w = int(page_info[pk]['w']); tgt_h = int(page_info[pk]['h'])
        sx = (tgt_w / float(src_w)) if src_w else 1.0
        sy = (tgt_h / float(src_h)) if src_h else 1.0

        # Build spans (scaled to saved JPEG coords)
        spans = []
        span_id = 0
        for b, t, s in zip(boxes, texts, scores):
            try:
                x0, y0, x1, y1 = map(float, b)
            except Exception:
                # Some builds may hand back [[x0,y0], [x1,y1], ...]; handle that too
                flat = [p for p in b]  # best-effort
                xs = [float(p[0]) for p in flat]; ys = [float(p[1]) for p in flat]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            bb = _xyxy_to_xywh(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            txt = (t or "").strip()
            if not txt:
                continue
            spans.append({
                'span_id': span_id, 'bbox': bb, 'text': txt,
                'font': None, 'size': float(bb[3]), 'bold': None, 'italic': None, 'color': None
            })
            span_id += 1

        # Group to lines → paragraphs → blocks
        lines_list = []
        for line_idx, grp in enumerate(_group_spans_to_lines(spans, tgt_w)):
            line_text = " ".join([s['text'] for s in grp]).strip()
            lbb = grp[0]['bbox']
            for s in grp[1:]:
                lbb = _union(lbb, s['bbox'])
            lines_list.append({'line_id': line_idx, 'bbox': lbb, 'text': line_text, 'spans': grp})

        tmp_lines = [{'bbox': ln['bbox'], 'text': ln['text'], 'spans': ln['spans'], 'page_w': tgt_w} for ln in lines_list]
        paragraphs = []
        for para_id, group in enumerate(_group_lines_into_paragraphs(tmp_lines, tgt_w)):
            pbox = None
            lines_out = []
            for ln in group:
                lines_out.append({'line_id': len(lines_out), 'bbox': ln['bbox'], 'text': ln['text'], 'spans': ln['spans']})
                pbox = ln['bbox'] if pbox is None else _union(pbox, ln['bbox'])
            paragraphs.append({'para_id': para_id, 'bbox': pbox or [0,0,1,1], 'lines': lines_out})

        blocks = [{'block_id': 0, 'type': 'text', 'bbox': paragraphs[0]['bbox'] if paragraphs else [0,0,tgt_w,tgt_h], 'paragraphs': paragraphs}]
        entries, lines_flat, spans_flat = _flatten_blocks(blocks)
        out[pk] = {
            'page_size': {'w': tgt_w, 'h': tgt_h, 'scale': 1.0, 'dpi': RENDER_DPI},
            'blocks': blocks, 'lines_flat': lines_flat, 'spans_flat': spans_flat, 'entries': entries
        }

        # Debug artifacts from Paddle
        if save_debug:
            try:
                r.save_to_json(os.path.join(ocr_json_dir, f"{pk}.json"))
            except Exception as e:
                print(f"[PADDLE] warn: save_to_json failed for {pk}: {e}", file=sys.stderr)
            try:
                r.save_to_img(os.path.join(ocr_img_dir, f"{pk}.jpg"))
            except Exception as e:
                print(f"[PADDLE] warn: save_to_img failed for {pk}: {e}", file=sys.stderr)

    return out

# ----------------------------------------------------------------------------
# WHOLE-PDF text selection (PyMuPDF OR OCR for ALL pages)
# ----------------------------------------------------------------------------
def _build_struct_all_or_nothing(
    pdf_path: str,
    page_info: Dict[str, Dict[str, float]],
    output_dir: str,
    device: str = "gpu",
    save_debug: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Whole-PDF logic:
      - If PyMuPDF finds ANY spans across the document, use PyMuPDF for ALL pages.
      - Else run Paddle OCR (PDF-in) for ALL pages.

    Returns: (unified_struct, engine_used) where engine_used in {"pymupdf","ocr"}.
    """
    # 1) Try PyMuPDF (300dpi coords), then scale to saved-JPEG coords
    struct300 = extract_struct_pymupdf(pdf_path)
    total_spans = sum(len(p.get("spans_flat", [])) for p in struct300.values())

    if total_spans > 0:
        unified_struct: Dict[str, Dict[str, Any]] = {}
        for page_key, pobj in struct300.items():
            sc = float(page_info.get(page_key, {}).get('scale', 1.0))
            pobj_scaled = _scale_page_struct(pobj, sc)
            pobj_scaled['page_size'] = {
                'w': page_info[page_key]['w'],
                'h': page_info[page_key]['h'],
                'scale': sc,
                'dpi': RENDER_DPI
            }
            unified_struct[page_key] = pobj_scaled
        print(f"[PIPELINE] PyMuPDF spans found (total={total_spans}) → using PyMuPDF for all pages")
        return unified_struct, "pymupdf"

    # 2) Else: OCR every page via Paddle on the PDF path
    print(f"[PIPELINE] PyMuPDF found no text → running Paddle OCR on the PDF for {len(page_info)} page(s)")
    ocr_struct = build_struct_via_paddle_pdf(
        pdf_path=pdf_path,
        page_info=page_info,
        output_dir=output_dir,
        device=device,
        save_debug=save_debug,
    )
    return ocr_struct, "ocr"

# ----------------------------------------------------------------------------
# CROP WORKER (whitebox text → CC mask → candidate crops)
# ----------------------------------------------------------------------------
def crop_worker(args):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    page_key, dirs = args
    page_img = cv2.imread(os.path.join(dirs['pages'], f"{page_key}.jpg"))
    H, W = page_img.shape[:2]

    entries_path = os.path.join(dirs['entries_pages'], f"{page_key}.json")
    with open(entries_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # 1) Whitebox text
    masked = page_img.copy()
    for e in entries:
        x, y, w, h = e['rect']
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(0, min(int(w), W - x))
        h = max(0, min(int(h), H - y))
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(masked, (x, y), (x + w, y + h), (255, 255, 255), -1)

    imwrite_jpg(os.path.join(dirs['mask'], f"{page_key}.jpg"), masked)

    # 2) Find figure boxes AND save the cleaned binary mask
    final = find_bounding_boxes(
        masked,
        save_mask_dir=dirs['mask_bin'],
        save_mask_name=page_key
    )

    # 3) Save crops (≥1024 both sides)
    os.makedirs(dirs['crops'], exist_ok=True)
    crops = []
    for idx, (x, y, w, h) in enumerate(final, start=1):
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(0, min(int(w), W - x))
        h = max(0, min(int(h), H - y))
        if w <= 0 or h <= 0:
            continue

        crop_img = masked[y:y+h, x:x+w]
        scale = max(1024.0 / float(w), 1024.0 / float(h), 1.0)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if scale > 1.0:
            crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        fname = f"{page_key}_crop{idx:02d}.jpg"
        imwrite_jpg(os.path.join(dirs['crops'], fname), crop_img)
        crops.append({'rect': [x, y, w, h], 'file': fname})

    return page_key, crops

# ----------------------------------------------------------------------------
# MAIN PIPELINE (image-first; unified text struct + legacy entries)
# ----------------------------------------------------------------------------
def extract_images_pipeline(pdf_path: str, output_dir: str, workers: int = None, ocr_device: str = "gpu", save_ocr_debug: bool = True):
    # prepare dirs
    dirs = {
        'pages': os.path.join(output_dir, 'page_images'),
        'mask': os.path.join(output_dir, 'page_masked'),
        'mask_bin': os.path.join(output_dir, 'page_mask_bin'),
        'crops': os.path.join(output_dir, 'bbox_crops'),
        'entries_pages': os.path.join(output_dir, 'text_boxes_pages'),
        'struct_pages': os.path.join(output_dir, 'text_struct_pages'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1) render pages (JPEG @ 300dpi, capped at MAX_LONG_SIDE)
    page_info = render_pages(pdf_path, dirs["pages"])  # contains per-page downscale 'scale'

    # 2) WHOLE-PDF text: use PyMuPDF if ANY spans exist; otherwise Paddle OCR (PDF-in) for ALL pages
    unified_struct, engine_used = _build_struct_all_or_nothing(
        pdf_path=pdf_path,
        page_info=page_info,
        output_dir=output_dir,
        device=ocr_device,
        save_debug=save_ocr_debug
    )

    # Save per-page rich structure + master file
    for pk, pobj in unified_struct.items():
        with open(os.path.join(dirs['struct_pages'], f"{pk}.json"), "w", encoding="utf-8") as f:
            json.dump(pobj, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "text_struct.json"), "w", encoding="utf-8") as f:
        json.dump(unified_struct, f, indent=2, ensure_ascii=False)

    # Derive legacy flat entries (rect,text) from the structured pages
    text_data: Dict[str, List[Dict]] = {pk: pobj.get('entries', []) for pk, pobj in unified_struct.items()}

    # 2b) Write per-page entries JSON (legacy)
    for page_key, entries in text_data.items():
        with open(os.path.join(dirs['entries_pages'], f"{page_key}.json"), 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)

    # Also save a combined legacy text_boxes.json (unchanged)
    with open(os.path.join(output_dir, 'text_boxes.json'), 'w', encoding='utf-8') as f:
        json.dump(text_data, f, indent=2)

    # 3 & 4: whitebox + bbox detection + cropping (+ save cleaned bin masks)
    page_keys = sorted(text_data.keys())
    args_list = [(pk, dirs) for pk in page_keys]

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 4)
    print(f"[PIPELINE] Running cropping with {workers} workers (engine={engine_used})")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(crop_worker, args_list))

    # manifest
    manifest = {'pages': {}}
    for page_num, crops in results:
        manifest['pages'][page_num] = {'text': text_data[page_num], 'crops': crops}
    with open(os.path.join(output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

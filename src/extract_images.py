#!/usr/bin/env python3
"""
Image-first pipeline (photograph candidate finder) with unified text extraction.

This version isolates OCR fallback into a separate helper process (ocr_struct_helper.py)
to avoid Torch↔Paddle CUDA DLL conflicts on Windows. No Paddle imports occur here.

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
    ocr_helper/             temp files for the OCR helper (page list + returned JSON)
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Optional, Any
import statistics
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
from pathlib import Path

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
MAX_LONG_SIDE = 2400                 # cap saved page image's longest side
JPG_QUALITY = 95                     # default JPEG quality for all writes

BBox = Tuple[int, int, int, int]     # x, y, w, h  (pixel coords in saved-page space)

# ----------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------
def imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    detect_scale: float = 1.0,         # e.g., 0.5 for speed; boxes will be rescaled up
    save_mask_binary: bool = True,     # write cleaned mask as JPEG if save dirs provided
) -> List[Tuple[int, int, int, int]]:
    """
    Given a "whiteboxed" page image (text painted white), find non-text blocks likely
    to be photographs / figures using adaptive threshold + morphology + CC.

    Steps:
      1) (optional) downscale to 'detect_scale' for speed.
      2) grayscale -> adaptiveThreshold (binary INV): dark blobs=1.
      3) morphological close to fill tiny holes; optional dilate bridge.
      4) connectedComponentsWithStats -> stats → boxes
      5) filter by absolute area, relative area, box size, and aspect ratio
      6) rescale to original if downscaled; dedupe overlaps.

    Returns: list of (x, y, w, h) in original image coordinates.
    """
    assert img is not None and img.ndim == 3, "expects BGR page image"

    # 1) Downscale for detection if requested
    H, W = img.shape[:2]
    scale = max(1e-6, float(detect_scale))
    if scale != 1.0:
        small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img

    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 2) Adaptive threshold to isolate dark-ish blocks (invert so figures = 255)
    bs = max(3, int(block_size) | 1)  # must be odd
    mask = cv2.adaptiveThreshold(
        gray_s, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs, int(C)
    )

    # 3) Morph close to join near components & fill pinholes
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Optional bridge/dilate
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

        # Oversize guard relative to scaled page (remove near-page-size blobs)
        if bw * bh > 0.85 * page_area_s:
            continue

        # Absolute min box size (scaled)
        if bw < max(1, int(round(min_box_size[0] * scale))) or \
           bh < max(1, int(round(min_box_size[1] * scale))):
            continue

        # Aspect ratio filter
        ar = bw / float(max(1, bh))
        if ar < lo_ar or ar > hi_ar:
            continue

        keep_labels.append(lbl)
        raw_boxes_scaled.append((x, y, bw, bh))

    # 5) Scale boxes back up to original coordinate space if downscaled
    if scale != 1.0:
        raw_boxes = [
            (int(round(x / scale)), int(round(y / scale)),
             max(1, int(round(w / scale))), max(1, int(round(h / scale))))
            for (x, y, w, h) in raw_boxes_scaled
        ]
    else:
        raw_boxes = raw_boxes_scaled

    # 6) Optionally write a cleaned binary mask JPEG (uses the labels we kept)
    if save_mask_dir and save_mask_name and save_mask_binary:
        if scale != 1.0:
            clean_small = np.zeros_like(labels, dtype=np.uint8)
            for lbl in keep_labels:
                clean_small[labels == lbl] = 255
            clean_full = cv2.resize(clean_small, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            clean_full = np.zeros_like(labels, dtype=np.uint8)
            for lbl in keep_labels:
                clean_full[labels == lbl] = 255

        mask_bgr = cv2.cvtColor(clean_full, cv2.COLOR_GRAY2BGR)
        os.makedirs(save_mask_dir, exist_ok=True)
        imwrite_jpg(os.path.join(save_mask_dir, f"{save_mask_name}.jpg"), mask_bgr, quality=95)

    # Dedupe near-overlapping boxes
    final_boxes = clean_boxes(raw_boxes)
    return final_boxes

# ----------------------------------------------------------------------------
# UNIFIED TEXT-STRUCTURE HELPERS (new)
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
    """
    Rich PyMuPDF extraction at 300dpi coords (PDF points × RENDER_DPI/72).
    Output per page:
      {
        'page_size': {'w': px_w, 'h': px_h, 'scale': 1.0, 'dpi': RENDER_DPI},
        'blocks': [ { block_id, type, bbox, paragraphs:[{ para_id, bbox, lines:[{ line_id, bbox, text, spans:[{ span_id, bbox, text, font, size, bold, italic, color }] }]}] } ],
        'lines_flat': [...],
        'spans_flat': [...],
        'entries': [ {'rect':[x,y,w,h], 'text': str}, ... ]   # legacy
      }
    """
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
                # keep non-text blocks as "image"/"other" for layout modeling
                blocks_out.append({
                    'block_id': block_id, 'type': 'image' if btype == 1 else 'other',
                    'bbox': bb, 'paragraphs': []
                })
                block_id += 1
                continue

            # Text block → lines/spans
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
    """
    Render each page at RENDER_DPI, then (if needed) downscale so the longest side is MAX_LONG_SIDE.
    Save as JPEG. Returns a map {page_key: {'w': int, 'h': int, 'scale': float}}, where 'scale'
    is the downscale factor applied to the 300-DPI pixel dimensions.
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_info: Dict[str, Dict[str, float]] = {}
    for i, page in enumerate(doc, start=1):
        page_key = f"page{i:03d}"
        out_path = os.path.join(out_dir, f"{page_key}.jpg")

        pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
        # Convert to BGR for OpenCV save
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
# OCR HELPER RUNNER (subprocess)
# ----------------------------------------------------------------------------
def _run_ocr_helper(pages_dir: str, page_keys: List[str], work_dir: str, device: str = "cpu", save_debug: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Launch ocr_struct_helper.py as a separate process so PaddleOCR loads
    in a different process (no Torch/Paddle DLL conflict).
    """
    if not page_keys:
        return {}

    # write page list file
    os.makedirs(work_dir, exist_ok=True)
    pages_file = os.path.join(work_dir, "need_ocr_pages.txt")
    with open(pages_file, "w", encoding="utf-8") as f:
        f.write("\n".join(page_keys))

    out_json = os.path.join(work_dir, "ocr_struct.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    helper_path = Path(__file__).with_name("ocr_struct_helper.py")
    cmd = [
        sys.executable,
        str(helper_path),
        "--pages-dir", pages_dir,
        "--pages-file", pages_file,
        "--out-json", out_json,
        "--device", device,
    ]
    if save_debug:
        cmd.append("--save-debug")

    print("[OCR-HELPER] Launching:", " ".join(cmd))
    # If the helper crashes, raise so you notice
    subprocess.run(cmd, check=True)

    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------------
# CROP WORKER (whitebox text → CC mask → candidate crops)
# ----------------------------------------------------------------------------
def crop_worker(args):
    # Ensure OpenCV single-threaded inside workers
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    page_key, dirs = args
    # Load page image
    page_img = cv2.imread(os.path.join(dirs['pages'], f"{page_key}.jpg"))
    H, W = page_img.shape[:2]

    # Load pre-scaled entries written per page
    entries_path = os.path.join(dirs['entries_pages'], f"{page_key}.json")
    with open(entries_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # 1) Whitebox text (paint rectangles over text spans)
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

    # Save whiteboxed page
    imwrite_jpg(os.path.join(dirs['mask'], f"{page_key}.jpg"), masked)

    # 2) Find figure boxes AND save the cleaned binary mask (inside find_bounding_boxes)
    final = find_bounding_boxes(
        masked,
        save_mask_dir=dirs['mask_bin'],
        save_mask_name=page_key
    )

    # 3) Save crops (ensure ≥1024 on both sides, preserve aspect)
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
def extract_images_pipeline(pdf_path: str, output_dir: str, workers: int = None):
    # prepare dirs
    dirs = {
        'pages': os.path.join(output_dir, 'page_images'),
        'mask': os.path.join(output_dir, 'page_masked'),       # whiteboxed pages (JPEG)
        'mask_bin': os.path.join(output_dir, 'page_mask_bin'), # cleaned thresh masks (JPEG)
        'crops': os.path.join(output_dir, 'bbox_crops'),
        'entries_pages': os.path.join(output_dir, 'text_boxes_pages'),  # legacy per-page entries JSON
        'struct_pages': os.path.join(output_dir, 'text_struct_pages'),  # rich per-page structure
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1) render pages (JPEG @ 300dpi, capped at MAX_LONG_SIDE)
    page_info = render_pages(pdf_path, dirs["pages"])  # contains per-page downscale 'scale'

    # 2) Build unified structured text (PyMuPDF at 300dpi coords)
    struct300 = extract_struct_pymupdf(pdf_path)  # rich structure at pre-downscale coords

    # Downscale structured pages to match saved JPEG coords (using page_info['scale'])
    unified_struct: Dict[str, Dict[str, Any]] = {}
    for page_key, pobj in struct300.items():
        sc = float(page_info.get(page_key, {}).get('scale', 1.0))
        pobj_scaled = _scale_page_struct(pobj, sc)
        # Replace page_size with actual saved JPEG size
        pobj_scaled['page_size'] = {
            'w': page_info[page_key]['w'], 'h': page_info[page_key]['h'], 'scale': sc, 'dpi': RENDER_DPI
        }
        unified_struct[page_key] = pobj_scaled

    # If a page has zero spans after PyMuPDF, fill it via OCR with the same structure (helper subprocess)
    need_ocr = [pk for pk, pobj in unified_struct.items() if len(pobj.get('spans_flat', [])) == 0]
    if need_ocr:
        print(f"[PIPELINE] {len(need_ocr)} page(s) had no text via PyMuPDF → running OCR struct fallback (external helper)")
        try:
            # device="cpu" avoids any CUDA dependency in the helper
            ocr_struct = _run_ocr_helper(
                pages_dir=dirs['pages'],
                page_keys=need_ocr,
                work_dir=os.path.join(output_dir, "ocr_helper"),
                device="gpu",
                save_debug=True
            )
            for pk in need_ocr:
                if pk in ocr_struct:
                    unified_struct[pk] = ocr_struct[pk]
                else:
                    print(f"[PIPELINE] warn: no OCR result for {pk}, leaving as-is")
        except subprocess.CalledProcessError as e:
            print(f"[PIPELINE] ERROR running OCR helper: {e}\nSkipping OCR fallback.")
    else:
        print("[PIPELINE] PyMuPDF found text on all pages; OCR struct fallback not needed")

    # Save per-page rich structure + master file
    for pk, pobj in unified_struct.items():
        with open(os.path.join(dirs['struct_pages'], f"{pk}.json"), "w", encoding="utf-8") as f:
            json.dump(pobj, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "text_struct.json"), "w", encoding="utf-8") as f:
        json.dump(unified_struct, f, indent=2, ensure_ascii=False)

    # Derive legacy flat entries (rect,text) from the structured pages (keeps downstream identical)
    text_data: Dict[str, List[Dict]] = {pk: pobj.get('entries', []) for pk, pobj in unified_struct.items()}

    # 2b) Write per-page entries JSON (unchanged legacy dir)
    for page_key, entries in text_data.items():
        with open(os.path.join(dirs['entries_pages'], f"{page_key}.json"), 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)

    # Also save a combined legacy text_boxes.json (unchanged)
    with open(os.path.join(output_dir, 'text_boxes.json'), 'w', encoding='utf-8') as f:
        json.dump(text_data, f, indent=2)

    # 3 & 4: whitebox + bbox detection + cropping (+ save cleaned bin masks)
    page_keys = sorted(text_data.keys())
    args_list = [(pk, dirs) for pk in page_keys]

    # decide how many workers
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 4)
    print(f"[PIPELINE] Running cropping with {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(crop_worker, args_list))

    # manifest
    manifest = {'pages': {}}
    for page_num, crops in results:
        manifest['pages'][page_num] = {'text': text_data[page_num], 'crops': crops}
    with open(os.path.join(output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

#!/usr/bin/env python3
"""
image_extraction.py

What changed (per your request):
- Whiteboxing now uses the **final compiled OCR file** (paddle_compiled.json).
  We read per-page items -> bbox_xyxy, convert to xywh, and whitebox those regions.
  If the compiled file is missing or a page has no items, we fall back to any
  existing text_boxes_pages/<page>.json (legacy entries) exactly as before.

Other notes (unchanged):
- Save-mode is strictly {all | final}.
- Mask image saved is the raw adaptive-mean mask (upsampled to page size if we detected at reduced scale).
- Speedups: downscale for mask/detect (1600 long side), vectorized whiteboxing, CC-with-stats, merge-only morphology.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local import (same folder)
import src.text_extraction as te

# ---------------- Global perf/threading limits ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# ---- Config ----
JPG_QUALITY = 95
CROP_MIN_SIDE = 1024

# Downscale-only knobs (fixed, not CLI)
MASK_DET_LIMIT_SIDE = 1600   # long side target for masking/detection only
MASK_MIN_AREA_ORIG = 1000    # (kept for compatibility; not used in new detector)
BBOX_PAD_SMALL = 3           # padding in SMALL-scale pixels before scaling up

# ---------------------------
# Low-level helpers
# ---------------------------

def imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def ensure_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    """Write JPEG only if file is missing."""
    if not os.path.exists(path):
        imwrite_jpg(path, img_bgr, quality)


def whitebox_text(page_img: np.ndarray, entries: List[Dict[str, Any]]) -> np.ndarray:
    """Fast text whiteboxing using NumPy slicing (fewer Python→C calls)."""
    H, W = page_img.shape[:2]
    out = page_img.copy()
    for e in entries:
        x, y, w, h = e.get('rect', [0, 0, 0, 0])
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(0, min(int(w), W - x))
        h = max(0, min(int(h), H - y))
        if w <= 0 or h <= 0:
            continue
        out[y:y+h, x:x+w] = 255
    return out


def adaptive_mean_mask(img_bgr: np.ndarray, block_size: int = 15, C: int = 2) -> np.ndarray:
    """Adaptive-mean thresholding to get figures=white (255), then *merge-only* morphology."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr  = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )
    # invert so figures are white for CC-based logic
    mask = thr

    # Merge-only morphology: one small dilation pass (no open/close/erode)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.dilate(mask, kernel3, iterations=1)
    return merged


# ---------------------------
# Detector (with gates + de-dup)
# ---------------------------

def clean_boxes(boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    """Remove near-duplicate boxes based on heavy overlap (≥90%)."""
    final: List[Tuple[int,int,int,int]] = []
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


def detect_bboxes_from_mask(mask: np.ndarray,
                            aspect_range: Tuple[float, float] = (0.3, 6.0),
                            morph_kernel: Tuple[int, int] = (5, 5),
                            morph_iterations: int = 1,
                            max_page_frac: float = 0.5) -> List[Tuple[int,int,int,int]]:
    """Connected-components on binary mask -> bounding boxes with thresholds."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if morph_kernel and morph_kernel[0] > 0 and morph_kernel[1] > 0 and morph_iterations > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(morph_kernel[0]), int(morph_kernel[1])))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(morph_iterations))

    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    page_area = H * W
    min_area = 0.01 * page_area
    max_area = max_page_frac * page_area
    lo_ar, hi_ar = max(aspect_range[0], 0.10), min(aspect_range[1], 10.0)

    out: List[Tuple[int,int,int,int]] = []
    for lbl in range(1, num_lbl):
        x  = int(stats[lbl, cv2.CC_STAT_LEFT])
        y  = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        if bw < 100 or bh < 100:
            continue
        area = bw * bh
        if area < min_area or area > max_area:
            continue

        ar = bw / float(max(1, bh))
        if ar < lo_ar or ar > hi_ar:
            continue

        out.append((x, y, bw, bh))

    return clean_boxes(out)


def save_crops(page_img: np.ndarray, boxes: List[Tuple[int, int, int, int]], out_dir: str, page_key: str, min_side: int = CROP_MIN_SIDE) -> List[Dict[str, Any]]:
    """Save crops for a page. Only write missing files. Return meta."""
    os.makedirs(out_dir, exist_ok=True)
    H, W = page_img.shape[:2]
    crops_meta: List[Dict[str, Any]] = []

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(0, min(int(w), W - x))
        h = max(0, min(int(h), H - y))
        if w <= 0 or h <= 0:
            continue

        crop = page_img[y:y+h, x:x+w]

        # Upscale small crops to a comfortable min-side size for downstream models
        scale = max(min_side / float(w), min_side / float(h), 1.0)
        if scale > 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        fname = f"{page_key}_crop{idx:02d}.jpg"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            imwrite_jpg(fpath, crop)

        crops_meta.append({'rect': [x, y, w, h], 'file': fname})

    return crops_meta


# ---------------------------
# Main per-page worker
# ---------------------------

def _process_one_page(
    pk: str,
    out_dir: str,
    dirs: Dict[str, str],
    save_mode: str,
    compiled_rects_xywh: Optional[Dict[str, List[List[int]]]] = None,  # NEW: whitebox from compiled OCR
) -> Tuple[str, List[Dict[str, Any]], List[List[int]], List[Dict[str, Any]], Dict[str, float]]:
    """Process a single page.

    Returns (pk, entries, boxes_xywh, crops_meta, timings).
    """

    # --- Timing buckets for this page ---
    _t_page_start = time.perf_counter()
    _timings: Dict[str, float] = {
        'image_load_s': 0.0,
        'adaptive_mask_s': 0.0,
        'bbox_cropping_s': 0.0,
        'total_page_s': 0.0,
    }

    # Paths
    masked_path    = os.path.join(dirs['masked'],  f"{pk}.jpg")
    mask_img_path  = os.path.join(dirs['mask'],    f"{pk}.jpg")
    overlay_path   = os.path.join(dirs['overlay'], f"{pk}.jpg")
    bbox_json_path = os.path.join(dirs['bbox_pages'], f"{pk}.json")
    page_img_path  = os.path.join(dirs['pages'],  f"{pk}.jpg")
    entries_json_path = os.path.join(dirs['entries_pages'], f"{pk}.json")

    # Load page image
    _t0 = time.perf_counter()
    page_img = cv2.imread(page_img_path)
    _timings['image_load_s'] += (time.perf_counter() - _t0)
    if page_img is None:
        _timings['total_page_s'] = (time.perf_counter() - _t_page_start)
        return pk, [], [], [], _timings

    # ---- Whiteboxing entries from compiled OCR (preferred) ----
    entries: List[Dict[str, Any]] = []
    used_compiled = False
    if compiled_rects_xywh and pk in compiled_rects_xywh:
        # compiled_rects_xywh[pk] is List[List[int]] as xywh
        for rect in compiled_rects_xywh[pk]:
            if len(rect) == 4:
                x, y, w, h = rect
                entries.append({'rect': [int(x), int(y), max(1, int(w)), max(1, int(h))]})
        used_compiled = len(entries) > 0

    # Fallback to legacy per-page entries if compiled missing/empty
    if not used_compiled and os.path.exists(entries_json_path):
        try:
            with open(entries_json_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        except Exception:
            entries = []

    # Whitebox for thresholding; optionally persist if save_mode='all'
    masked_img = whitebox_text(page_img, entries)
    if save_mode == 'all':
        ensure_jpg(masked_path, masked_img)

    # Prepare threshold source (whiteboxed)
    thresh_src = masked_img

    # --- Downscale for faster masking/detection (boxes scale back up) ---
    H, W = thresh_src.shape[:2]
    scale = 1.0
    if MASK_DET_LIMIT_SIDE and max(H, W) > MASK_DET_LIMIT_SIDE:
        scale = MASK_DET_LIMIT_SIDE / float(max(H, W))
        small_W = max(1, int(round(W * scale)))
        small_H = max(1, int(round(H * scale)))
        small_src = cv2.resize(thresh_src, (small_W, small_H), interpolation=cv2.INTER_AREA)
    else:
        small_src = thresh_src

    # Mask (adaptive mean) at small scale
    _t_mask = time.perf_counter()
    mask_small = adaptive_mean_mask(small_src)
    _timings['adaptive_mask_s'] += (time.perf_counter() - _t_mask)

    # Persist the raw mask image if requested
    if save_mode == 'all':
        vis_mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask_small
        imwrite_jpg(mask_img_path, cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR))

    # Detect boxes on small mask using stricter CC with gates + de-dup
    boxes_small = detect_bboxes_from_mask(mask_small)

    # Small-space padding before scaling back up (compensate for thresholding tightness)
    if BBOX_PAD_SMALL > 0 and boxes_small:
        SH, SW = mask_small.shape[:2]
        padded = []
        for (x, y, w, h) in boxes_small:
            x2 = max(0, x - BBOX_PAD_SMALL)
            y2 = max(0, y - BBOX_PAD_SMALL)
            w2 = min(SW - x2, w + 2 * BBOX_PAD_SMALL)
            h2 = min(SH - y2, h + 2 * BBOX_PAD_SMALL)
            padded.append((x2, y2, w2, h2))
        boxes_small = padded

    # Scale boxes back to original space
    if scale < 1.0:
        inv = 1.0 / scale
        boxes = []
        for (x, y, w, h) in boxes_small:
            X = int(round(x * inv))
            Y = int(round(y * inv))
            Wb = int(round(w * inv))
            Hb = int(round(h * inv))
            X = max(0, min(X, W - 1))
            Y = max(0, min(Y, H - 1))
            Wb = max(0, min(Wb, W - X))
            Hb = max(0, min(Hb, H - Y))
            if Wb > 0 and Hb > 0:
                boxes.append((X, Y, Wb, Hb))
    else:
        boxes = boxes_small

    boxes_xywh: List[List[int]] = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in boxes]

    # Save per-page bboxes JSON (always)
    with open(bbox_json_path, 'w', encoding='utf-8') as f:
        json.dump([{'rect': b} for b in boxes_xywh], f, indent=2)

    # Overlay image (only in 'all')
    crops_meta: List[Dict[str, Any]] = []
    if save_mode == 'all':
        base = page_img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(base, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imwrite_jpg(overlay_path, base)

    # Crops — saved in both modes; in 'final' these are the ONLY image files
    if boxes_xywh:
        _t_crop = time.perf_counter()
        for idx, (x, y, w, h) in enumerate(boxes_xywh, start=1):
            fname = f"{pk}_crop{idx:02d}.jpg"
            fpath = os.path.join(dirs['crops'], fname)
            if not os.path.exists(fpath):
                crop = page_img[y:y+h, x:x+w]
                scale_up = max(CROP_MIN_SIDE / float(w), CROP_MIN_SIDE / float(h), 1.0)
                if scale_up > 1.0:
                    new_w = int(round(w * scale_up))
                    new_h = int(round(h * scale_up))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                imwrite_jpg(fpath, crop)
            crops_meta.append({'rect': [x, y, w, h], 'file': fname})
        _timings['bbox_cropping_s'] += (time.perf_counter() - _t_crop)

    # Finalize per-page timings and return
    _timings['total_page_s'] = (time.perf_counter() - _t_page_start)
    return pk, entries, boxes_xywh, crops_meta, _timings


# ---------------------------
# Public entry (call from your main)
# ---------------------------

def run_image_extraction(
    pdf_path: str,
    out_dir: str,
    device: str = 'gpu',
    workers: Optional[int] = None,
    save_mode: str = 'final',  # 'all' or 'final'
) -> None:
    """Run full pipeline; callable from your main."""
    pdf_path = os.path.abspath(pdf_path)
    out_dir  = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # If prior text JSONs exist, reuse; else run text extraction.
    struct_path = os.path.join(out_dir, 'text_struct.json')
    boxes_path  = os.path.join(out_dir, 'text_boxes.json')

    # --- Timing: text extraction (includes page renders & OCR) ---
    _t_text = time.perf_counter()
    _text_extracted = False
    if os.path.exists(struct_path) and os.path.exists(boxes_path):
        try:
            with open(struct_path, 'r', encoding='utf-8') as f:
                unified_struct = json.load(f)
            print('[IMAGES] Using existing text_struct.json/text_boxes.json — skipping text extraction')
        except Exception:
            unified_struct = te.run_text_extraction(pdf_path, out_dir, device=device)
            _text_extracted = True
    else:
        unified_struct = te.run_text_extraction(pdf_path, out_dir, device=device)
        _text_extracted = True
    _text_extract_s = (time.perf_counter() - _t_text)
    print(f"[TIMING] text extraction: {_text_extract_s:.3f}s (ran={_text_extracted})")

    # ---- Load final compiled OCR (preferred for whiteboxing) ----
    compiled_path = os.path.join(out_dir, 'paddle_compiled.json')
    compiled_rects_xywh: Dict[str, List[List[int]]] = {}
    if os.path.exists(compiled_path):
        try:
            with open(compiled_path, 'r', encoding='utf-8') as f:
                compiled = json.load(f)
            # compiled is {page_key: {"items":[{"bbox_xyxy":[x1,y1,x2,y2], "text":..., "score":...}], ...}, ...}
            for pk, payload in compiled.items():
                items = payload.get('items') or []
                rects_xywh: List[List[int]] = []
                for it in items:
                    b = it.get('bbox_xyxy')
                    if isinstance(b, list) and len(b) == 4:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        w = max(1, int(x2 - x1))
                        h = max(1, int(y2 - y1))
                        rects_xywh.append([x1, y1, w, h])
                if rects_xywh:
                    compiled_rects_xywh[pk] = rects_xywh
            print(f"[IMAGES] Using paddle_compiled.json for whiteboxing ({len(compiled_rects_xywh)} pages)")
        except Exception as e:
            print(f"[IMAGES] Failed to read paddle_compiled.json ({e}); falling back to legacy entries.")
    else:
        print("[IMAGES] No paddle_compiled.json found; whiteboxing will fall back to legacy entries if present.")

    # Prepare dirs
    dirs = {
        'pages': os.path.join(out_dir, 'page_images'),
        'masked': os.path.join(out_dir, 'page_masked'),
        'mask': os.path.join(out_dir, 'page_mask'),
        'overlay': os.path.join(out_dir, 'page_bbox_overlay'),
        'crops': os.path.join(out_dir, 'bbox_crops'),
        'entries_pages': os.path.join(out_dir, 'text_boxes_pages'),
        'bbox_pages': os.path.join(out_dir, 'bbox_pages'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    combined_bboxes: Dict[str, List[List[int]]] = {}
    manifest: Dict[str, Any] = {'pages': {}}

    page_keys = sorted(unified_struct.keys())

    # Decide workers
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 2)

    # Parallel per-page processing
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _process_one_page,
                pk,
                out_dir,
                dirs,
                save_mode,
                compiled_rects_xywh,  # NEW: pass compiled rects map
            )
            for pk in page_keys
        ]

        for fut in as_completed(futures):
            pk, entries, boxes_xywh, crops_meta, _timings = fut.result()
            if not pk:
                continue
            combined_bboxes[pk] = boxes_xywh
            manifest['pages'][pk] = {
                'text': entries,
                'bboxes': boxes_xywh,
                'crops': crops_meta,
                'timings': _timings,
            }

            print(
                f"[TIMING][{pk}] image download/load: {_timings.get('image_load_s', 0.0):.3f}s | "
                f"adaptive means masking: {_timings.get('adaptive_mask_s', 0.0):.3f}s | "
                f"bbox cropping: {_timings.get('bbox_cropping_s', 0.0):.3f}s | "
                f"total page: {_timings.get('total_page_s', 0.0):.3f}s"
            )

            # Aggregate timings
            if '_timing_totals' not in locals():
                _timing_totals = {'image_load_s': 0.0, 'adaptive_mask_s': 0.0, 'bbox_cropping_s': 0.0, 'total_page_s': 0.0}
                _timing_count = 0
            _timing_totals['image_load_s'] += _timings.get('image_load_s', 0.0)
            _timing_totals['adaptive_mask_s'] += _timings.get('adaptive_mask_s', 0.0)
            _timing_totals['bbox_cropping_s'] += _timings.get('bbox_cropping_s', 0.0)
            _timing_totals['total_page_s'] += _timings.get('total_page_s', 0.0)
            _timing_count += 1

    # Timing summary
    if '_timing_count' in locals() and _timing_count > 0:
        _avg = {k: (v / _timing_count) for k, v in _timing_totals.items()}
        print(f"[TIMING] per-page averages over {_timing_count} pages — "
              f"image download/load: {_avg['image_load_s']:.3f}s | "
              f"adaptive means masking: {_avg['adaptive_mask_s']:.3f}s | "
              f"bbox cropping: {_avg['bbox_cropping_s']:.3f}s | "
              f"total page: {_avg['total_page_s']:.3f}s")
        manifest['timings'] = {
            'text_extraction_s': _text_extract_s if '_text_extract_s' in locals() else 0.0,
            'per_page_totals': _timing_totals,
            'per_page_averages': _avg,
            'pages_count': _timing_count,
        }
    else:
        manifest['timings'] = {
            'text_extraction_s': _text_extract_s if '_text_extract_s' in locals() else 0.0,
            'per_page_totals': {},
            'per_page_averages': {},
            'pages_count': 0,
        }

    # Save combined bboxes + manifest (overwrite is OK)
    with open(os.path.join(out_dir, 'bboxes.json'), 'w', encoding='utf-8') as f:
        json.dump(combined_bboxes, f, indent=2)
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[IMAGES] Done: pages={len(page_keys)} workers={workers} "
        f"save_mode={save_mode} crops=on"
    )

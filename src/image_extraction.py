#!/usr/bin/env python3
"""
image_extraction.py

Purpose
-------
Extract figure regions from a textbook PDF's rendered pages and save crops. Use
compiled OCR boxes for text whiteboxing to avoid masking figures, then detect
figure boxes via connected components on an adaptive-mean mask.

Flow
----
1) If final JSONs and output folders exist, reuse cached outputs.
2) Ensure text JSONs exist:
   - If out_dir/text_struct.json and out_dir/text_boxes.json exist, reuse them.
   - Else call te.run_text_extraction(...) to produce text structure.
3) Load compiled OCR rectangles:
   - Prefer out_dir/paddle_compiled.json (xyxy → xywh per page).
   - If missing for a page, fall back to legacy out_dir/text_boxes_pages/pageNNN.json.
4) Prepare output folders:
   - page_images/, page_masked/, page_mask/, page_bbox_overlay/, bbox_crops/,
     text_boxes_pages/, bbox_pages/
5) For each page (threaded):
   a) Load page image
   b) Whitebox text regions (from compiled rectangles) with fast NumPy slicing
   c) Downscale to MASK_DET_LIMIT_SIDE for speed
   d) Build mask with adaptive mean (binary INV) and one small dilation
   e) Connected components with gates and de-dup to get boxes
   f) Scale boxes back to original space and save per-page JSON
   g) Save overlay (only in save_mode="all") and write crops (always)
   h) Record timings
6) Save:
   - out_dir/bboxes.json         : {page_key: [[x,y,w,h], ...]}
   - out_dir/manifest.json       : per-page timings, crop manifest, averages

Outputs
-------
- bbox_crops/         : cropped figures (optionally upscaled to min side)
- page_bbox_overlay/  : drawn boxes on page (save_mode="all" only)
- page_mask/          : raw adaptive mask images (save_mode="all" only)
- page_masked/        : whiteboxed pages (save_mode="all" only)
- bbox_pages/         : per-page boxes JSON as [{'rect':[x,y,w,h]}, ...]
- bboxes.json         : combined boxes across pages
- manifest.json       : crops, timings, counts

Public API
----------
    run_image_extraction(pdf_path, out_dir, device='gpu', workers=None, save_mode='final') -> None

Notes
-----
- Existing bboxes.json, manifest.json, bbox_pages/, and bbox_crops/ are treated as complete stage outputs.
- Whiteboxing prefers paddle_compiled.json; legacy per-page entries are a fallback.
- Adaptive threshold is inverted so figures are white for CC detection.
- Crops are upscaled so the shorter side is at least CROP_MIN_SIDE.
- CPU BLAS threads are pinned to 1; OpenCV threads disabled.
- save_mode='final' writes only crops and JSONs; 'all' also writes masks/overlays.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project import
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
CROP_EXT = ".jpg"

JPG_QUALITY = 95
CROP_MIN_SIDE = 1024

# Downscale-only knobs (fixed, not CLI)
MASK_DET_LIMIT_SIDE = 1600   # long side target for masking/detection only
MASK_MIN_AREA_ORIG = 1000    # (kept for compatibility; not used in new detector)
BBOX_PAD_SMALL = 3           # padding in SMALL-scale pixels before scaling up

# ---------------------------
# Low-level helpers
# ---------------------------

def _page_num_from_key(pk: str) -> int:
    """
    Extract trailing integer page number from a page key like 'page0001' or '0001'.
    Falls back to 0 if not found.
    """
    m = re.search(r'(\d+)$', pk)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    try:
        return int(pk)
    except ValueError:
        return 0

def crop_name(pk: str, idx: int, ext: str = CROP_EXT) -> str:
    """
    Format: 0000_000.ext  -> page_crop with zero-padding
    Example: page 1, crop 2 -> 0001_002.jpg
    """
    return f"{_page_num_from_key(pk):04d}_{idx:03d}{ext}"

def imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def ensure_jpg(path: str, img_bgr: np.ndarray, quality: int = JPG_QUALITY) -> None:
    """Write JPEG only if file is missing."""
    if not os.path.exists(path):
        imwrite_jpg(path, img_bgr, quality)


def whitebox_text(page_img: np.ndarray, entries: List[Dict[str, Any]]) -> np.ndarray:
    """Whitebox rectangular text regions using fast array slicing."""
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
    """Adaptive-mean → invert (figures=white) → one small dilation to merge fragments."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr  = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )
    # figures white for CC
    mask = thr

    # Merge-only morphology
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.dilate(mask, kernel3, iterations=1)
    return merged


# ---------------------------
# Detector (with gates + de-dup)
# ---------------------------

def clean_boxes(boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    """Drop near-duplicates if IoU with an already-kept box ≥90%."""
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
    """Connected-components on mask → boxes filtered by size, area%, and aspect ratio."""
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
    """Write crops for a page, upscaling small ones to a minimum side length."""
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

        # Min-side upscale for downstream models
        scale = max(min_side / float(w), min_side / float(h), 1.0)
        if scale > 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        fname = crop_name(page_key, idx)
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
    compiled_rects_xywh: Optional[Dict[str, List[List[int]]]] = None,
) -> Tuple[str, List[Dict[str, Any]], List[List[int]], List[Dict[str, Any]], Dict[str, float]]:
    """
    Process one page:
      1) load page image
      2) load entries (text boxes) -> whitebox
      3) adaptive-mean threshold -> mask_small (inverted)
      4) detect bboxes -> boxes_xywh
      5) persist artifacts without re-doing work if files already exist
    """
    t0_all = time.perf_counter()
    timings: Dict[str, float] = {
        "image_load_s": 0.0,
        "adaptive_mask_s": 0.0,
        "bbox_cropping_s": 0.0,
        "total_page_s": 0.0,
    }

    # Paths
    page_img_path      = os.path.join(dirs["pages"],         f"{pk}.jpg")
    entries_json_path  = os.path.join(dirs["entries_pages"], f"{pk}.json")
    masked_path        = os.path.join(dirs["masked"],        f"{pk}.jpg")
    mask_img_path      = os.path.join(dirs["mask"],          f"{pk}.jpg")
    overlay_path       = os.path.join(dirs["overlay"],       f"{pk}.jpg")
    bbox_json_path     = os.path.join(dirs["bbox_pages"],    f"{pk}.json")

    # Fast skip when we already have boxes and the source page image
    if os.path.exists(bbox_json_path) and os.path.exists(page_img_path):
        try:
            with open(bbox_json_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            boxes_xywh = [e["rect"] for e in prev if isinstance(e, dict) and "rect" in e]
            timings["total_page_s"] = time.perf_counter() - t0_all
            return pk, [], boxes_xywh, [], timings
        except Exception:
            pass

    # Load page image
    t0 = time.perf_counter()
    page_img = cv2.imread(page_img_path)
    timings["image_load_s"] += time.perf_counter() - t0
    if page_img is None:
        timings["total_page_s"] = time.perf_counter() - t0_all
        return pk, [], [], [], timings

    H, W = page_img.shape[:2]

    # ---------- entries: load from per-page json or compiled dict ----------
    # Expected format per entry: {"rect": [x, y, w, h], ...}
    entries: List[Dict[str, Any]] = []

    if os.path.exists(entries_json_path):
        try:
            with open(entries_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept either a list[{"rect":[...]}] or {"entries":[...]}
            if isinstance(data, dict) and "entries" in data:
                entries = [e for e in data["entries"] if isinstance(e, dict) and "rect" in e]
            elif isinstance(data, list):
                entries = [e for e in data if isinstance(e, dict) and "rect" in e]
        except Exception:
            entries = []
    elif compiled_rects_xywh and pk in compiled_rects_xywh:
        entries = [{"rect": rect} for rect in compiled_rects_xywh[pk] if isinstance(rect, (list, tuple)) and len(rect) == 4]

    # ---------- whitebox using entries ----------
    # Uses your existing whitebox_text(page_img, entries) if present.
    # Fallback to a minimal in-place whitebox if whitebox_text is undefined.
    def _fallback_whitebox(img: np.ndarray, _entries: List[Dict[str, Any]]) -> np.ndarray:
        out = img.copy()
        for e in _entries:
            x, y, w, h = map(int, e["rect"])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = 255
        return out

    try:
        masked_img = whitebox_text(page_img, entries)  # type: ignore[name-defined]
    except NameError:
        masked_img = _fallback_whitebox(page_img, entries)

    if save_mode == "all" and not os.path.exists(masked_path):
        imwrite_jpg(masked_path, masked_img)

    # ---------- adaptive-mean threshold (inverted) to get mask_small ----------
    # Downscale for speed, remember the scale to re-map boxes.
    MAX_SIDE = 2000
    scale = 1.0
    h, w = masked_img.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / float(longest)
        masked_small = cv2.resize(masked_img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        masked_small = masked_img

    t0 = time.perf_counter()
    gray = cv2.cvtColor(masked_small, cv2.COLOR_BGR2GRAY) if masked_small.ndim == 3 else masked_small
    # Inverted binary so text/edges are white, background black
    mask_small = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=10,
    )
    timings["adaptive_mask_s"] += time.perf_counter() - t0

    if save_mode == "all" and not os.path.exists(mask_img_path):
        vis_mask = mask_small if scale == 1.0 else cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        imwrite_jpg(mask_img_path, cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR))

    # ---------- detect bboxes on mask_small and map back to full-res ----------
    # Uses your existing detect_bboxes_from_mask; falls back to a simple CC if missing.
    def _fallback_detect(mask_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 100 and h >= 100:
                out.append((x, y, w, h))
        return out

    try:
        boxes_small = detect_bboxes_from_mask(  # type: ignore[name-defined]
            mask_small,
            aspect_range=(0.3, 6.0),
            morph_kernel=(5, 5),
            morph_iterations=1,
            max_page_frac=0.5,
        )
    except NameError:
        boxes_small = _fallback_detect(mask_small)

    # Map to full-resolution coordinates
    inv = 1.0 / scale
    boxes_xywh: List[List[int]] = []
    for (x, y, w, h) in boxes_small:
        if scale != 1.0:
            X = int(round(x * inv)); Y = int(round(y * inv))
            Wb = int(round(w * inv)); Hb = int(round(h * inv))
        else:
            X, Y, Wb, Hb = int(x), int(y), int(w), int(h)
        # clamp
        X = max(0, min(X, W - 1)); Y = max(0, min(Y, H - 1))
        Wb = max(1, min(Wb, W - X)); Hb = max(1, min(Hb, H - Y))
        boxes_xywh.append([X, Y, Wb, Hb])

    # ---------- persist per-page bbox json ----------
    with open(bbox_json_path, "w", encoding="utf-8") as f:
        json.dump([{"rect": b} for b in boxes_xywh], f, indent=2)

    # ---------- overlay (only if absent and save_mode == 'all') ----------
    if save_mode == "all" and not os.path.exists(overlay_path):
        base = page_img.copy()
        for (x, y, w, h) in boxes_xywh:
            cv2.rectangle(base, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imwrite_jpg(overlay_path, base)

    # ---------- crops (skip if file exists) ----------
    crops_meta: List[Dict[str, Any]] = []
    if boxes_xywh:
        t0 = time.perf_counter()
        CROP_MIN_SIDE = globals().get("CROP_MIN_SIDE", 512)
        for idx, (x, y, w, h) in enumerate(boxes_xywh, start=1):
            fname = crop_name(pk, idx)
            fpath = os.path.join(dirs["crops"], fname)
            if not os.path.exists(fpath):
                crop = page_img[y:y + h, x:x + w]
                scale_up = max(CROP_MIN_SIDE / float(w), CROP_MIN_SIDE / float(h), 1.0)
                if scale_up > 1.0:
                    new_w = int(round(w * scale_up))
                    new_h = int(round(h * scale_up))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                imwrite_jpg(fpath, crop)
            crops_meta.append({"rect": [x, y, w, h], "file": fname})
        timings["bbox_cropping_s"] += time.perf_counter() - t0

    timings["total_page_s"] = time.perf_counter() - t0_all
    return pk, entries, boxes_xywh, crops_meta, timings


# ---------------------------
# Public entry
# ---------------------------

def run_image_extraction(
    pdf_path: str,
    out_dir: str,
    device: str = "gpu",
    workers: Optional[int] = None,
    save_mode: str = "final",
) -> Dict[str, Any]:
    """
    Image pass (assumes text pass already writes entries):
      a) Ensure text extraction bundle is available; read per-page structure.
      b) Whitebox using <out_dir>/text_boxes_pages/pageNNN.json (or compiled_rects_xywh if present).
      c) Adaptive-mean mask (in-memory B/W) -> bbox detection.
      d) Persist per-page and combined outputs, skipping files that already exist.

    Returns a small manifest dict. Artifacts are written to disk.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pdf_path = os.path.abspath(pdf_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    bboxes_path = os.path.join(out_dir, "bboxes.json")
    manifest_path = os.path.join(out_dir, "manifest.json")
    bbox_pages_dir = os.path.join(out_dir, "bbox_pages")
    crops_dir = os.path.join(out_dir, "bbox_crops")
    if (
        os.path.exists(bboxes_path)
        and os.path.exists(manifest_path)
        and os.path.isdir(bbox_pages_dir)
        and os.path.isdir(crops_dir)
    ):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            pages_count = int((manifest.get("timings") or {}).get("pages_count") or len(manifest.get("pages", {})))
        except Exception:
            pages_count = 0
        print(f"[IMAGES] Using cached outputs: {bboxes_path} and {manifest_path}")
        return {
            "bboxes_path": bboxes_path,
            "manifest_path": manifest_path,
            "pages": pages_count,
            "workers": workers,
            "source": "cached",
        }

    # ---- Step 1+2: make sure text extraction ran and load its results ----
    _t_text = time.perf_counter()

    struct_path = os.path.join(out_dir, "text_structure.json")
    entries_dir = os.path.join(out_dir, "text_boxes_pages")

    # Call the new text-extraction entrypoint which returns a bundle.
    try:
        bundle = te.run_text_extraction(pdf_path, out_dir, device=device)
    except Exception:
        bundle = None

    # Resolve structure from bundle or legacy file if needed.
    if isinstance(bundle, dict) and isinstance(bundle.get("structure"), dict):
        structure = bundle["structure"]
    else:
        with open(struct_path, "r", encoding="utf-8") as f:
            structure = json.load(f)

    # Basic sanity: ensure entries exist (text pass should have written them).
    if not os.path.isdir(entries_dir) or not os.listdir(entries_dir):
        raise FileNotFoundError(
            f"Missing per-page entries in {entries_dir}. "
            f"Text extraction must create text_boxes_pages/pageNNN.json."
        )

    _text_extract_s = time.perf_counter() - _t_text

    # ---- Step 3: set up IO directories for image artifacts ----
    dirs = {
        "pages":   os.path.join(out_dir, "page_images"),
        "masked":  os.path.join(out_dir, "page_masked"),
        "mask":    os.path.join(out_dir, "page_mask"),
        "overlay": os.path.join(out_dir, "page_bbox_overlay"),
        "crops":   os.path.join(out_dir, "bbox_crops"),
        "entries_pages": entries_dir,
        "bbox_pages": os.path.join(out_dir, "bbox_pages"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Optional compiled rectangles (e.g., from Paddle structure), used for whiteboxing if present.
    compiled_rects_xywh: Dict[str, List[List[int]]] = {}
    compiled_path = os.path.join(out_dir, "paddle_compiled.json")
    if os.path.exists(compiled_path):
        try:
            with open(compiled_path, "r", encoding="utf-8") as f:
                compiled = json.load(f)
            for pk, payload in compiled.items():
                rects = []
                for it in payload.get("items", []):
                    b = it.get("bbox_xyxy")
                    if isinstance(b, list) and len(b) == 4:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        rects.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
                if rects:
                    compiled_rects_xywh[pk] = rects
            print(f"[IMAGES] Using paddle_compiled.json for whiteboxing ({len(compiled_rects_xywh)} pages)")
        except Exception as e:
            print(f"[IMAGES] Failed to parse paddle_compiled.json: {e}")

    # ---- Step 4: run per-page processing (whitebox -> adaptive mean -> bboxes) ----
    page_keys = sorted(structure.keys())
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 2)

    combined_bboxes: Dict[str, List[List[int]]] = {}
    manifest: Dict[str, Any] = {"pages": {}}

    _tot = {"image_load_s": 0.0, "adaptive_mask_s": 0.0, "bbox_cropping_s": 0.0, "total_page_s": 0.0}
    _n = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_process_one_page, pk, out_dir, dirs, save_mode, compiled_rects_xywh)
            for pk in page_keys
        ]
        for fut in as_completed(futs):
            pk, entries, boxes_xywh, crops_meta, t = fut.result()
            combined_bboxes[pk] = boxes_xywh
            manifest["pages"][pk] = {
                "text": entries,          # entries used to whitebox this page
                "bboxes": boxes_xywh,     # final xywh bboxes for figures/photos
                "crops": crops_meta,      # saved crop files metadata
                "timings": t,
            }
            _tot["image_load_s"] += t.get("image_load_s", 0.0)
            _tot["adaptive_mask_s"] += t.get("adaptive_mask_s", 0.0)
            _tot["bbox_cropping_s"] += t.get("bbox_cropping_s", 0.0)
            _tot["total_page_s"] += t.get("total_page_s", 0.0)
            _n += 1

    manifest["timings"] = {
        "text_extraction_s": _text_extract_s,
        "per_page_totals": _tot,
        "per_page_averages": {k: (_tot[k] / _n if _n else 0.0) for k in _tot},
        "pages_count": _n,
        "workers": workers,
        "save_mode": save_mode,
        "device": device,
    }

    # ---- Step 5: persist combined outputs ----
    with open(bboxes_path, "w", encoding="utf-8") as f:
        json.dump(combined_bboxes, f, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[IMAGES] Done: pages={len(page_keys)} workers={workers} save_mode={save_mode}")

    return {
        "bboxes_path": bboxes_path,
        "manifest_path": manifest_path,
        "pages": len(page_keys),
        "workers": workers,
    }

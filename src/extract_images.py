import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Optional
from paddleocr import PaddleOCR
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ---- Render / scaling config (shared across functions) ----
RENDER_DPI = 300  # must match render_pages() so whiteboxes align with page images

BBox = Tuple[int, int, int, int]

# ----------------------------------------------------------------------------
# UTILITY FUNCTIONS (with logging)
# ----------------------------------------------------------------------------
def clean_boxes(boxes: List[BBox]) -> List[BBox]:
    """Remove near-duplicate bounding boxes based on overlap thresholds."""
    final: List[BBox] = []
    for x, y, w, h in boxes:
        new_area = w * h
        skip = False
        for ox, oy, ow, oh in final:
            # compute intersection
            xi, yi = max(x, ox), max(y, oy)
            xj, yj = min(x + w, ox + ow), min(y + h, oy + oh)
            if xj > xi and yj > yi:
                inter = (xj - xi) * (yj - yi)
                # skip if overlap > 90%
                if inter / new_area > 0.9 or inter / (ow * oh) > 0.9:
                    skip = True
                    break
        if not skip:
            final.append((x, y, w, h))
    return final

# assumes: from typing import List, Tuple
# BBox = Tuple[int, int, int, int]
def find_bounding_boxes(
    img: np.ndarray,
    min_pixel_area: int = 50,                        # CC speckle removal floor (px) — used to clean the mask
    min_box_size: Tuple[int, int] = (30, 30),      # ABSOLUTE bbox side floors (w, h)
    aspect_range: Tuple[float, float] = (0.3, 6.0),  # slightly tighter than (0.2, 10.0)
    block_size: int = 15,                            # must be odd
    C: int = 2,
    bridge_px: int = 0,                              # 0 = no morphology

    # STRICT relative area gate (e.g., 0.0025 = 0.25% of page)
    min_area_frac: float = 0.0025,

    # Optional: save cleaned mask (AFTER speckle removal)
    save_mask_dir: Optional[str] = None,
    save_mask_name: Optional[str] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Pipeline: adaptive-mean threshold (invert) → optional bridge → CC speckle cleanup (min_pixel_area)
              → [optional save mask] → CC stats on CLEAN mask → ABS bbox size → REL area → oversize → aspect → dedupe.
    Returns (x, y, w, h).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bs = block_size if (block_size % 2 == 1) else block_size + 1
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        bs, C
    )

    if bridge_px and bridge_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, bridge_px))
        mask = cv2.dilate(mask, k, iterations=1)

    # --- Speckle cleanup on mask (remove CCs smaller than min_pixel_area) ---
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lbl in range(1, num_lbl):
        if int(stats[lbl, cv2.CC_STAT_AREA]) < min_pixel_area:
            mask[labels == lbl] = 0

    # --- Save the CLEANED binary mask (after speckle removal) ---
    if save_mask_dir and save_mask_name:
        os.makedirs(save_mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_mask_dir, f"{save_mask_name}.png"), mask)

    # Recompute CCs on the CLEAN mask for bbox stats
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = gray.shape
    page_area = H * W
    rel_area_floor = max(1, int(min_area_frac * page_area))

    lo_ar, hi_ar = aspect_range
    lo_ar = max(lo_ar, 0.3)
    hi_ar = min(hi_ar, 6.0)

    raw_boxes: List[Tuple[int, int, int, int]] = []
    for lbl in range(1, num_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        x  = int(stats[lbl, cv2.CC_STAT_LEFT])
        y  = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        # 1) ABSOLUTE bbox pixel size (area + side lengths)
        if bw < min_box_size[0] or bh < min_box_size[1]:
            continue

        # 2) RELATIVE area (strict)
        if area < rel_area_floor:
            continue

        # 3) Oversize guard
        if bw * bh > 0.75 * page_area:
            continue

        # 4) Aspect ratio
        ar = bw / float(bh)
        if ar < lo_ar or ar > hi_ar:
            continue

        raw_boxes.append((x, y, bw, bh))

    final_boxes = clean_boxes(raw_boxes)
    return final_boxes

def save_resized_crop(
    page_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_path: str,
    min_side: int = 1024
) -> None:
    """
    Save a crop resized so BOTH dimensions are >= min_side (keeps aspect ratio).
    """
    x, y, w, h = map(int, bbox)
    H, W = page_img.shape[:2]

    # Clamp bbox to image boundaries
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    crop = page_img[y:y + h, x:x + w].copy()

    # Scale so both dims >= min_side
    scale = max(min_side / float(w), min_side / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Lanczos upsampling for best quality
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Ensure folder exists and save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, resized)
# ----------------------------------------------------------------------------
# PYMU PDF TEXT EXTRACTION (PRIMARY)
# ----------------------------------------------------------------------------
def extract_text_boxes_pymupdf(pdf_path: str) -> Dict[str, List[Dict]]:
    """
    Extract text spans via PyMuPDF and SCALE their rectangles from PDF space (points)
    to pixel space to match images rendered at RENDER_DPI.
    """
    doc = fitz.open(pdf_path)
    result: Dict[str, List[Dict]] = {}
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_key = f"page{page_index+1:03d}"
        entries: List[Dict] = []

        # Compute scale factors that match render_pages() exactly
        pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
        px_w, px_h = pix.width, pix.height
        rect = page.rect  # points (72 dpi)
        scale_x = px_w / float(rect.width if rect.width != 0 else 1.0)
        scale_y = px_h / float(rect.height if rect.height != 0 else 1.0)

        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type", 1) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    x0, y0, x1, y1 = span["bbox"]
                    # Scale to pixel coordinates
                    sx0 = int(round(x0 * scale_x))
                    sy0 = int(round(y0 * scale_y))
                    sx1 = int(round(x1 * scale_x))
                    sy1 = int(round(y1 * scale_y))
                    w = max(0, sx1 - sx0)
                    h = max(0, sy1 - sy0)
                    # Clamp to page image bounds (paranoia)
                    sx0 = max(0, min(sx0, px_w - 1))
                    sy0 = max(0, min(sy0, px_h - 1))
                    w = max(0, min(w, px_w - sx0))
                    h = max(0, min(h, px_h - sy0))
                    entries.append({
                        'rect': [sx0, sy0, w, h],
                        'text': span.get('text', '')
                    })
        result[page_key] = entries
    return result

# ----------------------------------------------------------------------------
# PADDLE OCR FALLBACK
# ----------------------------------------------------------------------------
def run_ocr_fallback(pages: List[str], image_dir: str) -> Dict[str, List[Dict]]:
    """Fallback to PaddleOCR.predict when PyMuPDF yields no text, with inline B/W preprocessing and filtering."""
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )
    root_dir = os.path.dirname(image_dir)
    ocr_img_dir = os.path.join(root_dir, 'ocr_images')
    ocr_json_dir = os.path.join(root_dir, 'ocr_json')
    os.makedirs(ocr_img_dir, exist_ok=True)
    os.makedirs(ocr_json_dir, exist_ok=True)

    fallback: Dict[str, List[Dict]] = {}
    for page_key in pages:
        print(f"[OCR_FALLBACK] Processing {page_key} with PaddleOCR.predict")
        img_path = os.path.join(image_dir, f"{page_key}.png")
        orig_img = cv2.imread(img_path)
        # Inline black-and-white preprocessing
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proc_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        ocr_result = ocr.predict(proc_img)[0]
        data = ocr_result.json.get('res', ocr_result.json)
        rec_boxes = data.get('rec_boxes', [])
        rec_texts = data.get('rec_texts', [])
        # Save raw JSON for reference
        json_path = os.path.join(ocr_json_dir, f"{page_key}.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(data, jf, indent=2)
        # Save reference overlays as page_key.png
        try:
            pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            for box in rec_boxes:
                x0, y0, x1, y1 = map(int, box)
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            overlay_path = os.path.join(ocr_img_dir, f"{page_key}.png")
            pil_img.save(overlay_path)
            print(f"[OCR_FALLBACK] Saved overlay image to {overlay_path}")
        except Exception as e:
            print(f"[OCR_FALLBACK] Warning: could not save overlay for {page_key}: {e}")
        entries: List[Dict] = []
        for box, text in zip(rec_boxes, rec_texts):
            if not text or len(text.strip()) <= 1:
                continue
            x0, y0, x1, y1 = map(int, box)
            entries.append({
                'rect': [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                'text': text
            })
        print(f"[OCR_FALLBACK] {page_key}: detected {len(entries)} entries")
        fallback[page_key] = entries
    return fallback

# ----------------------------------------------------------------------------
# RENDER PAGES AS IMAGES
# ----------------------------------------------------------------------------
def render_pages(pdf_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        out_path = os.path.join(out_dir, f"page{i:03d}.png")
        if not os.path.exists(out_path):
            pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out_path)

# ----------------------------------------------------------------------------
# CROP WORKER (WHITEBOX + BBOX DETECTION + MASK SAVE)
# ----------------------------------------------------------------------------
def crop_worker(args):
    page_key, entries, dirs = args
    page_img = cv2.cvtColor(np.array(Image.open(os.path.join(dirs['pages'], f"{page_key}.png"))), cv2.COLOR_RGB2BGR)
    H, W = page_img.shape[:2]

    # 1) Whitebox text
    masked = page_img.copy()
    for e in entries:
        x, y, w, h = e['rect']
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(0, min(w, W - x))
        h = max(0, min(h, H - y))
        if w > 0 and h > 0:
            cv2.rectangle(masked, (x, y), (x + w, y + h), (255, 255, 255), -1)

    Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)).save(os.path.join(dirs['mask'], f"{page_key}.png"))

    # 2) Find figure boxes AND save the cleaned binary mask (inside find_bounding_boxes)
    final = find_bounding_boxes(
        masked,
        save_mask_dir=dirs['mask_bin'],   # <- new: binary mask goes here
        save_mask_name=page_key           # <- new: filename stem
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

        # Scale so BOTH dimensions are at least 1024
        scale = max(1024.0 / float(w), 1024.0 / float(h), 1.0)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if scale > 1.0:
            crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        fname = f"{page_key}_crop{idx:02d}.png"
        Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)).save(os.path.join(dirs['crops'], fname))
        crops.append({'rect': [x, y, w, h], 'file': fname})

    return page_key, crops

# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def extract_images_pipeline(pdf_path: str, output_dir: str, num_workers: int = None):
    # prepare dirs
    dirs = {
        'pages': os.path.join(output_dir, 'page_images'),
        'mask': os.path.join(output_dir, 'page_masked'),       # whiteboxed pages (RGB)
        'mask_bin': os.path.join(output_dir, 'page_mask_bin'), # cleaned thresh masks (binary)
        'crops': os.path.join(output_dir, 'bbox_crops'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1. render pages
    render_pages(pdf_path, dirs["pages"])

    # 2. extract text via PyMuPDF (now scaled to pixel coords)
    text_data = extract_text_boxes_pymupdf(pdf_path)

    # 2a. fallback to OCR only if PyMuPDF returned no text
    if all(len(v) == 0 for v in text_data.values()):
        print("[PIPELINE] PyMuPDF returned no text; running OCR fallback for all pages")
        ocr_data = run_ocr_fallback(list(text_data.keys()), dirs['pages'])
        text_data.update(ocr_data)
    else:
        print("[PIPELINE] PyMuPDF found text; skipping OCR fallback")

    # save unified text_boxes.json
    json_path = os.path.join(output_dir, 'text_boxes.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(text_data, f, indent=2)

    # 3 & 4: whitebox + bbox detection + cropping (+ save cleaned bin masks)
    args_list = [(page_num, text_data[page_num], dirs) for page_num in sorted(text_data)]

    # decide how many workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"[PIPELINE] Running cropping with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(crop_worker, args_list))

    # manifest
    manifest = {'pages': {}}
    for page_num, crops in results:
        manifest['pages'][page_num] = {'text': text_data[page_num], 'crops': crops}
    with open(os.path.join(output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

#!/usr/bin/env python3
"""
ocr_struct_helper.py

Runs PaddleOCR in a separate process to build the SAME rich text structure
(blocks → paragraphs → lines → spans) for specific pages. This isolates
Paddle from any Torch imports in the main process.

Behavior guarantees:
  • Always writes --out-json (even `{}` on failure) so caller never crashes.
  • Accepts --device {cpu,gpu}. Using GPU here is safe (separate process).

Usage:
  python ocr_struct_helper.py \
    --pages-dir /path/to/output/page_images \
    --pages-file /path/to/need_ocr_pages.txt \
    --out-json /path/to/ocr_struct.json \
    --device gpu \
    [--save-debug]

Writes:
  - out-json: JSON mapping { "pageNNN": {page_size, blocks, lines_flat, spans_flat, entries}, ... }
  - (optional) debug overlays in <pages-dir>/../ocr_images
  - (optional) raw OCR JSON in <pages-dir>/../ocr_json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
import statistics

# keep threads light
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw

RENDER_DPI = 300

# ---------------- basic utils ----------------
def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def imwrite_jpg(path: str, img_bgr: np.ndarray, quality: int = 95) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

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

def _group_lines_into_paragraphs(lines: List[Dict[str, Any]], page_w: int) -> List[List[Dict[str, Any]]]:
    """
    Same heuristic as main: group consecutive lines into paragraphs if:
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

# ---------------- OCR helpers ----------------
def _ensure_paddle(device: str):
    # Import inside this separate process to isolate Paddle from Torch
    from paddleocr import PaddleOCR
    return PaddleOCR(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

def _paddle_predict_any(ocr, img_bgr: np.ndarray):
    """
    Try new .predict API, then fall back to classic .ocr.
    Returns list of ([x0,y0,x1,y1], text, score)
    """
    # new API
    try:
        res = ocr.predict(img_bgr)[0]
        data = res.json.get('res', res.json)
        boxes = data.get('rec_boxes', [])
        texts = data.get('rec_texts', [])
        scores = data.get('rec_scores', [None]*len(texts))
        out = []
        for b, t, s in zip(boxes, texts, scores):
            x0,y0,x1,y1 = map(float, b)
            out.append(([x0,y0,x1,y1], t, s))
        return out
    except Exception:
        pass
    # classic API
    out = []
    try:
        res = ocr.ocr(img_bgr, cls=False)
        itms = res[0] if res and isinstance(res, list) else []
        for it in itms:
            poly = it[0]; txt, sc = it[1]
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            x0,y0,x1,y1 = min(xs),min(ys),max(xs),max(ys)
            out.append(([x0, y0, x1, y1], txt, sc))
    except Exception:
        pass
    return out

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

# ---------------- main worker ----------------
def build_ocr_struct_for_pages(pages_dir: str, page_keys: List[str], device: str, save_debug: bool) -> Dict[str, Dict[str, Any]]:
    ocr = _ensure_paddle(device=device)

    root_dir = os.path.dirname(pages_dir)
    ocr_img_dir = os.path.join(root_dir, 'ocr_images')
    ocr_json_dir = os.path.join(root_dir, 'ocr_json')
    if save_debug:
        os.makedirs(ocr_img_dir, exist_ok=True)
        os.makedirs(ocr_json_dir, exist_ok=True)

    out: Dict[str, Dict[str, Any]] = {}
    for pk in page_keys:
        img_path = os.path.join(pages_dir, f"{pk}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            out[pk] = {'page_size': {'w':0,'h':0,'scale':1.0,'dpi':RENDER_DPI}, 'blocks':[], 'lines_flat':[], 'spans_flat':[], 'entries':[]}
            continue

        H, W = img.shape[:2]
        preds = _paddle_predict_any(ocr, img)

        # save raw json + overlay for debug
        if save_debug:
            try:
                data = {
                    'rec_boxes': [p[0] for p in preds],
                    'rec_texts': [p[1] for p in preds],
                    'rec_scores': [p[2] for p in preds],
                }
                with open(os.path.join(ocr_json_dir, f"{pk}.json"), 'w', encoding='utf-8') as jf:
                    json.dump(data, jf, indent=2)
                pil_img = bgr_to_pil(img)
                draw = ImageDraw.Draw(pil_img)
                for (x0,y0,x1,y1), _, _ in preds:
                    draw.rectangle([int(x0),int(y0),int(x1),int(y1)], outline="red", width=2)
                imwrite_jpg(os.path.join(ocr_img_dir, f"{pk}.jpg"), cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[OCR_HELPER] warn: debug save failed for {pk}: {e}", file=sys.stderr)

        # spans
        spans = []
        span_id = 0
        for (x0,y0,x1,y1), txt, _ in preds:
            t = (txt or "").strip()
            if not t:
                continue
            bb = _xyxy_to_xywh(x0, y0, x1, y1)
            spans.append({
                'span_id': span_id, 'bbox': bb, 'text': t,
                'font': None, 'size': float(bb[3]), 'bold': None, 'italic': None, 'color': None
            })
            span_id += 1

        # lines → paragraphs → block
        lines_list = []
        for line_idx, grp in enumerate(_group_spans_to_lines(spans, W)):
            line_text = " ".join([s['text'] for s in grp]).strip()
            lbb = grp[0]['bbox']
            for s in grp[1:]:
                lbb = _union(lbb, s['bbox'])
            lines_list.append({'line_id': line_idx, 'bbox': lbb, 'text': line_text, 'spans': grp})

        tmp_lines = [{'bbox': ln['bbox'], 'text': ln['text'], 'spans': ln['spans'], 'page_w': W} for ln in lines_list]
        paragraphs = []
        for para_id, group in enumerate(_group_lines_into_paragraphs(tmp_lines, W)):
            pbox = None
            lines_out = []
            for ln in group:
                lines_out.append({'line_id': len(lines_out), 'bbox': ln['bbox'], 'text': ln['text'], 'spans': ln['spans']})
                pbox = ln['bbox'] if pbox is None else _union(pbox, ln['bbox'])
            paragraphs.append({'para_id': para_id, 'bbox': pbox or [0,0,1,1], 'lines': lines_out})

        blocks = [{'block_id': 0, 'type': 'text', 'bbox': paragraphs[0]['bbox'] if paragraphs else [0,0,W,H], 'paragraphs': paragraphs}]
        entries, lines_flat, spans_flat = _flatten_blocks(blocks)

        out[pk] = {
            'page_size': {'w': W, 'h': H, 'scale': 1.0, 'dpi': RENDER_DPI},
            'blocks': blocks, 'lines_flat': lines_flat, 'spans_flat': spans_flat, 'entries': entries
        }

    return out

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", required=True, help="Directory containing pageNNN.jpg images.")
    ap.add_argument("--pages-file", required=True, help="Text file with one page key per line (e.g., page001).")
    ap.add_argument("--out-json", required=True, help="Path to write OCR struct JSON.")
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"], help="Paddle device to use in helper.")
    ap.add_argument("--save-debug", action="store_true", help="Save overlays and raw OCR JSON alongside pages.")
    return ap.parse_args()

def main():
    args = parse_args()
    # Read page keys
    try:
        with open(args.pages_file, "r", encoding="utf-8") as f:
            page_keys = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"[OCR_HELPER] failed to read pages file: {e}", file=sys.stderr)
        page_keys = []

    # Run OCR; never exit without writing JSON
    try:
        res = build_ocr_struct_for_pages(args.pages_dir, page_keys, device=args.device, save_debug=bool(args.save_debug))
    except Exception as e:
        print(f"[OCR_HELPER] ERROR during OCR: {e}", file=sys.stderr)
        res = {}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[OCR_HELPER] Wrote OCR struct for {len(res)} page(s) → {args.out_json}")

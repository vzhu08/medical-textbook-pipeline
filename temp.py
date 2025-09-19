#!/usr/bin/env python3
"""
draw_structure_boxes.py

Quick test tool:
- Input: a page image and a per-page (or compiled) text-structure JSON.
- Output: the image with structure boxes drawn.

What it draws:
- Layout blocks from PP-StructureV3 (bbox_xyxy).
- Paragraph boxes within each block, colored by role:
    paragraph = yellow, heading = magenta, title = cyan.

Usage:
  py -3.10 draw_structure_boxes.py --image path/to/page001.jpg --struct path/to/page001.structure.json --out out.jpg
  # If you pass a compiled file: add --page-key page001

Notes:
- Expects each block to have: {"type": "...", "bbox_xyxy": [x1,y1,x2,y2], "paragraphs":[{"bbox_xyxy":[...], "role": "..."}]}
- Works with compiled JSON (top-level dict keyed by page_key) or single-page JSON (has top-level "blocks").
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------- I/O helpers ----------------

def imread_any(path: str) -> np.ndarray:
    """Robust image reader for arbitrary paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    root, ext = os.path.splitext(path)
    if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        path = root + ".jpg"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    print(f"[OK] wrote {path}")



# ---------------- JSON structure loaders ----------------

def _to_xyxy(b: Any) -> Optional[List[int]]:
    """Accept [x1,y1,x2,y2] or [x,y,w,h] or polygon → [x1,y1,x2,y2]."""
    if b is None:
        return None
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        x1, y1, u, v = b
        if u > x1 and v > y1:
            return [int(round(x1)), int(round(y1)), int(round(u)), int(round(v))]
        x, y, w, h = b
        return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]
    if isinstance(b, (list, tuple)) and len(b) >= 4 and isinstance(b[0], (list, tuple)):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return None


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def infer_page_key_from_image(struct_map: Dict[str, Any], image_path: str) -> Optional[str]:
    """Try to match 'pageNNN' from image filename to a key in compiled struct."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = sorted(struct_map.keys(), key=_natural_key)
    if stem in struct_map:
        return stem
    # Try case-insensitive
    for k in candidates:
        if k.lower() == stem.lower():
            return k
    # Try extracting a number and matching page### pattern
    m = re.search(r'(\d+)', stem)
    if m:
        num = int(m.group(1))
        target1 = f"page{num:03d}"
        target2 = f"Page{num:03d}"
        for t in (target1, target2, str(num)):
            if t in struct_map:
                return t
    # Fallback: first key
    return candidates[0] if candidates else None


def load_page_structure(struct_path: str, page_key: Optional[str], image_path: str) -> Dict[str, Any]:
    """
    Accepts either a per-page structure JSON (with top-level 'blocks')
    or a compiled mapping {page_key: {...}}. Returns the page dict with 'blocks'.
    """
    with open(struct_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Per-page file
    if isinstance(d, dict) and "blocks" in d:
        return d

    # Compiled file
    if not isinstance(d, dict):
        raise ValueError("Unsupported structure JSON format.")

    key = page_key or infer_page_key_from_image(d, image_path)
    if key is None or key not in d:
        raise KeyError("Page key not found in compiled structure JSON.")
    return d[key]


# ---------------- Drawing ----------------

def draw_box(img: np.ndarray, xyxy: List[int], color: Tuple[int, int, int], thickness: int, alpha: float = 0.0) -> None:
    """Draw rectangle with optional translucent fill."""
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    if alpha > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)


def put_label(img: np.ndarray, xyxy: List[int], text: str, color: Tuple[int, int, int]) -> None:
    """Draw a small filled label above the box."""
    x1, y1, x2, _ = map(int, xyxy)
    x1 = max(0, x1); x2 = max(0, x2); y1 = max(0, y1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    th = 1
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    pad = 2
    bx2 = min(img.shape[1] - 1, x1 + tw + 2 * pad)
    by1 = max(0, y1 - th_text - 2 * pad - 2)
    by2 = max(0, y1 - 2)
    # black background for readability
    cv2.rectangle(img, (x1, by1), (bx2, by2), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(img, text, (x1 + pad, by2 - pad - 1), font, fs, color, th, cv2.LINE_AA)


def draw_structure(
    img: np.ndarray,
    page_struct: Dict[str, Any],
    draw_blocks: bool = True,
    draw_paragraphs: bool = True,
    block_thickness: int = 2,
    para_thickness: int = 2,
    block_alpha: float = 0.05,
    para_alpha: float = 0.08,
) -> np.ndarray:
    """
    Draw blocks and paragraphs onto a copy of the image.
    Colors (BGR):
      blocks     = blue
      paragraph  = yellow
      heading    = magenta
      title      = cyan
    """
    out = img.copy()

    color_block = (255, 0, 0)
    color_para = (0, 255, 255)
    color_heading = (255, 0, 255)
    color_title = (255, 255, 0)

    blocks = page_struct.get("blocks", [])
    for bi, b in enumerate(blocks):
        bb = b.get("bbox_xyxy") or b.get("bbox")
        bb = _to_xyxy(bb)
        if bb is None:
            continue

        if draw_blocks:
            draw_box(out, bb, color_block, block_thickness, alpha=block_alpha)
            put_label(out, bb, f"block {bi} [{b.get('type','?')}]", color_block)

        if draw_paragraphs:
            for pi, p in enumerate(b.get("paragraphs", [])):
                pb = p.get("bbox_xyxy") or p.get("bbox")
                pb = _to_xyxy(pb)
                if pb is None:
                    continue

                role = (p.get("role") or "paragraph").lower()
                col = color_para
                if role == "heading":
                    col = color_heading
                elif role == "title":
                    col = color_title

                draw_box(out, pb, col, para_thickness, alpha=para_alpha)
                put_label(out, pb, f"p{pi} [{role}]", col)

    return out


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Draw structure boxes on an image.")
    ap.add_argument("--image", required=True, help="Path to page image.")
    ap.add_argument("--struct", required=True, help="Path to per-page or compiled structure JSON.")
    ap.add_argument("--out", required=True, help="Path to save annotated image.")
    ap.add_argument("--page-key", default=None, help="Page key if struct is compiled (e.g., page001).")
    ap.add_argument("--no-blocks", action="store_true", help="Do not draw layout blocks.")
    ap.add_argument("--no-paras", action="store_true", help="Do not draw paragraph boxes.")
    ap.add_argument("--block-thick", type=int, default=2, help="Block box thickness.")
    ap.add_argument("--para-thick", type=int, default=2, help="Paragraph box thickness.")
    ap.add_argument("--block-alpha", type=float, default=0.05, help="Block fill alpha [0..1].")
    ap.add_argument("--para-alpha", type=float, default=0.08, help="Paragraph fill alpha [0..1].")
    args = ap.parse_args()

    img = imread_any(args.image)
    page_struct = load_page_structure(args.struct, args.page_key, args.image)

    annotated = draw_structure(
        img,
        page_struct,
        draw_blocks=not args.no_blocks,
        draw_paragraphs=not args.no_paras,
        block_thickness=args.block_thick,
        para_thickness=args.para_thick,
        block_alpha=max(0.0, min(1.0, args.block_alpha)),
        para_alpha=max(0.0, min(1.0, args.para_alpha)),
    )

    write_image(args.out, annotated)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()

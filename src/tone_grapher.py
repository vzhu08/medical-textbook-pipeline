# plot_skin_tones.py
import os
import csv
import random
import math
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2


def _load_csv_tints(csv_path: str) -> List[Tuple[str, float]]:
    """
    Returns list of (filename, tint_float) with tint in [0,100].
    Rows with blank or invalid tint are skipped.
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = (r.get('skin_tint_0_100') or '').strip()
            if not val:
                continue
            try:
                t = float(val)
            except ValueError:
                continue
            if 0.0 <= t <= 100.0:
                rows.append((r.get('filename', ''), t))
    return rows


def _color_from_rep_chip(stem: str, rep_color_dir: str) -> Optional[Tuple[float,float,float]]:
    """
    Try to read <rep_color_dir>/<stem>_repcolor.png and return mean RGB (0-1).
    Returns None if not found or unreadable.
    """
    chip_path = os.path.join(rep_color_dir, f"{stem}_repcolor.png")
    if not os.path.exists(chip_path):
        return None
    chip = cv2.imread(chip_path)
    if chip is None:
        return None
    # Convert BGR->RGB and average
    chip_rgb = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
    rgb = chip_rgb.reshape(-1, 3).mean(axis=0) / 255.0
    return tuple(np.clip(rgb, 0, 1).tolist())


def _color_from_Lstar(L: float) -> Tuple[float,float,float]:
    """
    Fallback: synthesize a skin-ish color from L* only.
    We fix a*, b* to mild skin chroma and vary L*.
    Convert Lab -> RGB via OpenCV convention (8-bit Lab).
    """
    # Choose gentle chroma (a*=20, b*=20) → tweak if you like
    L8  = np.uint8(np.clip(round(L * 255.0 / 100.0), 0, 255))
    a8  = np.uint8(128 + 20)
    b8  = np.uint8(128 + 20)
    lab = np.array([[[L8, a8, b8]]], dtype=np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)[0,0] / 255.0
    return tuple(np.clip(rgb, 0, 1).tolist())


def plot_skin_tone_distribution(
    csv_path: str,
    rep_color_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    point_size: int = 60,
    jitter: float = 0.08,
    bins: int = 20,
    show: bool = True
) -> None:
    """
    Plot a 1D distribution of skin tints (0–100) with each datapoint colored by its tone.
    - If rep_color_dir is provided, colors come from <stem>_repcolor.png chips.
    - Otherwise, colors are synthesized from L* (fixed a*, b*).
    """
    data = _load_csv_tints(csv_path)
    if not data:
        raise ValueError("No valid tint rows found in CSV.")

    fnames, tvals = zip(*data)
    colors = []
    for fn, L in data:
        stem, _ = os.path.splitext(os.path.basename(fn))
        rgb = None
        if rep_color_dir:
            rgb = _color_from_rep_chip(stem, rep_color_dir)
        if rgb is None:
            rgb = _color_from_Lstar(L)
        colors.append(rgb)

    # Build jittered y positions so points don’t overlap
    rng = random.Random(1337)
    y = [rng.uniform(-jitter, jitter) for _ in tvals]

    fig, ax = plt.subplots(figsize=(10, 3.2))  # single plot, no subplots
    # Light histogram backdrop (monochrome, low alpha)
    ax.hist(tvals, bins=bins, range=(0, 100), alpha=0.15, edgecolor='none')

    # Scatter of individual points colored by skin tone
    ax.scatter(tvals, y, s=point_size, c=colors, linewidths=0.5, edgecolors='k', alpha=0.95)

    ax.set_xlim(0, 100)
    ax.set_ylim(-3*jitter, 3*jitter)
    ax.set_xlabel("Skin tint (L* 0–100)")
    ax.set_yticks([])
    ax.set_title("Skin Tone Distribution")

    ax.grid(axis='x', alpha=0.2, linestyle='--')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)



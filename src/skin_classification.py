#!/usr/bin/env python3
"""
ABD skin segmentation (PyTorch UNet) + STRICT CRF + HSV fallback + light cleanup
GPU-aware: set use_gpu=True (and CUDA available) to run single-process batched
inference on GPU with mixed precision. Fallbacks to CPU otherwise.

Pipeline:
  1) Segment with UNet → per-pixel probabilities (saved if CRF is on)
  2) Strict, edge-respecting DenseCRF on the prob map → CRF mask
  3) If CRF mask covers <10% of the image, fallback to HSV skin mask
  4) On the final mask (CRF or fallback):
       - edge cleanup (remove 1% border ring)
       - remove tiny components <1% of the image area
  5) Save masked composites and representative color

Outputs:
    <output>/masks/<file>            raw threshold masks (0/255)
    <output>/masked/<file>           original × final mask composites
    <output>/probs/<stem>.npy        float32 prob maps (if CRF enabled)
    <output>/clusters/<stem>_*       KMeans centers/counts + swatch
    <output>/rep_color/<stem>_*      representative color chip
    <output>/skin_tones.csv          filename, L* tint (0–100)
    <output>/fallbacks.csv           filename, crf_coverage (0–1) for HSV fallbacks
"""
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import nullcontext
import concurrent.futures

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

import torch
import torch.nn as nn

# Prefer cuDNN autotune when on GPU
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ------------------------------ Config ------------------------------
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT_SIZE = 128
DEFAULT_THRESH = 0.7
DEFAULT_BATCH = 64

# --- STRICT CRF knobs (edge-respecting; gentle smoothing) ---
USE_CRF_DEFAULT = True
SAVE_PROBS_DEFAULT = True

CRF_ITERS = 3
CRF_W_GAUSS = 0
CRF_SXY_GAUSS = (3, 3)

CRF_W_BILAT = 4
CRF_SXY_BILAT = (35, 35)
CRF_SRGB = (5, 5, 5)

CRF_UNARY_SCALE = 2.5

# --- Fallback & cleanup knobs ---
FALLBACK_MIN_COVERAGE_FRAC = 0.10  # if CRF mask covers <10%, use HSV fallback
EDGE_ERODE_FRAC = 0.01             # shrink mask by ~1% of min(H, W)
SMALL_COMP_FRAC = 0.01             # remove CCs <1% of image area

# HSV skin ranges (OpenCV: H 0..179, S/V 0..255); union of two hue bands
HSV1_LO = (0,   30,  60)
HSV1_HI = (25, 180, 255)
HSV2_LO = (160, 30,  60)
HSV2_HI = (179, 180, 255)

# --- helper: torch API for CUDA ---
def _amp_autocast(device: str, enabled: bool):
    # Use the new torch.amp.autocast on CUDA; no-op elsewhere
    if enabled and device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

# --- helper: ensures each post-processing worker doesn't spawn OpenCV inner threads ---
def _init_post_worker():
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

# --- helper: expected outputs for a given file ---
def _expected_outputs_for(fname: str, output_dir: str):
    stem, _ = os.path.splitext(fname)
    return [
        os.path.join(output_dir, 'masked',   fname),
        os.path.join(output_dir, 'clusters', f"{stem}_centers.npy"),
        os.path.join(output_dir, 'clusters', f"{stem}_counts.npy"),
        os.path.join(output_dir, 'clusters', f"{stem}_clusters.png"),
        os.path.join(output_dir, 'rep_color', f"{stem}_repcolor.png"),
    ]

# --- helper: compute tint from saved npys (no reprocessing needed) ---
def _tint_from_npys(output_dir: str, stem: str) -> Optional[float]:
    c_path = os.path.join(output_dir, 'clusters', f"{stem}_centers.npy")
    n_path = os.path.join(output_dir, 'clusters', f"{stem}_counts.npy")
    if not (os.path.exists(c_path) and os.path.exists(n_path)):
        return None
    centers = np.load(c_path).astype(np.float32)
    counts  = np.load(n_path).astype(np.float32)
    if centers.size == 0 or counts.size == 0:
        return None
    top = int(min(3, len(counts)))
    order = np.argsort(counts)[::-1][:top]
    top_c, top_w = centers[order], counts[order]
    labs = []
    for bgr in top_c:
        labs.append(cv2.cvtColor(np.clip(bgr,0,255).astype(np.uint8)[None,None,:], cv2.COLOR_BGR2LAB)[0,0].astype(np.float32))
    lab = np.stack(labs, 0)
    rep = (lab * top_w[:,None]).sum(axis=0) / max(1.0, top_w.sum())
    return float(round(rep[0] / 2.55, 2))

# ------------------------------ layers.py (inline) ------------------------------
class _Layers:
    @staticmethod
    def double_conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
layers = _Layers()

# ------------------------------ model.py (inline) ------------------------------
class UNet(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv_1 = layers.double_conv_bn_relu(input_channels, 64)
        self.double_conv_2 = layers.double_conv_bn_relu(64, 128)
        self.double_conv_3 = layers.double_conv_bn_relu(128, 256)
        self.double_conv_4 = layers.double_conv_bn_relu(256, 256)

        self.deconv2d_1 = layers.double_conv_bn_relu(512, 128)
        self.deconv2d_2 = layers.double_conv_bn_relu(256, 64)
        self.deconv2d_3 = layers.double_conv_bn_relu(128, 64)

        self.final_double_conv = layers.double_conv_bn_relu(64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.double_conv_1(x)
        x2 = self.max_pool(x1)
        x3 = self.double_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.double_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.double_conv_4(x6)

        x8  = self.upsample(x7);  x8  = torch.cat([x8,  x5], dim=1); x8  = self.deconv2d_1(x8)
        x9  = self.upsample(x8);  x9  = torch.cat([x9,  x3], dim=1); x9  = self.deconv2d_2(x9)
        x10 = self.upsample(x9);  x10 = torch.cat([x10, x1], dim=1); x10 = self.deconv2d_3(x10)

        x11 = self.final_double_conv(x10)
        return self.sigmoid(self.final_conv(x11))  # (B,1,H,W)

# ------------------------------ Checkpoint loader ------------------------------
def abd_load_torch_unet(abd_model_path: str, device: str = "cpu") -> Tuple[nn.Module, str]:
    if not os.path.isfile(abd_model_path):
        raise FileNotFoundError(f"Model not found: {abd_model_path}")
    model = UNet(input_channels=3)

    ckpt = torch.load(abd_model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    elif isinstance(ckpt, nn.Module):
        model = ckpt
        model.to(device).eval()
        return model, device
    else:
        state = next((v for v in ckpt.values() if isinstance(v, dict)), None)
        if state is None:
            raise ValueError("Unsupported checkpoint format")

    prefixes = ("module.", "model.", "net.")
    clean = {}
    for k, v in state.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict mismatches:\n  missing={len(missing)}  unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, device

# ------------------------------ Worker/shared globals ------------------------------
_G_MODEL = None
_G_DEVICE = "cpu"
_G_IN_SIZE = DEFAULT_INPUT_SIZE
_G_THR = DEFAULT_THRESH
_G_BATCH = DEFAULT_BATCH

_G_SAVE_PROBS = False
_G_PROBS_DIR = None

# --- worker resolvers ---
def _resolve_seg_workers(provided: Optional[int]) -> int:
    return max(1, int(provided)) if provided is not None else 1

def _resolve_post_workers(provided: Optional[int]) -> int:
    if provided is not None:
        return max(1, int(provided))
    cpu = os.cpu_count() or 1
    return max(1, cpu - 2)

def _init_abd_worker(model_path: str, input_size: int, thr: float, batch_size: int,
                     save_probs: bool=False, probs_dir: Optional[str]=None, prob_format: str="npy"):
    import torch as _torch, cv2 as _cv2
    try:
        _cv2.setNumThreads(0)
    except Exception:
        pass
    _torch.set_num_threads(1)
    global _G_MODEL, _G_DEVICE, _G_IN_SIZE, _G_THR, _G_BATCH, _G_SAVE_PROBS, _G_PROBS_DIR
    _G_DEVICE = "cpu"  # workers are CPU-only
    _G_IN_SIZE = int(input_size)
    _G_THR = float(thr)
    _G_BATCH = int(batch_size)
    _G_SAVE_PROBS = bool(save_probs)
    _G_PROBS_DIR = probs_dir
    _G_MODEL, _ = abd_load_torch_unet(model_path, device=_G_DEVICE)

@torch.inference_mode()
def _predict_chunk(img_paths, in_size, thr, model, device, return_probs: bool = False, use_amp: bool = False):
    outs = []
    batch, metas = [], []
    for p in img_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h0, w0 = im.shape[:2]
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (in_size, in_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        batch.append(x); metas.append((p, h0, w0))
    if not batch:
        return outs

    X = torch.from_numpy(np.stack(batch, 0)).to(device, non_blocking=True)
    # Mixed precision on CUDA only
    with _amp_autocast(device, use_amp):
        y = model(X)[:, 0]
    y = y.float().cpu().numpy()  # back to float32 on CPU for CV ops

    for (p, h0, w0), yi in zip(metas, y):
        prob = cv2.resize(yi, (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m = (prob > thr).astype(np.uint8) * 255
        outs.append((p, m, prob) if return_probs else (p, m))
    return outs

def _abd_worker_chunk(paths: List[str], masks_dir: str) -> int:
    total = 0
    model = _G_MODEL
    for i in range(0, len(paths), _G_BATCH):
        chunk = paths[i:i+_G_BATCH]
        preds = _predict_chunk(chunk, _G_IN_SIZE, _G_THR, model, _G_DEVICE, return_probs=_G_SAVE_PROBS, use_amp=False)
        for item in preds:
            if _G_SAVE_PROBS:
                p, m, prob = item
            else:
                p, m = item
            cv2.imwrite(os.path.join(masks_dir, os.path.basename(p)), m)
            if _G_SAVE_PROBS and _G_PROBS_DIR:
                stem = os.path.splitext(os.path.basename(p))[0]
                np.save(os.path.join(_G_PROBS_DIR, f"{stem}.npy"), prob.astype(np.float32))
            total += 1
    return total

def run_segmentation_abd(
    input_dir: str,
    masks_dir: str,
    abd_model_path: str,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    thr: float = DEFAULT_THRESH,
    batch_size: int = DEFAULT_BATCH,
    workers: Optional[int] = None,
    files_per_worker: int = 256,
    save_probs: bool = False,
    probs_dir: Optional[str] = None,
    prob_format: str = "npy",
    use_gpu: bool = True,  # <— NEW: GPU toggle
) -> None:
    os.makedirs(masks_dir, exist_ok=True)
    if save_probs:
        assert probs_dir is not None, "probs_dir required when save_probs=True"
        os.makedirs(probs_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in EXTS]
    pending = [os.path.join(input_dir, f) for f in files if not os.path.exists(os.path.join(masks_dir, f))]
    if not pending:
        print("[SEG] All masks exist. Skipping."); return

    # Choose device
    have_cuda = torch.cuda.is_available()
    device = "cuda" if (use_gpu and have_cuda) else "cpu"
    use_amp = (device == "cuda")

    # --- GPU path: single-process, batched, mixed precision ---
    if device == "cuda":
        try:
            import cv2 as _cv2
            _cv2.setNumThreads(0)
        except Exception:
            pass
        torch.set_num_threads(1)
        print(f"[SEG] GPU mode | device=cuda | batch={batch_size} | in={input_size} | save_probs={save_probs}")
        model, _ = abd_load_torch_unet(abd_model_path, device=device)
        # Process sequential batches on the GPU
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            preds = _predict_chunk(chunk, input_size, thr, model, device, return_probs=save_probs, use_amp=use_amp)
            for item in preds:
                if save_probs: p, m, prob = item
                else:          p, m = item
                cv2.imwrite(os.path.join(masks_dir, os.path.basename(p)), m)
                if save_probs and probs_dir:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    np.save(os.path.join(probs_dir, f"{stem}.npy"), prob.astype(np.float32))
            print(f"[SEG][GPU] {min(i + batch_size, len(pending))}/{len(pending)}")
        return

    # --- CPU path: can use multi-processing workers ---
    w = max(1, int(workers)) if workers is not None else max(1, (os.cpu_count() or 1) - 2)
    if w <= 1:
        try:
            import cv2 as _cv2
            _cv2.setNumThreads(0)
        except Exception:
            pass
        th = max(1, (os.cpu_count() or 1) - 2)
        torch.set_num_threads(th)
        print(f"[SEG] CPU single-process | torch threads={th} | batch={batch_size} | in={input_size}")
        model, device = abd_load_torch_unet(abd_model_path, device="cpu")
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            preds = _predict_chunk(chunk, input_size, thr, model, device, return_probs=save_probs, use_amp=False)
            for item in preds:
                if save_probs: p, m, prob = item
                else:          p, m = item
                cv2.imwrite(os.path.join(masks_dir, os.path.basename(p)), m)
                if save_probs and probs_dir:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    np.save(os.path.join(probs_dir, f"{stem}.npy"), prob.astype(np.float32))
            print(f"[SEG][CPU] {min(i + batch_size, len(pending))}/{len(pending)}")
        return

    # CPU multi-process
    chunks = [pending[i:i+files_per_worker] for i in range(0, len(pending), files_per_worker)]
    print(f"[SEG] CPU multi-proc | {len(pending)} imgs → {len(chunks)} chunks | workers={w} | per-worker batch={batch_size} | in={input_size}")
    done = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=w,
        initializer=_init_abd_worker,
        initargs=(abd_model_path, input_size, thr, batch_size, save_probs, probs_dir, prob_format),
    ) as pool:
        futs = [pool.submit(_abd_worker_chunk, ch, masks_dir) for ch in chunks]
        for fut in concurrent.futures.as_completed(futs):
            try:
                done += fut.result(); print(f"[SEG] masks written: {done}/{len(pending)}")
            except Exception as e:
                print("[SEG][ERR] worker failed:", e)

# ------------------------------ STRICT CRF ------------------------------
def _refine_with_crf(img_bgr: np.ndarray, prob01: np.ndarray, iters: int = CRF_ITERS) -> np.ndarray:
    H, W = prob01.shape
    CRF_MAX_PIXELS = 1_800_000
    scale = 1.0
    if H * W > CRF_MAX_PIXELS:
        import math
        scale = math.sqrt(CRF_MAX_PIXELS / float(H * W))
        newW = max(32, int(W * scale)); newH = max(32, int(H * scale))
        img = cv2.resize(img_bgr, (newW, newH), interpolation=cv2.INTER_AREA)
        prb = cv2.resize(prob01,  (newW, newH), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr; prb = prob01

    hS, wS = prb.shape
    p = np.clip(prb.astype(np.float32), 1e-6, 1 - 1e-6)
    softmax = np.stack([1.0 - p, p], axis=0).astype(np.float32)
    U = unary_from_softmax(softmax) * float(CRF_UNARY_SCALE)

    d = dcrf.DenseCRF2D(wS, hS, 2)
    d.setUnaryEnergy(U)
    if CRF_W_GAUSS > 0:
        d.addPairwiseGaussian(sxy=CRF_SXY_GAUSS, compat=CRF_W_GAUSS)
    d.addPairwiseBilateral(sxy=CRF_SXY_BILAT, srgb=CRF_SRGB, rgbim=img, compat=CRF_W_BILAT)

    Q = np.array(d.inference(iters), dtype=np.float32).reshape(2, hS, wS)
    m_small = (Q[1] > Q[0]).astype(np.uint8) * 255

    if scale != 1.0:
        return cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
    return m_small

# ------------------------------ Helpers: fallback + cleanup ------------------------------
def _hsv_skin_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """Simple HSV-based skin mask (union of two hue bands). Returns 0/255 mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(HSV1_LO, np.uint8), np.array(HSV1_HI, np.uint8))
    m2 = cv2.inRange(hsv, np.array(HSV2_LO, np.uint8), np.array(HSV2_HI, np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    return mask

def _erode_mask_border(mask: np.ndarray, frac: float) -> np.ndarray:
    """
    Uniformly erode the *mask itself* by a ring thickness of frac * min(H,W).
    Works even when the mask touches image edges (pads with zeros first).
    Expects mask as 0/255 or 0/1; returns 0/255 uint8.
    """
    if frac <= 0:
        return mask

    h, w = mask.shape[:2]
    r = max(1, int(round(frac * min(h, w))))
    r = min(r, 64)  # safety cap

    bin_mask = (mask > 0).astype(np.uint8) * 255
    pad = r
    bin_pad = cv2.copyMakeBorder(bin_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode(bin_pad, k, iterations=1)
    eroded = eroded[pad:-pad, pad:-pad]
    return eroded

def _remove_small_components(mask: np.ndarray, min_frac: float) -> np.ndarray:
    """Keep CCs whose area >= min_frac * image_area."""
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    min_area = max(1, int(min_frac * H * W))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    keep = np.zeros_like(m)
    for lbl in range(1, num):
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == lbl] = 1
    return (keep * 255).astype(np.uint8)

# ------------------------------ Post-processing ------------------------------
def _postprocess_file(fname: str, input_dir: str, output_dir: str, k_unused: int) -> Tuple[str, Optional[float], bool, float]:
    """
    Returns: (filename, tint, used_fallback, crf_coverage_0_1)
    """
    stem, ext = os.path.splitext(fname)
    if ext.lower() not in EXTS:
        return (fname, None, False, 0.0)

    masks_dir  = os.path.join(output_dir, 'masks')
    masked_dir = os.path.join(output_dir, 'masked');    os.makedirs(masked_dir,  exist_ok=True)
    clusters   = os.path.join(output_dir, 'clusters');  os.makedirs(clusters,    exist_ok=True)
    rep_dir    = os.path.join(output_dir, 'rep_color'); os.makedirs(rep_dir,     exist_ok=True)

    orig = cv2.imread(os.path.join(input_dir, fname))
    mask = cv2.imread(os.path.join(masks_dir,  fname), cv2.IMREAD_GRAYSCALE)
    if orig is None or mask is None:
        return (fname, None, False, 0.0)

    H, W = orig.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 0).astype(np.uint8) * 255)

    # ---- Strict CRF refine; use CRF result directly (expand/shrink allowed) ----
    crf_coverage = 0.0
    prob_path = os.path.join(output_dir, 'probs', f"{stem}.npy")
    if os.path.exists(prob_path) and USE_CRF_DEFAULT:
        prob = np.load(prob_path).astype(np.float32)
        if prob.shape[:2] != (H, W):
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        crf_mask = _refine_with_crf(orig, prob, iters=CRF_ITERS)
        if crf_mask.shape[:2] != (H, W):
            crf_mask = cv2.resize(crf_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = crf_mask


    crf_coverage = float(np.count_nonzero(mask) / float(H * W))

    # ---- Fallback to HSV if CRF coverage is too small ----
    used_fallback = False
    if crf_coverage < FALLBACK_MIN_COVERAGE_FRAC:
        mask = _hsv_skin_fallback(orig)
        used_fallback = True

    # ---- Light cleanup on the final mask (edge erode + tiny CC removal) ----
    mask = _remove_small_components(mask, SMALL_COMP_FRAC)  # drop tiny CCs first
    mask = _erode_mask_border(mask, EDGE_ERODE_FRAC)        # shave the mask edge

    # ---- Save masked composite ----
    comp = cv2.bitwise_and(orig, orig, mask=mask)
    cv2.imwrite(os.path.join(masked_dir, fname), comp)

    # ---- Representative color (sample final mask) ----
    skin = orig[mask > 0]
    if skin.size == 0:
        return (fname, None, used_fallback, crf_coverage)

    pixels = skin.reshape(-1, 3).astype(np.float32)
    n_samples = pixels.shape[0]

    from sklearn.cluster import KMeans
    if n_samples >= 2:
        k_use = min(5, n_samples)
        try:
            km = KMeans(n_clusters=k_use, random_state=0, n_init='auto').fit(pixels)
        except TypeError:
            km = KMeans(n_clusters=k_use, random_state=0, n_init=10).fit(pixels)
        centers = km.cluster_centers_.astype(np.float32)
        counts  = np.bincount(km.labels_, minlength=k_use).astype(np.int64)
    else:
        centers = np.array([pixels.mean(axis=0)], dtype=np.float32)
        counts  = np.array([n_samples], dtype=np.int64)

    centers_npy = os.path.join(clusters, f"{stem}_centers.npy")
    counts_npy  = os.path.join(clusters, f"{stem}_counts.npy")
    np.save(centers_npy, centers); np.save(counts_npy, counts)

    cluster_png = os.path.join(clusters, f"{stem}_clusters.png")
    if not os.path.exists(cluster_png):
        top = min(3, len(counts))
        order = np.argsort(counts)[::-1][:top]
        swatch = np.zeros((50, top*50, 3), dtype=np.uint8)
        for i, col in enumerate(centers[order]):
            swatch[:, i*50:(i+1)*50] = np.clip(col, 0, 255).astype(np.uint8)
        cv2.imwrite(cluster_png, swatch)

    # Weighted LAB → L*
    top = min(3, len(counts))
    order = np.argsort(counts)[::-1][:top]
    top_c = centers[order]; top_w = counts[order].astype(np.float32)
    lab_centers = [cv2.cvtColor(np.clip(bgr,0,255).astype(np.uint8)[None,None,:], cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
                   for bgr in top_c]
    lab_centers = np.stack(lab_centers, 0)
    rep_lab = (lab_centers * top_w[:,None]).sum(axis=0) / max(1.0, top_w.sum())
    tint = float(rep_lab[0] / 2.55)

    rep_png = os.path.join(rep_dir, f"{stem}_repcolor.png")
    if not os.path.exists(rep_png):
        rep_lab_u8 = np.clip(np.round(rep_lab), 0, 255).astype(np.uint8)[None,None,:]
        bgr = cv2.cvtColor(rep_lab_u8, cv2.COLOR_LAB2BGR)[0,0]
        chip = np.zeros((50,50,3), np.uint8); chip[:] = bgr
        cv2.imwrite(rep_png, chip)

    return (fname, tint, used_fallback, crf_coverage)

def run_postprocessing(
    input_dir: str,
    output_dir: str,
    *,
    k: int = 5,                         # unused in this flow (kept for API compat)
    workers: Optional[int] = None
) -> None:
    files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in EXTS]
    masks_dir = os.path.join(output_dir, 'masks')

    # Only consider files that have a mask already
    files_with_masks = [f for f in files if os.path.exists(os.path.join(masks_dir, f))]

    # Select pending = any expected output missing
    pending = []
    for f in files_with_masks:
        outs = _expected_outputs_for(f, output_dir)
        if not all(os.path.exists(p) for p in outs):
            pending.append(f)

    # Resolve workers (same policy as elsewhere)
    cpu = os.cpu_count() or 1
    cap = max(1, cpu - 2)
    w = cap if workers is None else min(max(int(workers), 1), cap)

    print(f"[POST] {len(files_with_masks)} files | pending={len(pending)} | workers={w}")

    results: List[Tuple[str, Optional[float], bool, float]] = []
    if pending:
        if w <= 1:
            try:
                cv2.setNumThreads(0)
            except Exception:
                pass
            for f in pending:
                results.append(_postprocess_file(f, input_dir, output_dir, k))
        else:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=w,
                    initializer=_init_post_worker,  # <— add this
            ) as exe:
                futs = [exe.submit(_postprocess_file, f, input_dir, output_dir, k) for f in pending]
                for fut in concurrent.futures.as_completed(futs):
                    results.append(fut.result())

    # ------- Merge/update CSVs without losing previous rows -------

    # 1) skin_tones.csv
    tones_csv = os.path.join(output_dir, 'skin_tones.csv')
    tones_map = {}
    if os.path.exists(tones_csv):
        with open(tones_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                tones_map[row['filename']] = row.get('skin_tint_0_100', '')

    # update with new results
    for fn, ti, _, _ in results:
        if ti is not None:
            tones_map[fn] = f"{ti:.2f}"

    # ensure every file with cluster npys is represented (fill from npys if missing)
    for f in files_with_masks:
        if f not in tones_map or tones_map[f] in (None, ''):
            stem, _ = os.path.splitext(f)
            ti = _tint_from_npys(output_dir, stem)
            if ti is not None:
                tones_map[f] = f"{ti:.2f}"
            else:
                tones_map.setdefault(f, '')

    with open(tones_csv, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f); wr.writerow(['filename', 'skin_tint_0_100'])
        for fn in sorted(tones_map):
            wr.writerow([fn, tones_map[fn]])
    print(f"[POST] Wrote {tones_csv}")

    # 2) fallbacks.csv (merge existing + new)
    fb_csv = os.path.join(output_dir, 'fallbacks.csv')
    fb_map = {}
    if os.path.exists(fb_csv):
        with open(fb_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                fn = row.get('filename')
                cov = row.get('crf_coverage_0_1')
                if fn:
                    fb_map[fn] = cov if cov is not None else ''

    # add new fallbacks from this run
    for fn, _, used_fb, cov in results:
        if used_fb:
            fb_map[fn] = f"{cov:.6f}"

    with open(fb_csv, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f); wr.writerow(['filename', 'crf_coverage_0_1'])
        for fn in sorted(fb_map):
            wr.writerow([fn, fb_map[fn]])
    print(f"[POST] Wrote {fb_csv} ({len(fb_map)} total fallbacks)")

# ------------------------------ Orchestrator ------------------------------
def classify_skin(
    input_dir: str,
    output_dir: str,
    abd_model_path: str,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    thr: float = DEFAULT_THRESH,
    batch_size: int = DEFAULT_BATCH,
    k: int = 5,  # unused here
    seg_workers: Optional[int] = None,
    post_workers: Optional[int] = None,
    workers: Optional[int] = None,
    use_crf: bool = USE_CRF_DEFAULT,
    use_gpu: bool = True,   # <— NEW: pass from main.py
) -> None:
    for sub in ['masks','masked','clusters','rep_color'] + (['probs'] if use_crf else []):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Compute cap = max(cpu-2, 1)
    cpu = os.cpu_count() or 1
    cap = max(1, cpu - 2)

    def _resolve(requested: Optional[int]) -> int:
        """
        Resolve worker count with this policy:
        - If a specific value is provided, clamp it to [1, cap].
        - Else, if a global 'workers' is provided, clamp that to [1, cap].
        - Else, default to 'cap'.
        """
        base = requested if requested is not None else workers
        if base is None:
            return cap
        try:
            base = int(base)
        except Exception:
            return cap
        return min(max(base, 1), cap)

    seg_w  = _resolve(seg_workers)
    post_w = _resolve(post_workers)

    run_segmentation_abd(
        input_dir=input_dir,
        masks_dir=os.path.join(output_dir, 'masks'),
        abd_model_path=abd_model_path,
        input_size=input_size,
        thr=thr,
        batch_size=batch_size,
        workers=seg_w,
        files_per_worker=256,
        save_probs=use_crf,
        probs_dir=os.path.join(output_dir, 'probs') if use_crf else None,
        prob_format="npy",
        use_gpu=use_gpu,  # <— NEW
    )

    run_postprocessing(
        input_dir=input_dir,
        output_dir=output_dir,
        k=k,
        workers=post_w
    )

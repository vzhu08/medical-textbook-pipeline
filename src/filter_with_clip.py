# src/filter_with_clip.py
#
# CHANGES (per request):
# 1) Save CLIP similarity scores to a JSON in the *main output folder*.
#    - Includes per-label, per-category mean/max, decision scores (mean or max),
#      chosen category, uncertainty flag, and margin between top-2 decision scores.
# 2) Add optional "uncertain" category with a certainty threshold.
#    - If enabled and (top1 - top2) < threshold, route image to an "uncertain" subfolder.
# 3) Preserve existing API and behavior; no CLI added. "run_image_extraction" remains elsewhere.
#
# NOTES:
# - "Decision scores" are per-category. If use_mean=True we use category means; otherwise
#   we use category max (i.e., the best label within each category). Uncertainty is computed
#   on those decision scores to match whichever decision rule is active.
# - JSON is written once at the end to output_folder/clip_similarity_scores.json

import os
import glob
import json
import shutil
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from contextlib import nullcontext
from typing import Dict, List, Tuple
from threading import Lock

# Prefer fast matmul on Ampere+/RTX
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _amp_autocast(device: str, enabled: bool):
    """
    Use new torch.amp.autocast API on CUDA; no-op elsewhere.
    """
    if enabled and device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_clip_model(use_gpu: bool | None = None, device: str | None = None):
    """
    Load CLIP model, tokenizer, and image processor.
    - If use_gpu is True and CUDA is available, use CUDA with mixed precision.
    - If use_gpu is False (or CUDA unavailable), use CPU.
    Returns (model, tokenizer, image_processor, device, use_amp)
    """
    if device is None:
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    use_amp = (device.startswith("cuda") and torch.cuda.is_available())

    # Use FP16 weights on GPU for speed/memory; float32 on CPU for stability
    dtype = torch.float16 if use_amp else torch.float32

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        torch_dtype=dtype,
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()
    return model, tokenizer, image_processor, device, use_amp


def _compute_category_maps(categories: Dict[str, List[str]]) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """
    Flatten labels and build helper maps:
      - all_labels:        list of all label strings in input order
      - label_to_category: parallel list with each label's category name
      - category_to_indices: map from category -> indices in all_labels
    """
    all_labels: List[str] = []
    label_to_category: List[str] = []
    for cat, labels in categories.items():
        for lab in labels:
            all_labels.append(lab)
            label_to_category.append(cat)

    category_to_indices: Dict[str, List[int]] = {}
    for cat in categories:
        category_to_indices[cat] = [i for i, c in enumerate(label_to_category) if c == cat]

    return all_labels, label_to_category, category_to_indices


def filter_with_clip(
    input_folder: str,
    output_folder: str,
    categories: Dict[str, List[str]],
    use_mean: bool = False,
    batch_size: int = 32,
    workers: int | None = None,
    use_gpu: bool = True,  # toggle GPU from main
    include_uncertain: bool = False,  # NEW: enable/disable uncertain bucket
    uncertainty_threshold: float = 0.01,  # NEW: top1 - top2 margin threshold
    scores_json_name: str = "clip_similarity_scores.json",  # NEW: output JSON filename
):
    """
    Categorize all images in 'input_folder' into subfolders under 'output_folder' based on CLIP similarities.

    Parameters
    ----------
    use_mean : bool
        If True, pick category by MEAN score across that category's prompts.
        If False, pick by category MAX (i.e., highest label score within that category).
    include_uncertain : bool
        If True, also produce an 'uncertain' subfolder. If the margin between the
        top-1 and top-2 *decision scores* (mean or max, depending on use_mean) is < uncertainty_threshold,
        the image is routed to 'uncertain'.
    uncertainty_threshold : float
        Margin threshold for uncertainty test.
    scores_json_name : str
        Filename for the JSON containing similarity scores (written under output_folder).

    Notes
    -----
    - Decision scores are per-category. Uncertainty is computed on those decision scores to
      match the active decision rule (mean vs max).
    - JSON structure includes per-label, per-category mean/max, decision scores, chosen label/category, etc.
    - GPU: process batches sequentially (stable VRAM). CPU: parallel batches using threads.
    """

    # ---------------------------------------------------------------------
    # Prepare folders
    # ---------------------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)
    for cat in categories:
        path = os.path.join(output_folder, cat)
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                try:
                    os.remove(fp)
                except OSError:
                    pass
        else:
            os.makedirs(path, exist_ok=True)

    uncertain_folder = os.path.join(output_folder, "uncertain") if include_uncertain else None
    if include_uncertain:
        os.makedirs(uncertain_folder, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load model and embed labels once
    # ---------------------------------------------------------------------
    model, tokenizer, image_processor, device, use_amp = load_clip_model(use_gpu=use_gpu)

    all_labels, label_to_category, category_to_indices = _compute_category_maps(categories)

    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    amp_ctx = _amp_autocast(device, use_amp)
    with torch.no_grad(), amp_ctx:
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # ---------------------------------------------------------------------
    # Gather files
    # ---------------------------------------------------------------------
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_folder, pat)))
    files.sort()
    if not files:
        print(f"[CLIP] No images found in {input_folder}")
        return

    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

    # ---------------------------------------------------------------------
    # Shared collector for JSON output (thread-safe)
    # ---------------------------------------------------------------------
    results: List[dict] = []
    results_lock = Lock()

    # ---------------------------------------------------------------------
    # Helper to compute category-level decision vectors from raw label sims
    # ---------------------------------------------------------------------
    def _per_category_scores(score_vec: np.ndarray) -> Tuple[dict, dict, dict]:
        """
        Given a 1D score vector over all labels, compute:
          - per_category_mean
          - per_category_max
          - decision_scores (mean if use_mean else max)
        Returns 3 dicts keyed by category.
        """
        per_cat_mean: Dict[str, float] = {}
        per_cat_max: Dict[str, float] = {}
        decision: Dict[str, float] = {}

        for cat, idxs in category_to_indices.items():
            if not idxs:
                per_cat_mean[cat] = float("-inf")
                per_cat_max[cat] = float("-inf")
                decision[cat] = float("-inf")
                continue
            vals = score_vec[idxs]
            m = float(np.mean(vals))
            mx = float(np.max(vals))
            per_cat_mean[cat] = m
            per_cat_max[cat] = mx
            decision[cat] = m if use_mean else mx

        return per_cat_mean, per_cat_max, decision

    # ---------------------------------------------------------------------
    # Core batch classify + collect scores
    # ---------------------------------------------------------------------
    def classify_batch(batch_paths: List[str]):
        imgs, paths = [], []

        # Load images
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                paths.append(p)
            except Exception:
                # Skip unreadable images silently
                continue

        if not imgs:
            return

        # Preprocess to tensors on the appropriate device
        inputs = image_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), amp_ctx:
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            sims = (feats @ text_embeds.T).float().cpu().numpy()  # shape: (B, L)

        # Per-image routing + JSON bookkeeping
        for i, img_path in enumerate(paths):
            score_vec = sims[i]

            # Per-label scores (include in JSON)
            per_label_scores = {all_labels[j]: float(score_vec[j]) for j in range(len(all_labels))}

            # Per-category aggregates
            per_cat_mean, per_cat_max, decision_scores = _per_category_scores(score_vec)

            # Sort decision scores for best/second-best
            sorted_decisions = sorted(decision_scores.items(), key=lambda kv: kv[1], reverse=True)
            (best_cat, best_score), (second_cat, second_score) = sorted_decisions[0], sorted_decisions[1]
            margin = float(best_score - second_score)

            # Uncertainty gate (if enabled)
            is_uncertain = include_uncertain and (margin < uncertainty_threshold)

            # Destination folder
            dest_cat = "uncertain" if is_uncertain else best_cat
            dest_folder = uncertain_folder if is_uncertain else os.path.join(output_folder, best_cat)
            dest_path = os.path.join(dest_folder, os.path.basename(img_path))

            # Fast copy; fallback to PIL save if needed
            try:
                shutil.copy2(img_path, dest_path)
            except Exception:
                try:
                    Image.open(img_path).save(dest_path)
                except Exception:
                    pass

            # Record entry for JSON
            entry = {
                "filename": os.path.basename(img_path),
                "source_path": img_path,
                "routed_to": dest_cat,
                "use_mean": bool(use_mean),
                "uncertain": bool(is_uncertain),
                "uncertainty_threshold": float(uncertainty_threshold),
                "margin_top2": margin,
                "top1": {"category": best_cat, "score": float(best_score)},
                "top2": {"category": second_cat, "score": float(second_score)},
                "per_label": per_label_scores,
                "per_category_mean": per_cat_mean,
                "per_category_max": per_cat_max,
                "decision_scores": decision_scores,  # the actual vector used for routing
            }

            with results_lock:
                results.append(entry)

    # ---------------------------------------------------------------------
    # Execute over batches (GPU sequential, CPU threaded)
    # ---------------------------------------------------------------------
    if device.startswith("cuda"):
        for batch in batches:
            classify_batch(batch)
    else:
        if workers is None:
            workers = max(1, multiprocessing.cpu_count() - 2)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(classify_batch, batch) for batch in batches]
            for fut in as_completed(futures):
                fut.result()  # re-raise errors

    # ---------------------------------------------------------------------
    # Write JSON once at the end
    # ---------------------------------------------------------------------
    json_path = os.path.join(output_folder, scores_json_name)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "input_folder": input_folder,
                "output_folder": output_folder,
                "use_mean": bool(use_mean),
                "include_uncertain": bool(include_uncertain),
                "uncertainty_threshold": float(uncertainty_threshold),
                "categories": categories,
                "labels": all_labels,
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[CLIP] Wrote similarity scores JSON: {json_path}")
    except Exception as e:
        print(f"[CLIP] Failed to write JSON at {json_path}: {e}")

    print(f"[CLIP] Done. Sorted {len(files)} images into subfolders of '{output_folder}'.")

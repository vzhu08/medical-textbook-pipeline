"""
src/filter_with_clip.py

Purpose
-------
Classify images into category subfolders using CLIP and write a single JSON of
similarity and decision scores in the output root plus timestamped historical
snapshots. Optional "uncertain" bucket routes files whose top-1 vs top-2
decision margin is below a threshold.

Flow
----
1) If output_folder/clip_similarity_scores.json and category folders exist, reuse cached outputs.
2) Prepare folders under output_folder for each category (+ optional 'uncertain').
3) Load CLIP once, tokenize all label prompts, compute normalized text embeddings.
4) Collect input images from input_folder.
5) Process in batches:
   - compute image embeddings
   - cosine similarities to all labels
   - aggregate per-category scores (mean and max)
   - pick category by mean or max (use_mean flag)
   - apply uncertainty test on decision scores; route if needed
   - copy image to destination subfolder
   - record per-image details for JSON
6) Write output_folder/clip_similarity_scores.json with scores and routing,
   plus a timestamped copy under output_folder/clip_similarity_scores_runs/.

Outputs
-------
- output_folder/<category>/        : routed images
- output_folder/uncertain/         : optional, if include_uncertain=True
- output_folder/clip_similarity_scores.json : all per-image scores and decisions
- output_folder/clip_similarity_scores_runs/clip_similarity_scores_<run_id>.json
                                      : timestamped historical score snapshots

Access point
----------
    filter_with_clip(
        input_folder, output_folder, categories,
        use_mean=False, batch_size=32, workers=None, use_gpu=True,
        include_uncertain=False, uncertainty_threshold=0.01,
        scores_json_name="clip_similarity_scores.json",
    ) -> None

Notes
-----
- Decision scores are per-category. When use_mean=True, decisions use category means;
  otherwise they use category maxima. The uncertainty check compares those decision scores.
- Existing score JSONs plus expected category folders are treated as complete stage outputs.
- GPU runs sequential batches with autocast; CPU uses a thread pool for batches.
- JSON includes per-label scores, per-category mean/max, the decision vector,
  chosen category, uncertainty flag, and margin between top-2 decisions.
"""

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
from datetime import datetime, timezone

# Prefer higher matmul perf on recent NVIDIA GPUs; safe no-op elsewhere.
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _amp_autocast(device: str, enabled: bool):
    """
    Autocast context manager on CUDA; no-op otherwise.
    """
    if enabled and device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_clip_model(use_gpu: bool | None = None, device: str | None = None):
    """
    Load CLIP model, tokenizer, and image processor.

    Device/Dtype
    ------------
    - If CUDA available and allowed, run on 'cuda' with fp16 weights.
    - Else run on CPU with float32.
    Returns (model, tokenizer, image_processor, device, use_amp).
    """
    if device is None:
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    use_amp = (device.startswith("cuda") and torch.cuda.is_available())

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
      - all_labels: list of all label strings in input order
      - label_to_category: parallel list with each label's category
      - category_to_indices: map category -> indices of its labels in all_labels
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
    use_gpu: bool = True,
    include_uncertain: bool = False,
    uncertainty_threshold: float = 0.01,
    scores_json_name: str = "clip_similarity_scores.json",
):
    """
    Categorize images from input_folder into subfolders of output_folder using CLIP.

    Parameters
    ----------
    use_mean : bool
        If True, decide by per-category MEAN score. If False, decide by per-category MAX.
    include_uncertain : bool
        If True, route to 'uncertain' when (top1 - top2) < uncertainty_threshold
        computed on the decision scores (mean or max).
    scores_json_name : str
        Filename for the similarity JSON written under output_folder.

    JSON contents
    -------------
    Per-image record includes:
      - per_label scores
      - per_category_mean and per_category_max
      - decision_scores vector used for routing
      - chosen category and uncertainty info
    """

    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, scores_json_name)
    cache_dirs = [os.path.join(output_folder, cat) for cat in categories]
    if include_uncertain:
        cache_dirs.append(os.path.join(output_folder, "uncertain"))
    if os.path.exists(json_path) and all(os.path.isdir(path) for path in cache_dirs):
        print(f"[CLIP] Using cached outputs: {json_path}")
        return

    # ---------------------------------------------------------------------
    # Prepare destination folders
    # ---------------------------------------------------------------------
    dest_folders = [os.path.join(output_folder, cat) for cat in categories]
    uncertain_folder = os.path.join(output_folder, "uncertain") if include_uncertain else None
    if include_uncertain:
        dest_folders.append(uncertain_folder)

    for path in dest_folders:
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                try:
                    os.remove(fp)
                except OSError:
                    pass
        else:
            os.makedirs(path, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load model once and embed all labels
    # ---------------------------------------------------------------------
    model, tokenizer, image_processor, device, use_amp = load_clip_model(use_gpu=use_gpu)

    all_labels, _, category_to_indices = _compute_category_maps(categories)

    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    amp_ctx = _amp_autocast(device, use_amp)
    with torch.no_grad(), amp_ctx:
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # ---------------------------------------------------------------------
    # Gather inputs
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
    # Shared collector for JSON
    # ---------------------------------------------------------------------
    results: List[dict] = []
    results_lock = Lock()

    # ---------------------------------------------------------------------
    # Helper: reduce per-label scores to per-category aggregates
    # ---------------------------------------------------------------------
    def _per_category_scores(score_vec: np.ndarray) -> Tuple[dict, dict, dict]:
        """
        From a 1D label score vector, compute:
          - per_category_mean
          - per_category_max
          - decision_scores = mean or max per category (matches use_mean)
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
    # Batch classify and collect scores
    # ---------------------------------------------------------------------
    def classify_batch(batch_paths: List[str]):
        imgs, paths = [], []

        # Load images robustly; skip unreadable files
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                paths.append(p)
            except Exception:
                continue

        if not imgs:
            return

        # Preprocess → embeddings → similarities
        inputs = image_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), amp_ctx:
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            sims = (feats @ text_embeds.T).float().cpu().numpy()  # (B, L)

        # Route images and record JSON rows
        for i, img_path in enumerate(paths):
            score_vec = sims[i]

            # Per-label scores
            per_label_scores = {all_labels[j]: float(score_vec[j]) for j in range(len(all_labels))}

            # Per-category aggregates
            per_cat_mean, per_cat_max, decision_scores = _per_category_scores(score_vec)

            # Top-1 and top-2 decision scores
            sorted_decisions = sorted(decision_scores.items(), key=lambda kv: kv[1], reverse=True)
            (best_cat, best_score), (second_cat, second_score) = sorted_decisions[0], sorted_decisions[1]
            margin = float(best_score - second_score)

            # Uncertainty
            is_uncertain = include_uncertain and (margin < uncertainty_threshold)

            # Destination
            dest_cat = "uncertain" if is_uncertain else best_cat
            dest_folder = uncertain_folder if is_uncertain else os.path.join(output_folder, best_cat)
            dest_path = os.path.join(dest_folder, os.path.basename(img_path))

            # Copy; fall back to PIL save if metadata-preserving copy fails
            try:
                shutil.copy2(img_path, dest_path)
            except Exception:
                try:
                    Image.open(img_path).save(dest_path)
                except Exception:
                    pass

            # JSON row
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
                "decision_scores": decision_scores,
            }

            with results_lock:
                results.append(entry)

    # ---------------------------------------------------------------------
    # Execute over batches: GPU sequential, CPU threaded
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
                fut.result()

    # ---------------------------------------------------------------------
    # Write scores JSON once at the end
    # ---------------------------------------------------------------------
    created_dt = datetime.now(timezone.utc)
    run_id_base = created_dt.strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(output_folder, "clip_similarity_scores_runs")
    os.makedirs(run_dir, exist_ok=True)

    def _snapshot_path_for(run_id_candidate: str) -> str:
        stem, ext = os.path.splitext(scores_json_name)
        return os.path.join(run_dir, f"{stem}_{run_id_candidate}{ext or '.json'}")

    run_id = run_id_base
    snapshot_path = _snapshot_path_for(run_id)
    suffix = 2
    while os.path.exists(snapshot_path):
        run_id = f"{run_id_base}_{suffix:02d}"
        snapshot_path = _snapshot_path_for(run_id)
        suffix += 1

    payload = {
        "created_utc": created_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "input_folder": input_folder,
        "output_folder": output_folder,
        "use_mean": bool(use_mean),
        "include_uncertain": bool(include_uncertain),
        "uncertainty_threshold": float(uncertainty_threshold),
        "categories": categories,
        "labels": all_labels,
        "results": results,
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CLIP] Wrote similarity scores JSON: {json_path}")
    except Exception as e:
        print(f"[CLIP] Failed to write JSON at {json_path}: {e}")

    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CLIP] Wrote run snapshot JSON: {snapshot_path}")
    except Exception as e:
        print(f"[CLIP] Failed to write run snapshot JSON at {snapshot_path}: {e}")

    print(f"[CLIP] Done. Sorted {len(files)} images into subfolders of '{output_folder}'.")

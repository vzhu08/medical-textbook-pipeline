# src/filter_with_clip.py

import os
import glob
import math
import shutil
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from contextlib import nullcontext

# Prefer fast matmul on Ampere+/RTX
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def _amp_autocast(device: str, enabled: bool):
    # Use new torch.amp.autocast API on CUDA; no-op elsewhere
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
        torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()
    return model, tokenizer, image_processor, device, use_amp


def get_image_label_similarities(
    image_path: str,
    categories: dict[str, list[str]],
    use_gpu: bool | None = None,
):
    """
    Compute CLIP cosine similarities for a single image across multiple categories.
    Returns:
      {
        "label_scores": { category: {label: score, ...}, ... },
        "mean_scores":  { category: mean_score, ... },
        "classification": category with highest mean score
      }
    """
    model, tokenizer, image_processor, device, use_amp = load_clip_model(use_gpu=use_gpu)

    # Flatten labels and track their category
    all_labels, label_to_category = [], []
    for category, labels in categories.items():
        for label in labels:
            all_labels.append(label)
            label_to_category.append(category)

    # Embed labels once
    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    amp_ctx = _amp_autocast(device, use_amp)
    with torch.no_grad(), amp_ctx:
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Load & embed image
    img = Image.open(image_path).convert("RGB")
    img_inputs = image_processor(images=img, return_tensors="pt")
    img_inputs = {k: v.to(device, non_blocking=True) for k, v in img_inputs.items()}

    with torch.no_grad(), amp_ctx:
        img_feat = model.get_image_features(pixel_values=img_inputs["pixel_values"])
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        sims = (img_feat @ text_embeds.T).squeeze(0).float().cpu().numpy()

    # Collect per-category scores and mean
    label_scores, mean_scores = {}, {}
    for category in categories:
        scores = []
        label_scores[category] = {}
        for idx, label in enumerate(all_labels):
            if label_to_category[idx] == category:
                sc = float(sims[idx])
                label_scores[category][label] = sc
                scores.append(sc)
        mean_scores[category] = float(np.mean(scores)) if scores else float("-inf")

    classification = max(mean_scores, key=mean_scores.get)

    return {
        "label_scores": label_scores,
        "mean_scores": mean_scores,
        "classification": classification
    }


def filter_with_clip(
    input_folder: str,
    output_folder: str,
    categories: dict[str, list[str]],
    use_mean: bool = False,
    batch_size: int = 32,
    workers: int = None,
    use_gpu: bool = True,  # <— NEW: toggle GPU from main
):
    """
    Categorize all images in 'input_folder' into subfolders under 'output_folder'
    based on CLIP similarities.
    - If use_mean=True: pick category by mean score across that category's prompts.
    - If use_gpu=True and CUDA available: run GPU w/ mixed precision; batches processed sequentially for best throughput.
      If CPU: process batches in parallel threads.
    """
    # Create/clean category subfolders
    for category in categories:
        path = os.path.join(output_folder, category)
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                try:
                    os.remove(fp)
                except OSError:
                    pass
        else:
            os.makedirs(path, exist_ok=True)

    # Load model + embed labels once
    model, tokenizer, image_processor, device, use_amp = load_clip_model(use_gpu=use_gpu)

    all_labels, label_to_category = [], []
    for category, labels in categories.items():
        for label in labels:
            all_labels.append(label)
            label_to_category.append(category)

    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    amp_ctx = _amp_autocast(device, use_amp)
    with torch.no_grad(), amp_ctx:
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Gather files
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_folder, pat)))
    files.sort()
    if not files:
        print(f"[CLIP] No images found in {input_folder}")
        return

    batches = [files[i: i + batch_size] for i in range(0, len(files), batch_size)]

    # Precompute indices per category
    category_to_indices: dict[str, list[int]] = {}
    for category in categories:
        category_to_indices[category] = [i for i, cat in enumerate(label_to_category) if cat == category]

    def classify_batch(batch_paths: list[str]):
        # Load images (PIL) — keep on CPU; processor will tensorize
        imgs, paths = [], []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                paths.append(p)
            except Exception:
                continue
        if not imgs:
            return

        inputs = image_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), amp_ctx:
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            sims = (feats @ text_embeds.T).float().cpu().numpy()

        for idx_img, img_path in enumerate(paths):
            scores = sims[idx_img]
            if use_mean:
                # Mean-based category
                best_cat, best_score = None, float("-inf")
                for cat, indices in category_to_indices.items():
                    if not indices:
                        continue
                    mean_score = float(np.mean(scores[indices]))
                    if mean_score > best_score:
                        best_score = mean_score
                        best_cat = cat
            else:
                # Max-label-based category
                best_label_idx = int(np.argmax(scores))
                best_cat = label_to_category[best_label_idx]

            dest = os.path.join(output_folder, best_cat, os.path.basename(img_path))
            # Fast path: copy the original file (avoid re-encode)
            try:
                shutil.copy2(img_path, dest)
            except Exception:
                # Fallback save via PIL if copy fails
                try:
                    Image.open(img_path).save(dest)
                except Exception:
                    pass

    # Execution strategy:
    # - GPU: sequential over batches (one stream) for best throughput & VRAM stability
    # - CPU: parallel threads
    if device.startswith("cuda"):
        for batch in batches:
            classify_batch(batch)
    else:
        if workers is None:
            workers = max(1, multiprocessing.cpu_count() - 2)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(classify_batch, batch) for batch in batches]
            for fut in as_completed(futures):
                fut.result()

    print(f"[CLIP] Done. Sorted {len(files)} images into subfolders of '{output_folder}'.")

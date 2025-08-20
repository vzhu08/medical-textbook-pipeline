# src/filter_with_clip.py

import os
import glob
import math
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def load_clip_model(device: str = None):
    """
    Load CLIP model, tokenizer, and image processor.
    Returns (model, tokenizer, image_processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()
    return model, tokenizer, image_processor, device


def get_image_label_similarities(
    image_path: str,
    categories: dict[str, list[str]]
) -> dict:
    """
    Compute CLIP cosine similarities for a single image across multiple categories.
    'categories' maps subfolder names to lists of label prompts.

    Returns:
      {
        "label_scores": { category: {label: score, ...}, ... },
        "mean_scores":  { category: mean_score, ... },
        "classification": category with highest mean score
      }
    """
    # load model + tokenizer + processor
    model, tokenizer, image_processor, device = load_clip_model()

    # flatten labels and track their category
    all_labels = []
    label_to_category = []
    for category, labels in categories.items():
        for label in labels:
            all_labels.append(label)
            label_to_category.append(category)

    # embed all labels once
    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # load & embed image
    img = Image.open(image_path).convert("RGB")
    img_inputs = image_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = model.get_image_features(pixel_values=img_inputs["pixel_values"])
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        sims = (img_feat @ text_embeds.T).squeeze(0).cpu().numpy()

    # collect per-category label scores and mean
    label_scores = {}
    mean_scores = {}
    for category in categories:
        scores = []
        label_scores[category] = {}
        for idx, label in enumerate(all_labels):
            if label_to_category[idx] == category:
                score = float(sims[idx])
                label_scores[category][label] = score
                scores.append(score)
        mean_scores[category] = float(np.mean(scores)) if scores else float("-inf")

    # pick highest mean
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
    max_workers: int = None
):
    """
    Categorize all images in 'input_folder' into subfolders under 'output_folder' based on
    CLIP similarities. 'categories' maps each subfolder name to a list of label prompts.
    If use_mean is True, uses the mean score per category; otherwise picks the category
    of the single highest-scoring label. Processes in parallel threads.
    """
    # create/clean category subfolders
    for category in categories:
        path = os.path.join(output_folder, category)
        if os.path.isdir(path):
            for f in os.listdir(path):
                try:
                    os.remove(os.path.join(path, f))
                except OSError:
                    pass
        else:
            os.makedirs(path, exist_ok=True)

    # load model + embed labels
    model, tokenizer, image_processor, device = load_clip_model()
    all_labels = []
    label_to_category = []
    for category, labels in categories.items():
        for label in labels:
            all_labels.append(label)
            label_to_category.append(category)

    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # prepare files & batches
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_folder, pat)))
    files.sort()
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

    # precompute indices per category
    category_to_indices: dict[str, list[int]] = {}
    for category in categories:
        category_to_indices[category] = [
            i for i, cat in enumerate(label_to_category) if cat == category
        ]

    # thread worker
    def _worker(batch_paths: list[str]):
        imgs, paths = [], []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                paths.append(p)
            except:
                continue
        if not imgs:
            return

        inputs = image_processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            sims = (feats @ text_embeds.T).cpu().numpy()

        for idx_img, img_path in enumerate(paths):
            scores = sims[idx_img]
            if use_mean:
                # mean-based category
                best_cat = None
                best_score = float("-inf")
                for cat, indices in category_to_indices.items():
                    if not indices:
                        continue
                    mean_score = float(np.mean(scores[indices]))
                    if mean_score > best_score:
                        best_score = mean_score
                        best_cat = cat
            else:
                # max-label-based category
                best_label_idx = int(np.argmax(scores))
                best_cat = label_to_category[best_label_idx]

            dest = os.path.join(output_folder, best_cat, os.path.basename(img_path))
            try:
                Image.open(img_path).save(dest)
            except:
                pass

    # parallel execution
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, batch) for batch in batches]
        for fut in as_completed(futures):
            fut.result()

    print(f"Done. Sorted images into subfolders of '{output_folder}' based on categories.")

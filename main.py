# main.py

import os
import glob
import time

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline control flags
# ──────────────────────────────────────────────────────────────────────────────
RUN_EXTRACTION = False          # Extract images from PDF
RUN_FILTER_PHOTO = True        # Filter images into photos vs illustrations
RUN_FILTER_SKIN = True         # Filter photos into skin vs no_skin
RUN_SKIN_CLASSER = True        # Identify skin tone
RUN_TEXT = False               # Run text analysis

WORKERS = 12
USE_GPU = True

abd_model_path = "models/abd-skin-segmentation/final_unet_pytorch.pth"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {sec:.1f}s"
    h, rem = divmod(m, 60)
    return f"{int(h)}h {int(rem)}m {sec:.0f}s"

def _banner(msg: str) -> None:
    bar = "─" * max(10, len(msg) + 2)
    print(f"\n{bar}\n {msg}\n{bar}", flush=True)

def _timed_call(stage_name, fn, *args, **kwargs):
    print(f"[{stage_name}] start...", flush=True)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"[{stage_name}] done in {_fmt_secs(dt)}", flush=True)
    return result, dt


# ──────────────────────────────────────────────────────────────────────────────
# Per‑PDF processing
# ──────────────────────────────────────────────────────────────────────────────
def process_pdf(pdf_path: str):
    _banner(f"Processing PDF: {os.path.basename(pdf_path)}")
    # Derive base name (without extension) for this textbook
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Create a separate data directory for each textbook
    base_dir = os.path.join("data", base_name)
    extracted_dir = os.path.join(base_dir, "extracted_images")
    sorted_dir = os.path.join(base_dir, "sorted_images")
    photo_sort_dir = os.path.join(sorted_dir, "photo_illus")
    skin_sort_dir = os.path.join(sorted_dir, "if_skin")
    skin_class_dir = os.path.join(base_dir, "skin_class")
    text_dir = os.path.join(base_dir, "text_analysis")

    # Ensure directories exist
    for d in (
        base_dir,
        extracted_dir,
        sorted_dir,
        photo_sort_dir,
        skin_sort_dir,
        skin_class_dir,
        text_dir
    ):
        os.makedirs(d, exist_ok=True)

    # Stage timings (per PDF)
    stage_times = {}

    # Step 1: Extract all images from the PDF
    if RUN_EXTRACTION:
        from src.extract_images import extract_images_pipeline
        _, dt = _timed_call(
            "1/5 Extract Images",
            extract_images_pipeline,
            pdf_path=pdf_path,
            output_dir=extracted_dir,
            workers = WORKERS,
        )
        stage_times["extract"] = dt
    else:
        print("[1/5 Extract Images] skipped (flag off)")
        stage_times["extract"] = 0.0

    # Step 2: First-level CLIP filtering (photos vs illustrations)
    if RUN_FILTER_PHOTO:
        from src.filter_with_clip import filter_with_clip
        photo_labels = [
            "a photograph",
            "a high-resolution photo",
            "a low-resolution photo",
            "a real-life photo",
            "a photo taken with a camera",
        ]
        illus_labels = [
            "a drawing",
            "a textbook illustration",
            "a computer generated illustration"
        ]
        text_labels = [
            "a portion of text",
            "text on a page",
            "a blank page"
        ]
        _, dt = _timed_call(
            "2/5 CLIP Filter: photo vs illustration",
            filter_with_clip,
            input_folder=os.path.join(extracted_dir, "bbox_crops"),
            output_folder=photo_sort_dir,
            use_mean=True,
            categories={
                "photo": photo_labels,
                "illus": illus_labels,
                "text": text_labels
            },
            workers=WORKERS,
            use_gpu = USE_GPU
        )
        stage_times["clip_photo"] = dt
    else:
        print("[2/5 CLIP Filter: photo vs illustration] skipped (flag off)")
        stage_times["clip_photo"] = 0.0

    # Step 3: Second-level CLIP filtering (skin vs no_skin)
    if RUN_FILTER_SKIN:
        from src.filter_with_clip import filter_with_clip
        skin_labels = [
            "a human",
            "human skin",
            "a skin condition",
            "a leg",
            "a foot",
            "an arm",
            "a hand",
            "a patient's torso",
            "a patient's back",
            "a face",
            "a mouth",
            "a tongue",
            "an ear",
            "a nose",
            "a finger",
            "a patient's hair",
            "a wart",
        ]
        noskin_labels = [
            "a microscopic photo",
            "a photo through a microscope",
            "a biological cell",
            "a group of cells",
            "a molecule",
            "medical equipment",
            "a medical experiment",
            "a tool",
            "a page with text",
            "a blank page",
            "a screenshot with text",
            "a photograph of internal anatomy",
            "the inside of a mouth",
            "internal photograph of an organ"
        ]
        _, dt = _timed_call(
            "3/5 CLIP Filter: skin vs no_skin",
            filter_with_clip,
            input_folder=os.path.join(photo_sort_dir, "photo"),
            output_folder=skin_sort_dir,
            use_mean=False,
            categories={
                "skin": skin_labels,
                "no_skin": noskin_labels,
            },
            workers=WORKERS,
            use_gpu = USE_GPU
        )
        stage_times["clip_skin"] = dt
    else:
        print("[3/5 CLIP Filter: skin vs no_skin] skipped (flag off)")
        stage_times["clip_skin"] = 0.0

    # Step 4: Skin tone classification
    if RUN_SKIN_CLASSER:
        from src.skin_classification import classify_skin
        _, dt = _timed_call(
            "4/5 Skin Classification",
            classify_skin,
            input_dir=os.path.join(skin_sort_dir, "skin"),
            output_dir=skin_class_dir,
            abd_model_path=abd_model_path,
            workers=WORKERS,
            use_gpu = USE_GPU
        )
        stage_times["skin_class"] = dt
    else:
        print("[4/5 Skin Classification] skipped (flag off)")
        stage_times["skin_class"] = 0.0

    # Step 5: Text analysis (optional)
    if RUN_TEXT:
        from src.text_parser import analyze_text
        _, dt = _timed_call(
            "5/5 Text Analysis",
            analyze_text,
            input_folder=base_dir,
            output_folder=text_dir,
        )
        stage_times["text"] = dt
    else:
        print("[5/5 Text Analysis] skipped (flag off)")
        stage_times["text"] = 0.0

    # Summary for this PDF
    total = sum(stage_times.values())
    _banner(f"Finished '{base_name}' in {_fmt_secs(total)}")
    print("Stage breakdown:")
    for k, v in stage_times.items():
        print(f"  - {k:12s}: {_fmt_secs(v)}")
    print("", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Find all PDF files in the input folder
    input_pattern = os.path.join("textbook_inputs", "*.pdf")
    all_pdfs = sorted(glob.glob(input_pattern))

    if not all_pdfs:
        print("No PDF files found in 'textbook_inputs' directory.")
        return

    _banner(f"Found {len(all_pdfs)} PDF(s)")
    grand_t0 = time.perf_counter()

    for idx, pdf_path in enumerate(all_pdfs, start=1):
        print(f"[{idx}/{len(all_pdfs)}] {os.path.basename(pdf_path)}", flush=True)
        t0 = time.perf_counter()
        process_pdf(pdf_path)
        dt = time.perf_counter() - t0
        print(f"[{idx}/{len(all_pdfs)}] Done in {_fmt_secs(dt)}\n", flush=True)

    grand_dt = time.perf_counter() - grand_t0
    _banner(f"Pipeline complete for all textbooks in {_fmt_secs(grand_dt)}")


if __name__ == "__main__":
    main()

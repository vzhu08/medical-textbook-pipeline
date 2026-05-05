# main.py

import os
import glob
import time
import json
import tempfile
import textwrap
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline control flags
# ──────────────────────────────────────────────────────────────────────────────
RUN_EXTRACTION = True           # Extract images from PDF
RUN_FILTER_PHOTO = True         # Filter images into photos vs illustrations
RUN_FILTER_SKIN = False          # Filter photos into skin vs no_skin
RUN_FILTER_GENDER = False        # Filter photos into male vs female
RUN_FILTER_RACE = False          # Filter photos into putative race
RUN_SKIN_CLASSER = False         # Identify skin tone
RUN_TEXT = False                 # Run text analysis
RUN_SUMMARIZE = True            # Build final Image/Text datasets

WORKERS = 12
USE_GPU = True

abd_model_path = "models/abd-skin-segmentation/final_unet_pytorch.pth"


# ──────────────────────────────────────────────────────────────────────────────
# Helper
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
# Subprocess utility: import a module and call a function(**kwargs) in a CLEAN
# Python interpreter. We pass kwargs via a temp JSON file to avoid shell quoting
# issues and to support complex types (lists, dicts, etc.).
# ──────────────────────────────────────────────────────────────────────────────

def _run_stage_subprocess(module_path: str, func_name: str, kwargs: dict, *, env=None, cwd: str | None = None) -> None:
    """Execute `from <module_path> import <func_name>; <func_name>(**kwargs)`
    in a fresh Python interpreter using a tiny -c bootstrap and a temp JSON file.

    - `module_path`: e.g., 'src.image_extraction'
    - `func_name`:   e.g., 'run_image_extraction'
    - `kwargs`:      call arguments (must be JSON-serializable)
    - `env`:         optional environment overrides
    - `cwd`:         working dir (defaults to project root)
    """
    bootstrap = textwrap.dedent(
        f"""
        import json, sys, importlib
        args_path = sys.argv[1]
        with open(args_path, 'r', encoding='utf-8') as f:
            kwargs = json.load(f)
        mod = importlib.import_module('{module_path}')
        fn = getattr(mod, '{func_name}')
        rv = fn(**kwargs)
        """
    ).strip()

    with tempfile.TemporaryDirectory() as td:
        args_json = Path(td) / "kwargs.json"
        args_json.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding="utf-8")

        child_env = os.environ.copy()
        if env:
            child_env.update(env)

        cmd = [sys.executable, "-c", bootstrap, str(args_json)]
        proc = subprocess.run(cmd, env=child_env, cwd=cwd or os.getcwd())
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess for {module_path}.{func_name} failed with exit code {proc.returncode}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Per-PDF processing (subprocess calls for each step)
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
    gender_sort_dir = os.path.join(sorted_dir, "gender")
    race_sort_dir = os.path.join(sorted_dir, "race")
    skin_class_dir = os.path.join(base_dir, "skin_class")
    text_dir = os.path.join(base_dir, "text_analysis")
    final_dir = os.path.join(base_dir, "final_datasets")  # new stage output

    # Ensure directories exist
    for d in (
        base_dir,
        extracted_dir,
        sorted_dir,
        photo_sort_dir,
        skin_sort_dir,
        gender_sort_dir,
        race_sort_dir,
        skin_class_dir,
        text_dir,
        final_dir,   # ensure output exists for summary stage
    ):
        os.makedirs(d, exist_ok=True)

    # Stage timings (per PDF)
    stage_times: dict[str, float] = {}

    # Step 1: Extract all images from the PDF (subprocess)
    if RUN_EXTRACTION:
        def _extract():
            _run_stage_subprocess(
                module_path="src.image_extraction",
                func_name="run_image_extraction",
                kwargs={
                    "pdf_path": pdf_path,
                    "out_dir": extracted_dir,
                    "workers": WORKERS,
                    "save_mode": "final"
                },
            )
        _, dt = _timed_call("1/8 Extract Images", _extract)
        stage_times["extract"] = dt
    else:
        print("[1/8 Extract Images] skipped (flag off)")
        stage_times["extract"] = 0.0

    # Step 2: First-level CLIP filtering (photos vs illustrations) (subprocess)
    if RUN_FILTER_PHOTO:
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
            "a computer generated illustration",
        ]
        text_labels = [
            "a portion of text",
            "text on a page",
            "a blank page",
        ]

        def _clip_photo():
            _run_stage_subprocess(
                module_path="src.filter_with_clip",
                func_name="filter_with_clip",
                kwargs={
                    "input_folder": os.path.join(extracted_dir, "bbox_crops"),
                    "output_folder": photo_sort_dir,
                    "use_mean": True,
                    "categories": {
                        "photo": photo_labels,
                        "illus": illus_labels,
                        "text": text_labels,
                    },
                    "workers": WORKERS,
                    "use_gpu": USE_GPU,
                },
            )
        _, dt = _timed_call("2/8 CLIP Filter: photo vs illustration", _clip_photo)
        stage_times["clip_photo"] = dt
    else:
        print("[2/8 CLIP Filter: photo vs illustration] skipped (flag off)")
        stage_times["clip_photo"] = 0.0

    # Step 3: Second-level CLIP filtering (skin vs no_skin) (subprocess)
    if RUN_FILTER_SKIN:
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
            "internal photograph of an organ",
        ]

        def _clip_skin():
            _run_stage_subprocess(
                module_path="src.filter_with_clip",
                func_name="filter_with_clip",
                kwargs={
                    "input_folder": os.path.join(photo_sort_dir, "photo"),
                    "output_folder": skin_sort_dir,
                    "use_mean": False,
                    "categories": {
                        "skin": skin_labels,
                        "no_skin": noskin_labels,
                    },
                    "workers": WORKERS,
                    "use_gpu": USE_GPU,
                },
            )
        _, dt = _timed_call("3/8 CLIP Filter: skin vs no_skin", _clip_skin)
        stage_times["clip_skin"] = dt
    else:
        print("[3/8 CLIP Filter: skin vs no_skin] skipped (flag off)")
        stage_times["clip_skin"] = 0.0

    # Step 4: Third-level CLIP filtering (male vs female) (subprocess)
    if RUN_FILTER_GENDER:
        male_labels = [
            "a biological male",
            "a human male",
            "a man",
            "a male patient",
            "a male body",
            "male genitalia",
            "a penis"
        ]
        female_labels = [
            "a biological female",
            "a human female",
            "a woman",
            "a female patient",
            "a female body",
            "human female breasts",
            "female genitalia",
            "a vagina",
        ]

        def _clip_gender():
            _run_stage_subprocess(
                module_path="src.filter_with_clip",
                func_name="filter_with_clip",
                kwargs={
                    "input_folder": os.path.join(skin_sort_dir, "skin"),
                    "output_folder": gender_sort_dir,
                    "use_mean": False,
                    "include_uncertain": True,
                    "uncertainty_threshold": 0.01,
                    "categories": {
                        "male": male_labels,
                        "female": female_labels,
                    },
                    "workers": WORKERS,
                    "use_gpu": USE_GPU,
                },
            )

        _, dt = _timed_call("4/8 CLIP Filter: male vs female", _clip_gender)
        stage_times["_clip_gender"] = dt
    else:
        print("[4/8 CLIP Filter: male vs female] skipped (flag off)")
        stage_times["_clip_gender"] = 0.0

    # Step 5: Fourth-level CLIP filtering (putative race) (subprocess)
    if RUN_FILTER_RACE:
        black_labels = [
            "a body part of a Black person",
            "a body part with Black skin",
            "a Black patient",
            "a Black patient's skin",
            "a Black person",
            "an African person",
            "an African American person",
            "a Black person's skin",
            "an African person's skin",
        ]
        white_labels = [
            "a body part of a White person",
            "a body part with White skin",
            "a White patient",
            "a White patient's skin",
            "a White person",
            "a Caucasian person",
            "an Eastern European person",
            "a White person's skin",
        ]
        asian_labels = [
            "a body part of an Asian person",
            "a body part with Asian skin",
            "an Asian patient",
            "an Asian patient's skin",
            "an Asian person",
            "an Asian person's skin",
            "Asian skin"
        ]
        latine_labels = [
            "a body part of a Latine person",
            "a body part with Latine skin",
            "a Latine patient",
            "a Latino patient",
            "a Latine patient's skin",
            "a Latino patient's skin",
            "a Latine person",
            "a Latinx person",
            "a Latino person",
            "Latine skin",
            "Latinx skin",
            "Latino skin"
        ]

        def _clip_race():
            _run_stage_subprocess(
                module_path="src.filter_with_clip",
                func_name="filter_with_clip",
                kwargs={
                    "input_folder": os.path.join(skin_sort_dir, "skin"),
                    "output_folder": race_sort_dir,
                    "use_mean": False,
                    "include_uncertain": True,
                    "uncertainty_threshold": 0.005,
                    "categories": {
                        "black": black_labels,
                        "white": white_labels,
                        "asian": asian_labels,
                        "latine": latine_labels,
                    },
                    "workers": WORKERS,
                    "use_gpu": USE_GPU,
                },
            )

        _, dt = _timed_call("5/8 CLIP Filter: putative race", _clip_race)
        stage_times["_clip_race"] = dt
    else:
        print("[5/8 CLIP Filter: putative race] skipped (flag off)")
        stage_times["_clip_race"] = 0.0

    # Step 6: Skin tone classification (subprocess)
    if RUN_SKIN_CLASSER:
        def _skin_class():
            _run_stage_subprocess(
                module_path="src.skin_classification",
                func_name="classify_skin",
                kwargs={
                    "input_dir": os.path.join(skin_sort_dir, "skin"),
                    "output_dir": skin_class_dir,
                    "abd_model_path": abd_model_path,
                    "workers": WORKERS,
                    "use_gpu": USE_GPU,
                },
            )
        _, dt = _timed_call("6/8 Skin Classification", _skin_class)
        stage_times["skin_class"] = dt
    else:
        print("[6/8 Skin Classification] skipped (flag off)")
        stage_times["skin_class"] = 0.0

    # Step 7: Text analysis (optional) (subprocess)
    if RUN_TEXT:
        def _text():
            _run_stage_subprocess(
                module_path="src.text_parser",
                func_name="analyze_text_llm",
                kwargs={
                    "input_dir": extracted_dir,
                    "output_dir": text_dir,
                },
            )
        _, dt = _timed_call("7/8 Text Analysis", _text)
        stage_times["text"] = dt
    else:
        print("[7/8 Text Analysis] skipped (flag off)")
        stage_times["text"] = 0.0

    # Step 8: Final dataset summarization (subprocess)
    if RUN_SUMMARIZE:
        def _summarize():
            # Use the book's base_dir as input; write CSVs to final_dir
            _run_stage_subprocess(
                module_path="src.summarize_results",
                func_name="summarize_results",
                kwargs={
                    "input_dir": base_dir,
                    "output_dir": final_dir,
                },
            )
        _, dt = _timed_call("8/8 Final Datasets Summary", _summarize)
        stage_times["summarize"] = dt
    else:
        print("[8/8 Final Datasets Summary] skipped (flag off)")
        stage_times["summarize"] = 0.0

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

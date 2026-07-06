# Medical Textbook Representation Pipeline

## Reproducibility Report

This report explains the current medical textbook pipeline to provide full documentation for accurate reproducibility. It also mentions potential issues and areas for improvement. It is based on the repository files in this project, especially `main.py` and the modules in `src/`.

The goal of the pipeline is to turn medical textbook PDFs into research artifacts about representation in images and text. It extracts page text and figure crops, filters the crops with CLIP, estimates visible skin tone on skin-containing photographs, uses an OpenAI model to count explicit race and gender mentions in the extracted text, and then compiles final CSV datasets and summary plots.

This project is still very much work in progress, and the accuracy of certain modules could certainly be improved. New functionality, such as the classification of illustrations, would likely be helpful for downstream analysis.

For any questions regarding this report or the repository, contact Vincent Zhu at vzhu08@gmail.com or 610-808-7989.
## Table of Contents

| Section | What it explains                                                                                                                                        |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [1. Introduction and Overview](#1-introduction-and-overview) | The purpose of the pipeline, its major stages, its module split, the expected input/output file structure, and the subprocess design.                   |
| [2. Prerequisites and Setup](#2-prerequisites-and-setup) | The Python environment, required packages, external models, API keys, PDF naming convention, and basic setup checks needed before running the pipeline. |
| [3. Running the Pipeline](#3-running-the-pipeline) | The main command, default `main.py` configuration flags, and what each flag does.                                                                       |
| [4. Module-by-Module Guide](#4-module-by-module-guide) | The detailed step-by-step behavior of each pipeline module, including tools used, inputs, outputs, validation steps, and caveats.                       |
| [5. Safe Reruns and Cache Invalidation](#5-safe-reruns-and-cache-invalidation) | How cached artifacts behave and which folders to remove when rerunning changed stages.                                                                  |
| [6. Known Issues and Risks](#6-known-issues-and-risks) | Current technical, reproducibility, and methodological risks that should be understood before using results.                                            |
| [7. Troubleshooting](#7-troubleshooting) | Common failure modes and practical checks for missing PDFs, dependency problems, model downloads, API failures, and empty outputs.                      |
| [8. Reproducibility Checklist](#8-reproducibility-checklist) | The run metadata, files, model versions, manual reviews, and validation checks to record before accepting a run as reproducible.                        |
| [9. Handoff Checklist for the Next RA](#9-handoff-checklist-for-the-next-ra) | The materials, notes, credentials guidance, outputs, and review records to provide to the next research assistant.                                      |
| [10. Recommended Next Development Priorities](#10-recommended-next-development-priorities) | Suggested improvements for making the pipeline easier to configure, rerun, validate, and maintain.                                                      |

## 1. Introduction and Overview

### 1.1 What the Pipeline Does

For every PDF in `textbook_inputs/`, `main.py` creates a textbook-specific workspace under `data/<pdf-stem>/` and runs up to eight stages:

1. Render PDF pages, extract text, and crop likely figures.
2. Use CLIP to sort crops into photographs, illustrations, and text-like crops.
3. Use CLIP to sort photographs into skin and no-skin images.
4. Use CLIP to sort skin images into gender categories.
5. Use CLIP to sort skin images into race categories.
6. Use a skin-segmentation model plus color analysis to estimate skin tone and Monk Skin Tone category.
7. Use an OpenAI model to identify explicit race and gender mentions in textbook text.
8. Compile final image and text datasets, plots, and summary statistics.

### 1.2 Module Split

The repository is organized around one orchestration script and six main pipeline modules:

| File | Role | Main entry point |
|---|---|---|
| `main.py` | Finds PDFs, creates per-book folders, and runs stages in subprocesses | `main()`, `process_pdf(pdf_path)` |
| `src/text_extraction.py` | Renders pages, extracts native PDF text boxes, falls back to PaddleOCR when needed, and writes Markdown | `run_text_extraction(...)` |
| `src/image_extraction.py` | Whiteboxes text areas, detects likely figure regions, and saves crops | `run_image_extraction(...)` |
| `src/filter_with_clip.py` | Classifies images into prompt-defined folders with CLIP | `filter_with_clip(...)` |
| `src/skin_classification.py` | Segments skin, refines masks, estimates representative color, calculates ITA, and maps to Monk tone | `classify_skin(...)` |
| `src/text_parser.py` | Finds candidate demographic text mentions and classifies them with an OpenAI model | `analyze_text_llm(...)` |
| `src/summarize_results.py` | Builds final CSVs, plots, and `summary.txt` | `summarize_results(...)` |

`src/__init__.py` is empty and only marks `src` as a Python package.

### 1.3 File Structure

Before running the pipeline, the project should look like this:

```text
medical-textbook-pipeline/
|-- main.py
|-- README
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- text_extraction.py
|   |-- image_extraction.py
|   |-- filter_with_clip.py
|   |-- skin_classification.py
|   |-- text_parser.py
|   `-- summarize_results.py
|-- textbook_inputs/
|   `-- <book-name>_<edition>_<year>.pdf
`-- models/
    `-- abd-skin-segmentation/
        `-- final_unet_pytorch.pth
```

`textbook_inputs/`, `models/`, and `data/` are expected runtime directories. They may not be present in a fresh clone and may be ignored by Git.

After processing `textbook_inputs/fitzpatrick_9th_2023.pdf`, the main output tree is:

```text
data/fitzpatrick_9th_2023/
|-- extracted_images/
|   |-- page_images/
|   |-- text_boxes_pages/
|   |-- bbox_pages/
|   |-- bbox_crops/
|   |-- page_masked/             # populated only when save_mode="all"
|   |-- page_mask/               # populated only when save_mode="all"
|   |-- page_bbox_overlay/       # populated only when save_mode="all"
|   |-- pm4l.md
|   |-- text_structure.json
|   |-- pm4l_page_index.json
|   |-- compiled_text_boxes.json
|   |-- bboxes.json
|   `-- manifest.json
|-- sorted_images/
|   |-- photo_illus/
|   |   |-- photo/
|   |   |-- illus/
|   |   |-- text/
|   |   `-- clip_similarity_scores.json
|   |-- if_skin/
|   |   |-- skin/
|   |   |-- no_skin/
|   |   `-- clip_similarity_scores.json
|   |-- gender/
|   |   |-- male/
|   |   |-- female/
|   |   |-- uncertain/
|   |   `-- clip_similarity_scores.json
|   `-- race/
|       |-- black/
|       |-- white/
|       |-- asian/
|       |-- latine/
|       |-- uncertain/
|       `-- clip_similarity_scores.json
|-- skin_class/
|   |-- masks/
|   |-- probs/
|   |-- masked/
|   |-- clusters/
|   |-- rep_color/
|   |-- skin_tones.csv
|   `-- fallbacks.csv
|-- text_analysis/
|   |-- sentences_by_page.json
|   |-- race_candidates.json
|   |-- gender_candidates.json
|   |-- race_api_batches/
|   |-- gender_api_batches/
|   |-- race_results.json
|   `-- gender_results.json
`-- final_datasets/
    |-- image_dataset.csv
    |-- text_dataset.csv
    |-- monk_hist.jpg
    |-- lab_scatter.jpg
    `-- summary.txt
```

Some folders are created even when a stage is disabled or produces no records.

### 1.4 Artifact + Cache Design

The pipeline is artifact-driven. Modules communicate mostly by reading and writing files on disk rather than passing large Python objects in memory. This makes it easier to inspect intermediate results and rerun individual stages, but it also means stale cached files can affect later runs. When changing settings or code, manually delete outputs from past runs to ensure accurate new outputs.

### 1.5 Subprocess Design

`main.py` runs each heavy stage in a fresh Python subprocess using `_run_stage_subprocess(...)`. The parent process writes function arguments to a temporary JSON file, starts a clean Python interpreter, imports the relevant module, and calls the module entry point.

This design is rather clunky, but it was necessary to solve certain conflicts between PyTorch and PaddleOCR. There may be a cleaner solution.

## 2. Prerequisites and Setup

### 2.1 Python Environment

Use a clean virtual environment on Python 3.12.8. Version are rather finicky, so make sure the venv is set up correctly.

### 2.2 Install Dependencies

The packages used in this pipeline must be installed in a very specific order to prevent conflicts. Recommended order:

1. Create and activate a clean virtual environment.
2. Install the correct PaddlePaddle 3.1.0 build for the machine. Locate correct command at: https://www.paddlepaddle.org.cn/install/old?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html
3. If the PaddlePaddle install page gives a newer default command, adjust it to install version 3.1.0.
4. Run `pip install paddleocr[all]`.
5. Install the correct PyTorch build for the machine.
6. Install remaining requirements from `requirements.txt`.
7. Ensure GPU availability on the machine

The current `requirements.txt` lists:

```text
numpy
opencv-python
transformers
torch
PyMuPDF
spacy
nltk
matplotlib
setuptools
paddlepaddle==3.1.0
paddlex==3.1.0
paddleocr==3.1.0
cython
pydensecrf
pymupdf4llm
pandas
Pillow
scikit-learn
openai
python-dotenv
```

Reference `environment_freeze.txt` for exact versions. PaddlePaddle versions will likely differ due to different CUDA versions.

### 2.3 External Model Files

The skin-classification stage expects the ABD skin-segmentation checkpoint here:

```text
models/abd-skin-segmentation/final_unet_pytorch.pth
```
The correct model is already in the repo, but you can download it here: https://github.com/MRE-Lab-UMD/abd-skin-segmentation/blob/master/Models/final_unet_pytorch.pth

### 2.4 Input PDFs and Naming Convention

Place PDFs directly inside `textbook_inputs/`. Subdirectories are not searched.

`main.py` processes all files matching:

```text
textbook_inputs/*.pdf
```

The filename stem becomes the output folder name. The summary module expects this stem to have three underscore-separated parts:

```text
bookname_edition_year.pdf
```

Example:

```text
fitzpatrick_9th_2023.pdf
```

`src/summarize_results.py` parses:

- first part as book name
- second part as edition number
- third part as year

## 3. Running the Pipeline

### 3.1 Main Command

Run commands from the repository root:

```powershell
python main.py
```

On Windows, use the virtual environment's Python executable if `python` does not resolve to the intended interpreter:

```powershell
.\.venv\Scripts\python.exe main.py
```

### 3.2 Default `main.py` Flags

The pipeline is controlled by constants near the top of `main.py`. There is no command-line interface yet.

| Flag |                                         Default value | Meaning                                                                                            |
|---|------------------------------------------------------:|----------------------------------------------------------------------------------------------------|
| `RUN_EXTRACTION` |                                                `True` | Run text extraction, page rendering, and crop extraction                                           |
| `RUN_FILTER_PHOTO` |                                                `True` | Sort crops into photo, illustration, or text                                                       |
| `RUN_FILTER_SKIN` |                                                `True` | Sort photos into skin or no-skin                                                                   |
| `RUN_FILTER_GENDER` |                                                `True` | Sort skin photos into male, female, or uncertain                                                   |
| `RUN_FILTER_RACE` |                                               `False` | Sort skin photos into race categories or uncertain                                                 |
| `RUN_SKIN_CLASSER` |                                                `True` | Segment skin and estimate skin tone                                                                |
| `RUN_TEXT` |                                                `True` | Run OpenAI text analysis                                                                           |
| `RUN_SUMMARIZE` |                                                `True` | Build final datasets, plots, and summary files                                                     |
| `WORKERS` |                                                  `12` | Worker count passed to supported stages                                                            |
| `USE_GPU` |                                                `True` | Allow GPU use where supported (pipeline likely breaks if this is false. CPU use is not supported). |
| `abd_model_path` | `models/abd-skin-segmentation/final_unet_pytorch.pth` | ABD checkpoint path                                                                                |

The pipeline is currently intended to run without `RUN_FILTER_RACE` as it does not produce accurate results.

## 4. Module-by-Module Guide
Each section hear explains each module in detail, providing:
- The flag to enable/disable it
- The main function
- Purpose
- Tools and libraries used
- Inputs and outputs
- Step-by-step behavior
- Cache behavior
- Files to validate when checking for accuracy
- Any worthwhile notes

### 4.1 Pipeline Orchestration - `main()` and `process_pdf(pdf_path)`

Runs when:

- The user runs `python main.py` from the repository root.

Main functions:

```python
main()
process_pdf(pdf_path: str)
```

Purpose:

`main.py` is the only normal entry point. It finds PDFs, creates output folders, runs each enabled stage, and prints timing summaries.

Inputs:

- `textbook_inputs/*.pdf`
- stage flags and constants defined near the top of `main.py`
- environment variables needed by child modules, namely `OPENAI_API_KEY`
- model files under `models/`

Outputs:

- one folder per PDF under `data/<pdf-stem>/`
- stage-specific artifacts inside that folder
- console timing summaries

Step-by-step behavior:

1. `main()` builds `input_pattern = textbook_inputs/*.pdf`.
2. It sorts all matching PDFs by filename.
3. For each PDF, `process_pdf(pdf_path)` derives `base_name` from the PDF filename stem.
4. It creates:
   - `data/<base_name>/extracted_images`
   - `data/<base_name>/sorted_images`
   - `data/<base_name>/sorted_images/photo_illus`
   - `data/<base_name>/sorted_images/if_skin`
   - `data/<base_name>/sorted_images/gender`
   - `data/<base_name>/sorted_images/race`
   - `data/<base_name>/skin_class`
   - `data/<base_name>/text_analysis`
   - `data/<base_name>/final_datasets`
5. For each enabled flag, it starts a fresh Python subprocess that imports the target module and calls the target function with JSON-serialized arguments.
6. It records elapsed time for each stage and prints a final per-PDF timing breakdown.

Notes:

- Stage settings are edited directly in source code.
- The subprocess inherits the parent environment.
- Relative paths are resolved from the repository root when `python main.py` is run there.
- Later stages can run even when earlier stages are disabled, but they may produce empty outputs if required artifacts are missing.

### 4.2 Module 1: Text Extraction - `src.text_extraction.run_text_extraction(...)`

Runs when:

- `RUN_EXTRACTION=True`, as part of the image-extraction stage.
- Direct call path: `main.py` calls `src.image_extraction.run_image_extraction(...)`, which calls `src.text_extraction.run_text_extraction(...)`.

Main function:

```python
run_text_extraction(
    pdf_path: str,
    out_dir: str,
    device: str = "gpu",
    *,
    det_limit_side_len: int = DET_LIMIT_SIDE_LEN,
    det_box_thresh: float = DET_BOX_THRESH,
    rec_score_thresh: float = REC_SCORE_THRESH,
    rec_batch: int = REC_BATCH,
) -> Dict[str, Any]
```

Purpose:

This module renders PDF pages, extracts page text boxes, creates page-structured Markdown, and prepares text boxes used by image extraction.

Tools and libraries:

- PyMuPDF (`fitz`) for page rendering and native PDF text blocks
- `pymupdf4llm` for Markdown and word/line structure
- PaddlePaddle and PaddleOCR `PPStructureV3` for OCR fallback
- OpenCV and NumPy for image I/O

Inputs:

- `pdf_path`: one textbook PDF
- `out_dir`: normally `data/<pdf-stem>/extracted_images`
- `device`: `"gpu"` or `"cpu"` for PaddleOCR fallback
- OCR tuning constants such as `DET_LIMIT_SIDE_LEN`, `DET_BOX_THRESH`, `REC_SCORE_THRESH`, and `REC_BATCH` (defaults should work fine)

Step-by-step behavior:

1. Render every PDF page to `out_dir/page_images/pageNNN.jpg` at 300 DPI, but shrink pages whose longest side exceeds 5,000 pixels.
2. Attempt to extract native text blocks from the PDF with PyMuPDF.
3. Write pixel-aligned text boxes to `out_dir/text_boxes_pages/pageNNN.json`.
4. If the entire PDF produced zero native text boxes, run PaddleOCR `PPStructureV3` on the rendered page images.
5. For OCR-only PDFs, write OCR entries to `text_boxes_pages/`, add invisible OCR text to a temporary PDF, and replace the original source PDF with that OCR-enhanced version while preserving the original filename.
6. (For both native and OCR extractions) Run `pymupdf4llm.to_markdown(...)` with `page_chunks=True`, `extract_words=True`, and `table_strategy="lines_strict"` on the original PDF path, which now points to the OCR-enhanced PDF when OCR was needed.
7. Write page-separated Markdown to `pm4l.md`.
8. Write structured words and lines to `text_structure.json`.
9. Write page character offsets to `pm4l_page_index.json`.
10. Compile per-page text entries into `compiled_text_boxes.json`.

Primary outputs:

```text
extracted_images/page_images/pageNNN.jpg
extracted_images/text_boxes_pages/pageNNN.json
extracted_images/pm4l.md
extracted_images/text_structure.json
extracted_images/pm4l_page_index.json
extracted_images/compiled_text_boxes.json
```

Cache behavior:

If `pm4l.md`, `text_structure.json`, and at least one file in `text_boxes_pages/` already exist, the module is skipped and returns cached results. 

Validation:

- Confirm `page_images/` has one image per PDF page.
- Open `pm4l.md` and confirm pages are separated by `## Page NNN`.
- Inspect a few `text_boxes_pages/pageNNN.json` files.
- For scanned PDFs, confirm OCR ran and text is present in Markdown.

Notes:

- OCR is all-or-nothing. It only runs if the entire native-text pass produces zero boxes. A mixed PDF with some native text and some scanned pages may not OCR the scanned pages.
- When OCR runs, the input PDF is intentionally replaced in place with an OCR-enhanced copy that has the same filename and visible appearance plus an invisible text layer. Keep an original backup outside the run folder if the unmodified PDF must be preserved.

### 4.3 Module 2: Image Extraction - `src.image_extraction.run_image_extraction(...)`

Runs when:

- `RUN_EXTRACTION=True`.

Main function:

```python
run_image_extraction(
    pdf_path: str,
    out_dir: str,
    device: str = "gpu",
    workers: Optional[int] = None,
    save_mode: str = "final",
) -> Dict[str, Any]
```

Purpose:

This module finds likely figure regions on each rendered page and saves them as image crops.

Tools and libraries:

- `src.text_extraction` for page images and text boxes
- OpenCV and NumPy for whiteboxing, thresholding, connected components, resizing, and JPEG writing
- `ThreadPoolExecutor` for per-page processing

Inputs:

- textbook PDF
- extraction output directory
- worker count
- OCR device passed through to text extraction
- `save_mode`, usually `"final"` from `main.py`

Step-by-step behavior:

1. Convert `pdf_path` and `out_dir` to absolute paths.
2. If `bboxes.json`, `manifest.json`, `bbox_pages/`, and `bbox_crops/` already exist, return cached paths immediately and skip image extraction.
3. Call `run_text_extraction(...)` to ensure page images, Markdown, structure, and text boxes exist.
4. Create output folders:
   - `page_images/`
   - `page_masked/`
   - `page_mask/`
   - `page_bbox_overlay/`
   - `bbox_crops/`
   - `text_boxes_pages/`
   - `bbox_pages/`
5. For each page, load `page_images/pageNNN.jpg`.
6. Load text boxes from `text_boxes_pages/pageNNN.json`.
7. Whitebox text regions by filling text rectangles with white (this is meant to improve figure recognition by removing text noise).
8. Downscale the page for faster detection if needed.
9. Convert to grayscale and use adaptive thresholding to produce a binary mask.
10. Run connected components and filter boxes by size, area fraction, aspect ratio, and near-duplicate overlap.
11. Scale boxes back to full-resolution page coordinates.
12. Write per-page boxes to `bbox_pages/pageNNN.json`.
13. Save figure crops to `bbox_crops/` with names like `0001_002.jpg`.
14. If `save_mode="all"`, also save whiteboxed pages, masks, and bbox overlays (for validation and debug purposes).
15. Write combined boxes to `bboxes.json` and page timing/crop metadata to `manifest.json`.

Primary outputs:

```text
extracted_images/bbox_crops/0001_001.jpg
extracted_images/bbox_pages/page001.json
extracted_images/bboxes.json
extracted_images/manifest.json
```

Optional inspection outputs when `save_mode="all"`:

```text
extracted_images/page_masked/
extracted_images/page_mask/
extracted_images/page_bbox_overlay/
```

Cache behavior:

If `bboxes.json`, `manifest.json`, `bbox_pages/`, and `bbox_crops/` already exist, the module skips immediately and returns the cached artifact paths. If those combined outputs are missing, individual pages can still be skipped when `bbox_pages/pageNNN.json` and the rendered page image already exist; skipped pages reconstruct manifest text and crop metadata from existing artifacts and are marked with `cached_page: True`.

Validation:

- Inspect `bbox_crops/` for obvious false positives and missed figures.
- Run once with `save_mode="all"` on a small PDF to inspect masks and overlays.
- Confirm crop filenames encode the page number correctly.
- Check `manifest.json` for pages with zero crops or suspiciously high crop counts.

Notes:

- `compiled_text_boxes.json` is generated by text extraction for readability and auditing, but image extraction uses `text_boxes_pages/pageNNN.json` as the source of text boxes for whiteboxing.
- `manifest.json` is also for readability and lists all crop locations.

### 4.4 Module 3: CLIP Filtering - `src.filter_with_clip.filter_with_clip(...)`

Runs when:

- `RUN_FILTER_PHOTO=True` for stage 2, photo vs illustration vs text.
- `RUN_FILTER_SKIN=True` for stage 3, skin vs no-skin.
- `RUN_FILTER_GENDER=True` for stage 4, image gender routing.
- `RUN_FILTER_RACE=True` for stage 5, image race routing.

Main function:

```python
filter_with_clip(
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
)
```

Purpose:

This reusable module routes images into category folders using CLIP similarity to prompt lists. This lists can be found in `main.py` and were constructed over many trials of test runs. Accuracy can likely still be improved by adding/change/removing prompts from these lists.

Tools and libraries:

- Hugging Face Transformers `CLIPModel`
- `CLIPTokenizer`
- `CLIPImageProcessor`
- PyTorch
- Pillow
- NumPy

Model:

```text
openai/clip-vit-base-patch32
```

General step-by-step behavior:

1. If `clip_similarity_scores.json` and the expected category folders already exist in the stage output folder, skip CLIP inference and reuse the existing routed outputs.
2. Create one output folder for each category.
3. Clear existing files inside category folders.
4. Create `uncertain/` if `include_uncertain=True`.
5. Load CLIP model, tokenizer, and image processor.
6. Tokenize all prompt labels and compute normalized CLIP text embeddings.
7. Collect images from `input_folder` with `.jpg`, `.jpeg`, `.png`, or `.bmp` extensions.
8. Batch images.
9. Compute normalized CLIP image embeddings.
10. Compute cosine similarity between each image and each prompt.
11. Aggregate prompt scores into per-category means and maxima.
12. Choose a category using either per-category mean or max, depending on `use_mean`.
13. If uncertainty is enabled, route to `uncertain/` when the top-1 minus top-2 margin is below `uncertainty_threshold`.
14. Copy each image to its routed category folder.
15. Write `clip_similarity_scores.json`.

Output JSON contents:

- source input and output folders
- category prompt lists
- flattened prompt labels
- per-image `routed_to`
- per-image `uncertain`
- top-1 and top-2 category scores
- top-2 margin
- per-label scores
- per-category mean and max scores
- decision scores used for routing

Cache behavior:

If the stage's `clip_similarity_scores.json` and expected category folders exist, the CLIP module returns before loading CLIP, clearing category folders, or copying images. Delete the relevant CLIP output folder to force rerouting.

General notes:

- Category folders are cleared before routing, but the optional `uncertain/` folder is created without clearing old files.
- Prompt changes are only recorded in the overwritten score JSON.
- The module assumes at least two categories because it compares top-1 and top-2 decisions.

#### Stage 2: Photo vs Illustration vs Text

Input:

```text
extracted_images/bbox_crops/
```

Output:

```text
sorted_images/photo_illus/
```

Categories:

- `photo`
- `illus`
- `text`

`main.py` uses `use_mean=True` for this stage. Downstream stages currently consume only:

```text
sorted_images/photo_illus/photo/
```

Validation:

- Manually review a sample of all three folders.
- Pay special attention to clinical illustrations, tables, and designs like book covers.

Notes:

- Accuracy has been very high here, upwards of 95%

#### Stage 3: Skin vs No-Skin

Input:

```text
sorted_images/photo_illus/photo/
```

Output:

```text
sorted_images/if_skin/
```

Categories:

- `skin`
- `no_skin`

`main.py` uses `use_mean=False`. Downstream image demographic and skin-tone stages consume:

```text
sorted_images/if_skin/skin/
```

Validation:

- Check microscopy, internal anatomy, medical equipment, and mouth/oral cavity images.

Notes:

- Accuracy is also high here, upwards of 90%

#### Stage 4: Image Gender Routing

Input:

```text
sorted_images/if_skin/skin/
```

Output:

```text
sorted_images/gender/
```

Categories:

- `male`
- `female`
- `uncertain`

`main.py` uses `use_mean=False` and `include_uncertain=True`.

Validation:

- Treat labels as putative and uncertain.
- Review `uncertain/` images and low-margin decisions.
- Flag images where gender cannot reasonably be inferred from the crop.

Notes:

- Accuracy is surprisingly high here, around 75% for identifiable images.

#### Stage 5: Image Race Routing

Input:

```text
sorted_images/if_skin/skin/
```

Output:

```text
sorted_images/race/
```

Categories:

- `black`
- `white`
- `asian`
- `latine`
- `uncertain`

`main.py` uses `use_mean=False` and `include_uncertain=True`.

Validation:

- Review `uncertain/` images and low-margin decisions.

Notes:

- Accuracy is pretty abysmal here, less than 40% for identifiable images.

### 4.5 Module 4: Skin Tone Classification - `src.skin_classification.classify_skin(...)`

Runs when:

- `RUN_SKIN_CLASSER=True`.

Main function:

```python
classify_skin(
    input_dir: str,
    output_dir: str,
    abd_model_path: str,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    thr: float = DEFAULT_THRESH,
    batch_size: int = DEFAULT_BATCH,
    k: int = 5,
    seg_workers: Optional[int] = None,
    post_workers: Optional[int] = None,
    workers: Optional[int] = None,
    use_crf: bool = USE_CRF_DEFAULT,
    use_gpu: bool = True,
) -> None
```

Purpose:

This module segments visible skin, computes a representative skin color, calculates Individual Typology Angle (ITA), and maps ITA to a Monk Skin Tone category.

Tools and libraries:

- PyTorch for the inline UNet architecture and checkpoint inference
- ABD skin-segmentation checkpoint
- OpenCV and NumPy for image processing
- `pydensecrf` for DenseCRF mask refinement
- scikit-learn `KMeans` for color clustering
- CSV output via the standard library

Inputs:

```text
sorted_images/if_skin/skin/
models/abd-skin-segmentation/final_unet_pytorch.pth
```

Step-by-step behavior:

1. Create output subfolders: `masks/`, `masked/`, `clusters/`, `rep_color/`, and `probs/` when CRF is enabled.
2. If `skin_tones.csv` and `fallbacks.csv` already exist, skip segmentation and postprocessing.
3. Load the ABD checkpoint into the inline UNet model.
4. Resize each image to `DEFAULT_INPUT_SIZE = 128` for model inference.
5. Predict per-pixel skin probabilities.
6. Threshold probabilities at `DEFAULT_THRESH = 0.7` to create raw binary masks.
7. Save raw masks to `masks/`.
8. Save probability maps to `probs/` when CRF is enabled.
9. During postprocessing, optionally refine masks with DenseCRF.
10. If the refined mask covers less than `FALLBACK_MIN_COVERAGE_FRAC = 0.10` of the image, use the HSV fallback mask.
11. Remove small connected components.
12. Erode mask borders slightly to reduce edge contamination.
13. Save masked composites to `masked/`.
14. Extract skin pixels under the final mask.
15. Cluster skin pixels with KMeans.
16. Save cluster centers, counts, and swatches under `clusters/`.
17. Compute the weighted representative color from the three largest clusters.
18. Convert OpenCV Lab to CIE L*, a*, b*.
19. Calculate ITA.
20. Map ITA to Monk tone 1 through 10.
21. Write or update `skin_tones.csv`.
22. Write or update `fallbacks.csv`.

Primary outputs:

```text
skin_class/masks/<image>.jpg
skin_class/probs/<image-stem>.npy
skin_class/masked/<image>.jpg
skin_class/clusters/<image-stem>_centers.npy
skin_class/clusters/<image-stem>_counts.npy
skin_class/clusters/<image-stem>_clusters.png
skin_class/rep_color/<image-stem>_repcolor.png
skin_class/skin_tones.csv
skin_class/fallbacks.csv
```

Cache behavior:

If both `skin_tones.csv` and `fallbacks.csv` already exist, `classify_skin(...)` returns before loading the ABD model or processing images. Delete `skin_class/` to force a full skin-classification rerun.

`skin_tones.csv` columns:

```text
filename
skin_tint_0_100
rep_L
rep_a
rep_b
ITA
monk_tone
```

Validation:

- Inspect masks and masked composites.
- Inspect every image listed in `fallbacks.csv`.
- Check whether masks capture skin rather than page background, clothing, lesions only, or shadows.
- Check cluster swatches and representative color chips.
- Review distribution plots later generated by the summarizer.

Important caveats:

- CSV rows from previous runs are preserved and updated unless `skin_class/` is removed.
- Existing masks cause segmentation to skip images.
- Any change to checkpoint, threshold, CRF settings, fallback logic, or input images should be followed by deleting `skin_class/` and rerunning this stage.

### 4.6 Module 5: Text Analysis - `src.text_parser.analyze_text_llm(...)`

Runs when:

- `RUN_TEXT=True`.

Main function:

```python
analyze_text_llm(
    input_dir: str,
    output_dir: str,
    *,
    cost_cap_usd: float = 10.0,
    verbose: bool = True,
    save_batch_json: bool = True,
    reuse_existing_outputs: bool = True,
) -> Tuple[str, str, Dict[str, Any]]
```

Purpose:

This module counts explicit race and gender mentions in the extracted textbook text.

Tools and services:

- OpenAI Python package
- OpenAI Chat Completions API
- `python-dotenv`
- regular expressions for candidate mining
- `ThreadPoolExecutor` for concurrent race and gender tasks

Model:

```text
gpt-5-mini
```

Inputs:

```text
extracted_images/pm4l.md
OPENAI_API_KEY
```

Step-by-step behavior:

1. Load `.env` with `load_dotenv()`.
2. Find the first Markdown file under the input directory. Under the normal layout, this is `extracted_images/pm4l.md`.
3. Split Markdown into pages using `## Page NNN` headings and page anchors.
4. Split each page into sentences with punctuation-based regex splitting.
5. Save or reuse `sentences_by_page.json`.
6. Use word-boundary regexes to find race candidate sentences.
7. Use word-boundary regexes to find gender candidate sentences.
8. Save or reuse `race_candidates.json` and `gender_candidates.json`.
9. Run race and gender classification concurrently.
10. Batch up to 10 candidate sentences per API call.
11. Use Chat Completions JSON mode.
12. Ask the model to return compact JSON with labels and notes.
13. Aggregate labels and notes by page.
14. Save raw batch responses when enabled.
15. Write `race_results.json` and `gender_results.json`.

Primary outputs:

```text
text_analysis/sentences_by_page.json
text_analysis/race_candidates.json
text_analysis/gender_candidates.json
text_analysis/race_api_batches/batch_0001.json
text_analysis/gender_api_batches/batch_0001.json
text_analysis/race_results.json
text_analysis/gender_results.json
```

Race labels used by the text parser:

```text
asian
black
white
latino
```

Gender labels used by the text parser:

```text
male
female
```

Validation:

- Review candidate JSON files to check missed or overly broad keyword matches.
- Review raw batch JSONs for malformed or empty model outputs.
- Manually compare a sample of per-page results with `pm4l.md`.
- Record model name, run date, prompts, token usage, and actual billed cost.

Important caveats:

- Existing `race_results.json` and `gender_results.json` are reused by default.
- Candidate JSONs are reused if present.
- Cost estimates depend on hard-coded prices and should be verified.
- Race and gender tasks share one budget, but concurrent in-flight requests can overshoot a cap.
- Persistent rate limits may lead to repeated resubmission.

### 4.7 Module 6: Summarization - `src.summarize_results.summarize_results(...)`

Runs when:

- `RUN_SUMMARIZE=True`.

Main function:

```python
summarize_results(input_dir: str, output_dir: str) -> None
```

Purpose:

This module creates final research-facing datasets and summary plots for one textbook folder.

Tools and libraries:

- pandas
- NumPy
- Matplotlib
- JSON and CSV files from upstream stages

Inputs:

```text
data/<pdf-stem>/extracted_images/page_images/
data/<pdf-stem>/skin_class/skin_tones.csv
data/<pdf-stem>/sorted_images/race/clip_similarity_scores.json
data/<pdf-stem>/sorted_images/gender/clip_similarity_scores.json
data/<pdf-stem>/text_analysis/race_results.json
data/<pdf-stem>/text_analysis/gender_results.json
```

Step-by-step behavior:

1. If `image_dataset.csv`, `text_dataset.csv`, and `summary.txt` already exist, skip summarization.
2. Parse book metadata from the input folder basename.
3. Count total page images in `extracted_images/page_images/`.
4. Build the image dataset from `skin_class/skin_tones.csv`.
5. Parse page number from crop filenames such as `0001_002.jpg`.
6. Add total pages, book name, edition, and year.
7. Add skin tone estimate and Monk tone.
8. Add race and gender routes from CLIP score JSONs by filename.
9. Build the text dataset from `race_results.json` and `gender_results.json`.
10. Create one text row per page.
11. Add race mention counts for White, Black, Asian, and Latine.
12. Add gender mention counts for Male and Female.
13. Write `image_dataset.csv`.
14. Write `text_dataset.csv`.
15. If skin-tone data exists, write `monk_hist.jpg`.
16. If Lab data exists, write `lab_scatter.jpg`.
17. Write `summary.txt`.

Primary outputs:

```text
final_datasets/image_dataset.csv
final_datasets/text_dataset.csv
final_datasets/monk_hist.jpg
final_datasets/lab_scatter.jpg
final_datasets/summary.txt
```

Cache behavior:

If `image_dataset.csv`, `text_dataset.csv`, and `summary.txt` already exist, summarization returns before rebuilding datasets or plots. Delete `final_datasets/` to force regeneration.

Image dataset columns:

```text
Photo_id
Page number
Total pages in the book
Book name
Edition number
Year of release of edition
Skin tone estimate
Monk skin tone
Race
Gender
```

Text dataset columns:

```text
Page number
Total pages in the book-edition
Book name
Edition number
Year of release of edition
White
Black
Asian
Latine
Male
Female
```

Validation:

- Confirm image row count equals the number of rows in `skin_tones.csv`.
- Confirm page numbers in `Photo_id` match the original textbook page images.
- Confirm book metadata parsed correctly from the folder name.
- Compare image race/gender fields with CLIP score JSONs.
- Compare text totals with `race_results.json` and `gender_results.json`.
- Open generated plots and check for plausible values.

Important caveats:

- Missing upstream files often result in empty datasets or empty fields rather than a hard error.
- Image dataset rows are created from `skin_tones.csv`; images without skin-tone rows do not appear.
- The image dataset reads some skin-tone values by column position, so schema changes to `skin_tones.csv` can break or misalign fields.
- The text dataset uses `Latine` as the output column but reads the `latino` key from `text_parser.py`.

## 5. Safe Reruns and Cache Invalidation

The pipeline is partially resumable, but it does not fingerprint source PDFs, code, prompts, model versions, thresholds, or package versions. When in doubt, remove stale downstream artifacts before rerunning.

Text extraction cache:

- Reuses outputs when `pm4l.md`, `text_structure.json`, and `text_boxes_pages/` exist.
- Delete `extracted_images/` after changing the PDF, OCR settings, rendering settings, or text extraction code.

Image extraction cache:

- Reuses the whole module when `bboxes.json`, `manifest.json`, `bbox_pages/`, and `bbox_crops/` exist.
- If the combined outputs are missing, individual pages can still be skipped when `bbox_pages/pageNNN.json` and the page image already exist. Skipped pages reconstruct manifest metadata from existing text boxes and deterministic crop filenames.
- Delete `bbox_pages/`, `bbox_crops/`, `bboxes.json`, and `manifest.json` after changing detection settings or source page/text artifacts.

CLIP cache:

- Reuses the whole CLIP stage when `clip_similarity_scores.json` and the expected category folders exist in that stage's output folder.
- If the JSON is missing, category folders are cleared and rebuilt.
- Delete the relevant CLIP output directory and downstream stages after changing prompts, thresholds, input crops, or model versions.

Skin classification cache:

- Reuses the whole module when both `skin_tones.csv` and `fallbacks.csv` exist.
- If the CSVs are missing, existing masks and postprocessing outputs can still cause individual files to be skipped.
- Delete `skin_class/` after changing the skin model, source images, segmentation threshold, CRF settings, fallback logic, or tone calculations.

Text-analysis cache:

- Final results are reused when `race_results.json` and `gender_results.json` exist.
- Candidate files are reused if present.
- Delete `text_analysis/` after changing Markdown, regex terms, prompts, model name, or parser code.

Summary cache:

- Reuses the whole module when `image_dataset.csv`, `text_dataset.csv`, and `summary.txt` exist.
- Rerun summarization after any upstream change.

## 6. Known Issues and Risks

1. Most package versions are not pinned.
2. There is no command-line or config-file interface; stage flags are edited in `main.py`.
3. The default run is incomplete for full demographic analysis.
4. Cache validity is based mainly on file existence.
5. OCR only runs when the whole PDF has zero native text boxes.
6. OCR replaces scanned input PDFs in place with OCR-enhanced copies, so preserve raw originals separately when needed.
7. CLIP `uncertain/` folders may retain stale files.
8. Summarization may succeed with empty or incomplete upstream artifacts.
9. `summarize_results.py` reads some skin-tone CSV fields by column position.
10. OpenAI pricing is hard-coded and may become outdated.
11. OpenAI cost caps can be exceeded by concurrent requests.
12. Persistent rate limits may cause repeated text-analysis resubmissions.
13. There is no automated test suite.
14. Ethical and validity risks are substantial for image-based demographic labels.

## 7. Troubleshooting

No PDFs found:

- Confirm the folder is named `textbook_inputs`.
- Confirm PDFs are directly inside it.
- Confirm filenames end in `.pdf`.
- Run from the repository root.

PaddleOCR errors:

- Confirm PaddlePaddle matches the machine's CUDA setup or use CPU mode.
- Confirm Paddle dependencies were installed before PyTorch for GPU OCR.
- Test a scanned PDF separately.

PyTorch or CUDA errors:

- Confirm `torch.cuda.is_available()`.
- Check NVIDIA driver and CUDA compatibility.
- Reduce batch sizes or set `USE_GPU=False` for a CPU validation run.

CLIP cannot download:

- Confirm internet access and Hugging Face availability.
- Pre-download or preserve the model cache for restricted environments.
- Record the resolved model revision.

ABD model missing:

- Confirm `models/abd-skin-segmentation/final_unet_pytorch.pth` exists.
- Confirm the checkpoint loads with PyTorch.
- Record checksum and source.

Skin masks are empty or implausible:

- Inspect `skin_class/masks/`, `skin_class/masked/`, and `fallbacks.csv`.
- Confirm the input image is actually a skin photograph.
- Verify the ABD checkpoint and threshold.

OpenAI text analysis fails:

- Confirm `OPENAI_API_KEY` is available to the subprocess.
- Confirm `openai` and `python-dotenv` are installed.
- Reduce concurrency with `OPENAI_GLOBAL_MAX_WORKERS`.
- Remove stale `text_analysis/` outputs before retrying after prompt or Markdown changes.

Final datasets are empty:

- Confirm all required upstream stages were enabled.
- Confirm `skin_class/skin_tones.csv` exists for image rows.
- Confirm CLIP race/gender JSONs exist if race/gender fields are expected.
- Confirm `text_analysis/race_results.json` and `gender_results.json` exist for text counts.
- Remember that the current default flags do not produce a complete run.

## 8. Reproducibility Checklist

Before accepting a run as reproducible, record or verify:

- [ ] Repository commit hash or a copy of the exact source files.
- [ ] Any uncommitted source modifications.
- [ ] Python executable and Python version.
- [ ] Full package export, such as `pip freeze`.
- [ ] Operating system.
- [ ] CPU, GPU, CUDA, and driver details.
- [ ] PaddlePaddle, PaddleOCR, PaddleX, PyTorch, and Transformers versions.
- [ ] ABD checkpoint source, file size, and checksum.
- [ ] Hugging Face CLIP model revision or cached model snapshot.
- [ ] OpenAI model name, prompts, run date, token usage, and billed cost.
- [ ] All `main.py` flags and worker/GPU settings.
- [ ] Input PDF provenance and checksums.
- [ ] Page-image count equals PDF page count.
- [ ] Text boxes and Markdown manually inspected.
- [ ] Figure crops manually inspected.
- [ ] CLIP folders and low-margin outputs manually inspected.
- [ ] Skin masks, fallback cases, clusters, and representative colors manually inspected.
- [ ] LLM candidate sentences and final labels manually inspected.
- [ ] Final CSV row counts checked against upstream artifacts.
- [ ] Intermediate JSONs and manual review notes archived.
- [ ] Exclusions, reruns, corrections, and unresolved failures documented.

## 9. Handoff Checklist for the Next RA

Provide:

- [ ] This report.
- [ ] The research question and approved interpretation of each generated label.
- [ ] The exact textbook PDF list and provenance.
- [ ] The latest code state or commit hash.
- [ ] Environment setup notes and dependency export.
- [ ] ABD model file or verified download instructions.
- [ ] API setup instructions without exposing secret keys.
- [ ] The current `main.py` configuration.
- [ ] A stage-completion record for each textbook.
- [ ] Paths to archived outputs.
- [ ] Manual review procedures and completed review samples.
- [ ] Known bad pages, bad crops, failed textbooks, and model failure patterns.
- [ ] Manual corrections made after model inference.
- [ ] Actual API costs.
- [ ] Recommended next development priorities.

## 10. Recommended Next Development Priorities

1. Add a complete pinned dependency file.
2. Replace source-edited flags with a config file or CLI.
3. Write a run manifest containing input checksums, settings, model revisions, prompts, dependency versions, and source commit.
4. Add cache fingerprints and dependency-aware invalidation.
5. Add stage-level completeness checks before summarization.
6. Add automated tests for page parsing, filename parsing, artifact schemas, ITA/Monk mapping, and final dataset joins.
7. Preserve uncertainty scores and manual-review status in final datasets.
8. Build a structured manual-review workflow for image and text labels.
9. Reevaluate whether image-based race and gender routing should be used for the research question.

Update this report whenever module interfaces, artifact schemas, prompts, models, dependencies, or research procedures change.

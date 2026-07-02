# file: summarize_results.py
# Build final Image and Text datasets for a single book folder + summary plots + summary stats.
#
# Public API:
#   summarize_results(input_dir: str, output_dir: str) -> None
#
# Outputs in output_dir:
#   - image_dataset.csv
#   - text_dataset.csv
#   - monk_hist.jpg
#   - lab_scatter.jpg
#   - summary.txt
#
# Assumed input layout under input_dir:
#   extracted_images/page_images/                          # to count total pages
#   skin_class/skin_tones.csv                              # per-image tone metrics
#   sorted_images/race/clip_similarity_scores.json       # race labels (routed_to) by filename
#   sorted_images/gender/clip_similarity_scores.json         # gender labels (routed_to) by filename
#   text_analysis/race_results.json                        # per_page race counts
#   text_analysis/gender_results.json                      # per_page gender counts
#
# Notes:
#   - If image_dataset.csv, text_dataset.csv, and summary.txt exist, summarization is skipped.
#   - CSVs are read as strings to avoid dtype issues.
#   - Page number parsed from Photo_id like "0001_002.jpg" -> 1.
#   - Book name fields parsed from input_dir basename "name_edition_year".
#   - All outputs are saved only in output_dir.

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


# ---------- helpers ----------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_book_fields(input_dir: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse "name_edition_year" from the input_dir basename.
    Returns (book_name, edition_number, year).
    """
    base = os.path.basename(os.path.normpath(input_dir))
    parts = base.split("_")

    if len(parts) < 3:
        return base, None, None

    book_name = parts[0]

    ed_match = re.search(r"\d+", parts[1])
    edition_number = int(ed_match.group(0)) if ed_match else None

    yr_match = re.search(r"\d{4}", parts[2])
    year = int(yr_match.group(0)) if yr_match else None

    return book_name, edition_number, year


def _count_total_pages(input_dir: str) -> int:
    """
    Count files in extracted_images/page_images with image extensions.
    """
    page_dir = os.path.join(input_dir, "extracted_images", "page_images")
    if not os.path.isdir(page_dir):
        return 0

    return sum(
        1 for f in os.listdir(page_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS and os.path.isfile(os.path.join(page_dir, f))
    )


def _parse_page_from_photo_id(photo_id: str) -> Optional[int]:
    """
    From a filename like '0001_002.jpg' -> 1.
    Heuristic: take leading digits before '_' or '-'. Fallback: leading digits.
    """
    name = os.path.basename(photo_id)
    stem, _ = os.path.splitext(name)

    m = re.match(r"^(\d+)[_-]", stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    m = re.match(r"^(\d+)", stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    return None


def _load_clip_routed_map(json_path: str) -> Dict[str, str]:
    """
    Map {basename(filename) -> routed_to} from clip_similarity_scores.json.
    """
    if not os.path.isfile(json_path):
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    out = {}
    for r in results:
        fname = os.path.basename(str(r.get("filename", "")))
        route = r.get("routed_to", "")
        if fname:
            out[fname] = route
    return out


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_label(x: str) -> str:
    """
    Normalize labels: lowercase, strip, collapse spaces to underscores.
    """
    if not isinstance(x, str):
        return ""
    return re.sub(r"\s+", "_", x.strip().lower())


# ---------- image dataset ----------

def _build_image_dataset(input_dir: str) -> pd.DataFrame:
    book_name, edition_number, year = _parse_book_fields(input_dir)
    total_pages = _count_total_pages(input_dir)

    skin_csv = os.path.join(input_dir, "skin_class", "skin_tones.csv")
    if not os.path.isfile(skin_csv):
        cols = [
            "Photo_id", "Page number", "Total pages in the book",
            "Book name", "Edition number", "Year of release of edition",
            "Skin tone estimate", "Monk skin tone", "Race", "Gender"
        ]
        return pd.DataFrame(columns=cols)

    df_skin = pd.read_csv(skin_csv, dtype=str)

    race_map = _load_clip_routed_map(
        os.path.join(input_dir, "sorted_images", "race", "clip_similarity_scores.json")
    )
    gender_map = _load_clip_routed_map(
        os.path.join(input_dir, "sorted_images", "gender", "clip_similarity_scores.json")
    )

    records: List[Dict] = []

    for i in range(len(df_skin)):
        # Expected positions (defensive):
        #   0: filename
        #   1: skin tone estimate
        #   6: monk
        photo_id = str(df_skin.iloc[i, 0]).strip()
        skin_tone_est = df_skin.iloc[i, 1] if df_skin.shape[1] > 1 else ""
        monk_tone = df_skin.iloc[i, 6] if df_skin.shape[1] > 6 else ""

        photo_base = os.path.basename(photo_id)
        page_num = _parse_page_from_photo_id(photo_base)

        race_label = race_map.get(photo_base, "")
        gender_label = gender_map.get(photo_base, "")

        records.append({
            "Photo_id": photo_base,
            "Page number": page_num if page_num is not None else "",
            "Total pages in the book": total_pages,
            "Book name": book_name,
            "Edition number": edition_number if edition_number is not None else "",
            "Year of release of edition": year if year is not None else "",
            "Skin tone estimate": skin_tone_est,
            "Monk skin tone": monk_tone,
            "Race": race_label,
            "Gender": gender_label,
        })

    cols_order = [
        "Photo_id",
        "Page number",
        "Total pages in the book",
        "Book name",
        "Edition number",
        "Year of release of edition",
        "Skin tone estimate",
        "Monk skin tone",
        "Race",
        "Gender",
    ]
    return pd.DataFrame.from_records(records, columns=cols_order)


# ---------- text dataset ----------

def _build_text_dataset(input_dir: str) -> pd.DataFrame:
    book_name, edition_number, year = _parse_book_fields(input_dir)
    total_pages = _count_total_pages(input_dir)

    race_json = _load_json(os.path.join(input_dir, "text_analysis", "race_results.json"))
    gender_json = _load_json(os.path.join(input_dir, "text_analysis", "gender_results.json"))

    race_per_page = race_json.get("per_page", {}) if isinstance(race_json, dict) else {}
    gender_per_page = gender_json.get("per_page", {}) if isinstance(gender_json, dict) else {}

    pages: set[int] = set(range(1, total_pages + 1))
    for k in list(race_per_page.keys()) + list(gender_per_page.keys()):
        try:
            pages.add(int(k))
        except Exception:
            pass

    if not pages:
        cols = [
            "Page number",
            "Total pages in the book-edition",
            "Book name",
            "Edition number",
            "Year of release of edition",
            "White",
            "Black",
            "Asian",
            "Latine",
            "Male",
            "Female",
        ]
        return pd.DataFrame(columns=cols)

    rows: List[Dict] = []

    for p in sorted(pages):
        p_str = str(p)

        race_counts = (race_per_page.get(p_str, {}) or {}).get("counts", {}) or {}
        gender_counts = (gender_per_page.get(p_str, {}) or {}).get("counts", {}) or {}

        rows.append({
            "Page number": p,
            "Total pages in the book-edition": total_pages,
            "Book name": book_name,
            "Edition number": edition_number if edition_number is not None else "",
            "Year of release of edition": year if year is not None else "",
            "White": _safe_int(race_counts.get("white", 0)),
            "Black": _safe_int(race_counts.get("black", 0)),
            "Asian": _safe_int(race_counts.get("asian", 0)),
            "Latine": _safe_int(race_counts.get("latino", 0)),  # input key 'latino'
            "Male": _safe_int(gender_counts.get("male", 0)),
            "Female": _safe_int(gender_counts.get("female", 0)),
        })

    cols_order = [
        "Page number",
        "Total pages in the book-edition",
        "Book name",
        "Edition number",
        "Year of release of edition",
        "White",
        "Black",
        "Asian",
        "Latine",
        "Male",
        "Female",
    ]
    return pd.DataFrame(rows, columns=cols_order)


# ---------- plotting utils (Monk histogram + L* vs chroma scatter) ----------

# Approx Monk palette (sRGB 0..1)
_MONK_RGB01 = {
    1: (246/255, 230/255, 216/255),
    2: (235/255, 207/255, 191/255),
    3: (224/255, 189/255, 168/255),
    4: (211/255, 166/255, 142/255),
    5: (193/255, 142/255, 114/255),
    6: (168/255, 116/255,  88/255),
    7: (143/255,  94/255,  67/255),
    8: (115/255,  74/255,  51/255),
    9: ( 90/255,  56/255,  38/255),
    10:( 63/255,  39/255,  27/255),
}

# D65 reference white
_Xn, _Yn, _Zn = 95.047, 100.000, 108.883

def _lab_to_xyz(L: float, a: float, b: float) -> Tuple[float, float, float]:
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def f_inv(t: float) -> float:
        t3 = t ** 3
        return t3 if t3 > 0.008856 else (t - 16.0/116.0) / 7.787

    xr = f_inv(fx)
    yr = f_inv(fy)
    zr = f_inv(fz)
    return xr * _Xn, yr * _Yn, zr * _Zn


def _xyz_to_srgb01(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    x = X / 100.0
    y = Y / 100.0
    z = Z / 100.0

    rl =  3.2406*x - 1.5372*y - 0.4986*z
    gl = -0.9689*x + 1.8758*y + 0.0415*z
    bl =  0.0557*x - 0.2040*y + 1.0570*z

    def g(u: float) -> float:
        return 12.92*u if u <= 0.0031308 else 1.055*(u ** (1/2.4)) - 0.055

    r = g(rl); g_ = g(gl); b_ = g(bl)
    return float(np.clip(r, 0, 1)), float(np.clip(g_, 0, 1)), float(np.clip(b_, 0, 1))


def _lab_to_srgb01(L: float, a: float, b: float) -> Tuple[float, float, float]:
    X, Y, Z = _lab_to_xyz(L, a, b)
    return _xyz_to_srgb01(X, Y, Z)


def _first_present_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _plot_monk_hist(df_skin: pd.DataFrame, out_dir: str) -> Optional[str]:
    """
    Histogram of Monk tones with bar colors reflecting Monk scale.
    Expects Monk in a column containing 'monk' or column index 6.
    """
    monk_col = _first_present_column(df_skin, ["monk", "monk skin tone", "monk_tone", "monk_skin_tone"])
    if monk_col is None and df_skin.shape[1] > 6:
        monk_col = df_skin.columns[6]

    try:
        monk_raw = pd.to_numeric(df_skin[monk_col], errors="coerce").dropna()
    except Exception:
        return None

    if monk_raw.empty:
        return None

    monk = monk_raw.clip(lower=1, upper=10)

    plt.figure(figsize=(7, 4.5))
    bins = np.arange(0.5, 10.6, 1.0)
    counts, edges, patches = plt.hist(monk, bins=bins)

    for i, patch in enumerate(patches):
        tone = i + 1
        c = _MONK_RGB01.get(tone, (0.5, 0.5, 0.5))
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    plt.xticks(range(1, 11))
    plt.xlabel("Monk skin tone")
    plt.ylabel("Image count")
    plt.title("Monk distribution")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "monk_hist.jpg")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path


def _plot_lab_scatter(df_skin: pd.DataFrame, out_dir: str) -> Optional[str]:
    """
    Scatter of L* vs chroma C*_ab, colored by representative Lab color.
    Requires columns for L, a, b. We try common variants.
    """
    Lc = _first_present_column(df_skin, ["mean_L", "L*", "L", "rep_L"])
    ac = _first_present_column(df_skin, ["mean_a", "a*", "a", "rep_a"])
    bc = _first_present_column(df_skin, ["mean_b", "b*", "b", "rep_b"])
    if not all([Lc, ac, bc]):
        return None

    L = pd.to_numeric(df_skin[Lc], errors="coerce")
    a = pd.to_numeric(df_skin[ac], errors="coerce")
    b = pd.to_numeric(df_skin[bc], errors="coerce")
    mask = L.notna() & a.notna() & b.notna()
    if not mask.any():
        return None

    Lv = L[mask].to_numpy()
    av = a[mask].to_numpy()
    bv = b[mask].to_numpy()
    chroma = np.sqrt(av**2 + bv**2)

    colors = np.array([_lab_to_srgb01(Li, ai, bi) for Li, ai, bi in zip(Lv, av, bv)], dtype=float)

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(Lv, chroma, c=colors, s=28, edgecolors="none", alpha=0.95)
    plt.xlabel("L* (lightness)")
    plt.ylabel("C*_ab (chroma)")
    plt.title("Skin tone scatter: L* vs chroma")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "lab_scatter.jpg")
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path


# ---------- summary.txt writer ----------

def _write_summary_txt(
    output_dir: str,
    book_name: str,
    edition: Optional[int],
    year: Optional[int],
    total_pages: int,
    total_images: int,
    img_race: Counter,
    img_gender: Counter,
    txt_race: Counter,
    txt_gender: Counter,
    filename: str = "summary.txt",
) -> str:
    """
    Write a human-readable summary file into output_dir.
    """
    lines: List[str] = []

    lines.append(f"name: {book_name}")
    lines.append(f"edition: {'' if edition is None else edition}")
    lines.append(f"year: {'' if year is None else year}")
    lines.append(f"total_pages: {int(total_pages)}")
    lines.append(f"total_images: {int(total_images)}")
    lines.append("")

    lines.append("[images] race_counts:")
    for k in sorted(img_race.keys()):
        lines.append(f"  {k}: {int(img_race[k])}")

    lines.append("[images] gender_counts:")
    for k in sorted(img_gender.keys()):
        lines.append(f"  {k}: {int(img_gender[k])}")

    lines.append("[text] race_counts:")
    for k in sorted(txt_race.keys()):
        lines.append(f"  {k}: {int(txt_race[k])}")

    lines.append("[text] gender_counts:")
    for k in sorted(txt_gender.keys()):
        lines.append(f"  {k}: {int(txt_gender[k])}")

    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def _sum_image_label_counts(df_img: pd.DataFrame) -> Tuple[Counter, Counter]:
    """
    Sum race and gender label counts from image dataset.
    Uses columns 'Race' and 'Gender'. Case-insensitive aggregation.
    """
    race_c = Counter()
    gender_c = Counter()

    if "Race" in df_img.columns:
        for v in df_img["Race"].fillna(""):
            lab = _norm_label(v)
            if lab:
                race_c[lab] += 1

    if "Gender" in df_img.columns:
        for v in df_img["Gender"].fillna(""):
            lab = _norm_label(v)
            if lab:
                gender_c[lab] += 1

    return race_c, gender_c


def _sum_text_label_counts(df_txt: pd.DataFrame) -> Tuple[Counter, Counter]:
    """
    Sum race and gender counts from text dataset columns.
    Expected columns: 'White','Black','Asian','Latine','Male','Female'.
    """
    race_c = Counter()
    gender_c = Counter()

    # Race
    for col, key in [("White", "white"), ("Black", "black"), ("Asian", "asian"), ("Latine", "latine")]:
        if col in df_txt.columns and not df_txt.empty:
            race_c[key] = int(pd.to_numeric(df_txt[col], errors="coerce").fillna(0).sum())

    # Gender
    for col, key in [("Male", "male"), ("Female", "female")]:
        if col in df_txt.columns and not df_txt.empty:
            gender_c[key] = int(pd.to_numeric(df_txt[col], errors="coerce").fillna(0).sum())

    return race_c, gender_c


# ---------- Main entry point ----------

def summarize_results(input_dir: str, output_dir: str) -> None:
    """
    Build Image and Text datasets for a single book and write CSVs, plots, and summary.txt.

    Args:
        input_dir: path to a single book folder named "name_edition_year"
        output_dir: folder to write outputs
    """
    _ensure_dir(output_dir)

    img_csv_path = os.path.join(output_dir, "image_dataset.csv")
    txt_csv_path = os.path.join(output_dir, "text_dataset.csv")
    summary_path = os.path.join(output_dir, "summary.txt")
    if os.path.exists(img_csv_path) and os.path.exists(txt_csv_path) and os.path.exists(summary_path):
        print(f"[SUMMARY] Using cached outputs: {img_csv_path}, {txt_csv_path}, and {summary_path}")
        return

    # Parse book metadata and core counts
    book_name, edition_number, year = _parse_book_fields(input_dir)
    total_pages = _count_total_pages(input_dir)

    # Build datasets
    df_img = _build_image_dataset(input_dir)
    df_txt = _build_text_dataset(input_dir)

    # Write CSVs
    df_img.to_csv(img_csv_path, index=False, encoding="utf-8")
    df_txt.to_csv(txt_csv_path, index=False, encoding="utf-8")

    # Plots from skin_tones.csv only; saved into output_dir
    skin_csv = os.path.join(input_dir, "skin_class", "skin_tones.csv")
    if os.path.isfile(skin_csv):
        df_skin = pd.read_csv(skin_csv, dtype=str)

        # Monk histogram
        _ = _plot_monk_hist(df_skin, output_dir)

        # L* vs chroma scatter
        _ = _plot_lab_scatter(df_skin, output_dir)

    # ---------- Summary.txt ----------
    total_images = int(len(df_img)) if not df_img.empty else 0

    img_race_c, img_gender_c = _sum_image_label_counts(df_img)
    txt_race_c, txt_gender_c = _sum_text_label_counts(df_txt)

    _ = _write_summary_txt(
        output_dir=output_dir,
        book_name=book_name,
        edition=edition_number,
        year=year,
        total_pages=total_pages,
        total_images=total_images,
        img_race=img_race_c,
        img_gender=img_gender_c,
        txt_race=txt_race_c,
        txt_gender=txt_gender_c,
    )

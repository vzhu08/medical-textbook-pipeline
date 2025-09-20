import os
import json
import csv
import re
import spacy
from spacy.matcher import PhraseMatcher
from spacy.cli import download as spacy_download
from typing import Dict, Tuple, List

# Base seed terms for each race category
RACE_TERMS = {
    "white": ["white skin", "caucasian", "caucasoid", "european"],
    "black": ["black skin", "african", "negroid", "afro-caribbean"],
    "asian": ["asian", "mongoloid", "chinese"],
    "others": ["hispanic", "latino", "latina", "latinx",
               "native american", "american indian", "indigenous",
               "pacific islander",
               "middle eastern", "arab", "north african", "mena",
               "multiracial", "mixed race", "racially pigmented"]
}

def get_spacy_matcher():
    """
    Load spaCy model and build PhraseMatcher on lemmas of seed terms.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    if "lemmatizer" not in nlp.pipe_names:
        nlp.add_pipe("lemmatizer", after="attribute_ruler")
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    for cat, terms in RACE_TERMS.items():
        patterns = [nlp(term) for term in terms]
        matcher.add(cat, patterns)
    return nlp, matcher


def rect_distance(r1: Tuple[float, float, float, float],
                  r2: Tuple[float, float, float, float]) -> float:
    """
    Shortest distance between rectangles r1 and r2 (x0,y0,x1,y1).
    Overlap yields 0.
    """
    x0, y0, x1, y1 = r1
    a0, b0, a1, b1 = r2
    dx = max(a0 - x1, x0 - a1, 0)
    dy = max(b0 - y1, y0 - b1, 0)
    return (dx*dx + dy*dy)**0.5


def analyze_text(input_folder: str, output_folder: str, dpi: int = 300):
    """
    Analyze text around image crops using precomputed text_boxes.json and manifest.json.
    input_folder: root containing 'extracted_images' and 'sorted_images/if_skin/skin'
    output_folder: directory to write CSV and text outputs
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load spaCy and matcher
    nlp, matcher = get_spacy_matcher()

    # Paths
    manifest_path = os.path.join(input_folder, 'extracted_images', 'manifest.json')
    text_boxes_path = os.path.join(input_folder, 'extracted_images', 'text_boxes.json')
    images_dir = os.path.join(input_folder, 'sorted_images', 'if_skin', 'skin')

    # Load JSON data
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(text_boxes_path, 'r', encoding='utf-8') as f:
        text_boxes = json.load(f)

    relevant_files = set(os.listdir(images_dir))

    # PART 1: image_text_analysis.csv
    img_csv = os.path.join(output_folder, 'image_text_analysis.csv')
    with open(img_csv, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["page", "file", "rect", "nearest_text", "race_categories", "matched_terms"])

        for page_key, entry in manifest.get('pages', {}).items():
            page_num = int(page_key.replace('page', ''))
            for crop in entry.get('crops', []):
                fname = crop.get('file')
                if fname not in relevant_files:
                    continue
                x0, y0, w, h = crop['rect']
                crop_rect = (x0, y0, x0+w, y0+h)

                # find nearest text box and reconstruct full line
                boxes = []
                for rec in text_boxes.get(page_key, []):
                    bx, by, bw, bh = rec['rect']
                    boxes.append(((bx, by, bx+bw, by+bh), rec['text'], rec['rect']))
                nearest = min(boxes, key=lambda bt: rect_distance(crop_rect, bt[0]), default=((0,0,0,0), "", None))
                default_text = nearest[1].replace('\n', ' ').strip()
                default_rect = nearest[2] or (0, 0, 0, 0)

                # group boxes on same line by vertical overlap
                line_entries = []
                ny, nh = default_rect[1], default_rect[3]
                for rec in text_boxes.get(page_key, []):
                    ry, rh = rec['rect'][1], rec['rect'][3]
                    if abs(ry - ny) <= max(nh, rh) / 2:
                        line_entries.append(rec)
                line_entries = sorted(line_entries, key=lambda r: r['rect'][0])
                full_line = ' '.join(r['text'] for r in line_entries).replace('\n', ' ').strip()
                snippet = full_line if full_line else default_text

                # spaCy matching
                doc_snip = nlp(snippet)
                matches = matcher(doc_snip)
                cats = sorted({nlp.vocab.strings[mid] for mid,_,_ in matches})
                terms = [doc_snip[s:e].text for _,s,e in matches]
                if not cats:
                    cats = ['none']
                    terms = []

                writer.writerow([
                    page_num, fname, json.dumps(crop['rect']), snippet,
                    ";".join(cats), ";".join(terms)
                ])

    # PART 2: general_text_analysis.csv
    all_lines: List[str] = []
    for key in sorted(text_boxes.keys()):
        for rec in text_boxes[key]:
            text = rec.get('text', '')
            if text:
                all_lines.append(text.replace('\n', ' '))
    full_text = "\n".join(all_lines)
    ft_lower = full_text.lower()

    general_counts: Dict[str, int] = {}
    for cat, terms in RACE_TERMS.items():
        total = 0
        for term in terms:
            pat = re.compile(r'\b' + re.escape(term.lower()) + r'\w*')
            total += len(pat.findall(ft_lower))
        general_counts[cat] = total

    gen_csv = os.path.join(output_folder, 'general_text_analysis.csv')
    with open(gen_csv, 'w', newline='', encoding='utf-8') as gf:
        writer = csv.writer(gf)
        writer.writerow(["category", "count"])
        for cat, cnt in general_counts.items():
            writer.writerow([cat, cnt])

    # Save full_text
    txt_path = os.path.join(output_folder, 'full_text.txt')
    with open(txt_path, 'w', encoding='utf-8') as tf:
        tf.write(full_text)

    print(f"Wrote image_text_analysis.csv to {img_csv}")
    print(f"Wrote general_text_analysis.csv to {gen_csv}")
    print(f"Wrote full_text.txt to {txt_path}")

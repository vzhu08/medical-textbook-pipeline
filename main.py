# main.py

import os
import glob
import torch

from src.extract_images import extract_images_pipeline
from src.filter_with_clip import filter_with_clip
from src.skin_classification import classify_skin
from src.text_parser import analyze_text

# Pipeline control flags
RUN_EXTRACTION = False          # Extract images from PDF
RUN_FILTER_PHOTO = False        # Filter images into photos vs illustrations
RUN_FILTER_SKIN = False         # Filter photos into skin vs no_skin
RUN_SKIN_CLASSER = True         # Identify skin tone
RUN_RACE_IDENTIFIER = False     # Filter race using CLIP
RUN_TEXT = False                # Run text analysis

abd_model_path = "models/abd-skin-segmentation/final_unet_pytorch.pth"


def process_pdf(pdf_path: str):
    # Derive base name (without extension) for this textbook
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Create a separate data directory for each textbook
    base_dir = os.path.join("data", base_name)
    extracted_dir = os.path.join(base_dir, "extracted_images")
    sorted_dir = os.path.join(base_dir, "sorted_images")
    photo_sort_dir = os.path.join(sorted_dir, "photo_illus")
    skin_sort_dir = os.path.join(sorted_dir, "if_skin")
    race_sort_dir = os.path.join(sorted_dir, "race")
    skin_class_dir = os.path.join(base_dir, "skin_class")
    text_dir = os.path.join(base_dir, "text_analysis")

    # Ensure directories exist
    for d in (
            base_dir,
            extracted_dir,
            sorted_dir,
            photo_sort_dir,
            skin_sort_dir,
            race_sort_dir,
            skin_class_dir,
            text_dir
    ):
        os.makedirs(d, exist_ok=True)

    # Step 1: Extract all images from the PDF
    if RUN_EXTRACTION:
        extract_images_pipeline(
            pdf_path=pdf_path,
            output_dir=extracted_dir,
        )

    # Step 2: First-level CLIP filtering (photos vs illustrations)
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
            "a computer generated illustration"
        ]

        text_labels = [
            "a portion of text",
            "text on a page",
            "a blank page"
        ]

        filter_with_clip(
            input_folder=os.path.join(extracted_dir, "bbox_crops"),
            output_folder=photo_sort_dir,
            use_mean=True,
            categories={
                "photo": photo_labels,
                "illus": illus_labels,
                "text": text_labels
            },
            max_workers=10
        )

    # Step 3: Second-level CLIP filtering (skin vs no_skin)
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
            "internal photograph of an organ"
        ]
        filter_with_clip(
            input_folder=os.path.join(photo_sort_dir, "photo"),
            output_folder=skin_sort_dir,
            use_mean=False,
            categories={
                "skin": skin_labels,
                "no_skin": noskin_labels,
            },
            max_workers=10
        )

    if RUN_SKIN_CLASSER:
        classify_skin(input_dir=os.path.join(skin_sort_dir, "skin"),
                      output_dir=skin_class_dir,
                      abd_model_path=abd_model_path,
                      workers=10)

    if RUN_RACE_IDENTIFIER:
        white_labels = [
            "a caucasian patient",
            "a white person's skin",
            "caucasian skin",
            "white skin",
            "fair skin"
        ]
        black_labels = [
            "a black patient",
            "a black person's skin",
            "black skin",
            "african skin",
            "dark skin"
        ]
        asian_labels = [
            "an asian patient",
            "a asian person's skin",
            "asian skin",
        ]
        latino_labels = [
            "a latino patient",
            "a latino person's skin",
            "latino skin",
        ]

        filter_with_clip(
            input_folder=os.path.join(skin_sort_dir, "skin"),
            output_folder=race_sort_dir,
            use_mean=True,
            categories={
                "white": white_labels,
                "black": black_labels,
                "asian": asian_labels,
                "latino": latino_labels
            },
            max_workers=10
        )
    if RUN_TEXT:
        analyze_text(
            input_folder=base_dir,
            output_folder=text_dir,
        )

    print(f"Finished processing '{base_name}'")


def main():
    # Find all PDF files in the input folder
    input_pattern = os.path.join("textbook_inputs", "*.pdf")
    all_pdfs = glob.glob(input_pattern)

    if not all_pdfs:
        print("No PDF files found in 'textbook_inputs' directory.")
        return

    for pdf_path in all_pdfs:
        process_pdf(pdf_path)

    print("Pipeline complete for all textbooks.")


if __name__ == "__main__":
    main()

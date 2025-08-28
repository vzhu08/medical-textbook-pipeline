from paddleocr import PaddleOCR

ocr = PaddleOCR(
        device="gpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

result = ocr.predict("textbook_inputs/dermatology1.pdf")

result[0].save_to_json("data/debug")
result[0].save_to_img("data/debug")
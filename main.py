from signature_classifier import load_model
from utils import pdf_to_images, extract_signature_regions
import os

# Config
PDF_PATH = "test_docs/sample.pdf"
MODEL_PATH = "model/best_signature_model.pth"
CLASS_NAMES = ['John_Doe', 'Alice_Smith', 'David_Lane']  # Replace with your real classes

# Load model
predict_signature = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)

# Convert PDF to images
pages = pdf_to_images(PDF_PATH)

# Loop over all pages
for idx, page in enumerate(pages):
    print(f"\n📄 Processing page {idx+1}/{len(pages)}...")
    regions = extract_signature_regions(page)

    for i, region in enumerate(regions):
        name = predict_signature(region)
        print(f"🖋️ Signature #{i+1}: {name}")
        # Optional: save the signature image
        region.save(f"outputs/page{idx+1}_sig{i+1}_{name}.png")

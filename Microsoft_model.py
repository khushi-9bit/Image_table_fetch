from pdf2image import convert_from_path
from transformers import AutoProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch
import os

# === Step 1: PDF to images ===
pdf_path = r"C:\Users\KhushiOjha\Downloads\OpTransactionHistoryTpr14-05-2025-2.pdf"
output_dir = "output_tables"
os.makedirs(output_dir, exist_ok=True)

pages = convert_from_path(pdf_path, dpi=300)

# === Step 2: Load model ===
processor = AutoProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
model.eval()

# === Step 3: Process each page ===
for page_idx, page in enumerate(pages, start=1):
    page_img_path = os.path.join(output_dir, f"page_{page_idx}.png")
    page.save(page_img_path, "PNG")
    print(f"\n📄 Processing Page {page_idx}...")

    image = Image.open(page_img_path).convert("RGB")

    # Process with transformer
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get boxes
    results = processor.post_process_object_detection(
        outputs, target_sizes=[image.size[::-1]], threshold=0.9
    )

    boxes = results[0]["boxes"]

    print(f"🟦 Found {len(boxes)} table(s) on page {page_idx}")

    for i, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = map(int, box.tolist())
        cropped = image.crop((x0, y0, x1, y1))
        crop_path = os.path.join(output_dir, f"page_{page_idx}_table_{i}.png")
        cropped.save(crop_path)
        print(f"   ✅ Saved table {i} to {crop_path}")

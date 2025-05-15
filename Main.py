import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import json
from PIL import Image

PDF_PATH = r"Bank_statement2.pdf"

OUTPUT_JSON = "icici_table_output.json"

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    return thresh

def extract_table_data_from_image(image):
    preprocessed = preprocess_image(image)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(preprocessed, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(preprocessed, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine lines
    table_mask = cv2.add(detect_vertical, detect_horizontal)

    # Find contours for table cells
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 40 and cv2.boundingRect(c)[3] > 20]
    boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Sort by Y, then X

    table_data = []
    row = []
    i = 0
    last_y = boxes[0][1] if boxes else 0
    for x, y, w, h in boxes:
        if abs(y - last_y) > 20:  # New row
            if row:
                table_data.append(row)
            row = []
            last_y = y
            i += 1
        cropped = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cropped, config='--psm 6').strip()
        row.append(text)

    if row:
        table_data.append(row)
    return table_data, i

def main():
    print("[INFO] Converting PDF to images...")
    images = convert_from_path(PDF_PATH, dpi=300)
    
    final_tables = []

    for idx, pil_img in enumerate(images):
        print(f"[INFO] Processing page {idx+1}")
        img = np.array(pil_img)
        table, count = extract_table_data_from_image(img)
        if table:
            final_tables.append({
                "page": idx + 1,
                "table_data": table
            })

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_tables, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Extracted table data saved to {OUTPUT_JSON}")
    print("Number of rows in table:",count)

    #image = images[0]  # Get the first image (page)
    #image.save("output_page_1.png", "PNG")
if __name__ == "__main__":
    main()

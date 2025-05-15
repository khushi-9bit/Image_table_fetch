# Finding coordinates of the file
# import os
# import json
# from PIL import Image
# import pytesseract
# from pdf2image import convert_from_path

# # Optional: Set path to tesseract if it's not in your system PATH
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def ocr_image_to_json(image: Image.Image, page_number: int = 1):
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     results = []
#     for i in range(len(data['text'])):
#         word = data['text'][i].strip()
#         if word != "":
#             results.append({
#                 'word': word,
#                 'x': data['left'][i],
#                 'y': data['top'][i],
#                 'width': data['width'][i],
#                 'height': data['height'][i],
#                 'page': page_number
#             })
#     return results

# def process_file(input_path: str, output_json_path: str):
#     all_results = []

#     if input_path.lower().endswith('.pdf'):
#         pages = convert_from_path(input_path, dpi=300)
#         for i, page in enumerate(pages):
#             print(f"Processing page {i+1}...")
#             page_results = ocr_image_to_json(page, page_number=i+1)
#             all_results.extend(page_results)
#     else:
#         image = Image.open(input_path)
#         all_results = ocr_image_to_json(image)

#     with open(output_json_path, 'w', encoding='utf-8') as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=2)
#     print(f"Results saved to: {output_json_path}")

# # === Usage ===
# input_file = 'Discover-Prime.pdf'  # or 'your_file.png', 'your_file.jpg', etc.
# output_json = 'ocr_results.json'

# process_file(input_file, output_json)


#Finding text, color, horizontal line, vertical line

import os
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from sklearn.cluster import KMeans
from collections import Counter

# Optional: Set path to tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image_to_json(image: Image.Image, page_number: int = 1):
    # Convert to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # --- OCR Text Detection ---
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word != "":
            results.append({
                'type': 'text',
                'word': word,
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'page': page_number
            })

    # --- Line Detection ---
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100,
                            minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            orientation = 'horizontal' if abs(y1 - y2) < 10 else 'vertical' if abs(x1 - x2) < 10 else 'diagonal'
            results.append({
                'type': 'line',
                'orientation': orientation,
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'page': page_number
            })

    # --- Background Color Detection ---
    img_np = np.array(image)
    reshaped = img_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(reshaped)
    counts = Counter(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]].astype(int)

    results.append({
        'type': 'background_color',
        'color_rgb': dominant_color.tolist(),
        'page': page_number
    })

    return results

def process_file(input_path: str, output_json_path: str):
    all_results = []

    if input_path.lower().endswith('.pdf'):
        pages = convert_from_path(input_path, dpi=300)
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}...")
            page_results = ocr_image_to_json(page, page_number=i+1)
            all_results.extend(page_results)
    else:
        image = Image.open(input_path).convert("RGB")
        all_results = ocr_image_to_json(image)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_json_path}")

# === Usage ===
input_file = 'Discover-Prime.pdf'  # or 'your_file.png', 'your_file.jpg', etc.
output_json = 'ocr_results.json'

process_file(input_file, output_json)

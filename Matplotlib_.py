# SIMPLE WINDOW

import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import pytesseract
import numpy as np
from PIL import Image

# === Step 1: Convert PDF to Image ===
dpi = 300
pdf_path = 'Discover-Prime.pdf'
pages = convert_from_path(pdf_path, dpi=dpi)
page_image = pages[0]  # First page
image_np = np.array(page_image)

# === Step 2: Run OCR ===
ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)

# === Step 3: Plot with Accurate Coordinates ===
fig, ax = plt.subplots(figsize=(12, 12))
height = image_np.shape[0]  # image height in pixels

# imshow extent to fix coordinate alignment
ax.imshow(image_np, extent=[0, image_np.shape[1], height, 0])

# Draw OCR bounding boxes
for i in range(len(ocr_data['text'])):
    word = ocr_data['text'][i].strip()
    if word != "":
        x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                      ocr_data['width'][i], ocr_data['height'][i])
        rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y - 2, word, fontsize=6, color='blue')

# Display coordinates on hover
def format_coord(x, y):
    return f'x={int(x)}, y={int(y)}'
ax.format_coord = format_coord

plt.title('PDF Page with OCR Coordinates (DPI 300)')
plt.xlabel('X Pixels')
plt.ylabel('Y Pixels')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# OCR COORDINATES


# import matplotlib.pyplot as plt
# from pdf2image import convert_from_path
# import pytesseract
# import numpy as np
# from PIL import Image

# # === Step 1: Convert PDF to Image ===
# dpi = 300
# pdf_path = 'Discover-Prime.pdf'
# pages = convert_from_path(pdf_path, dpi=dpi)
# page_image = pages[1]  # First page
# image_np = np.array(page_image)

# # === Step 2: Run OCR ===
# ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)

# # === Step 3: Plot with Accurate Coordinates ===
# fig, ax = plt.subplots(figsize=(12, 12))
# height = image_np.shape[0]  # image height in pixels

# # imshow extent to fix coordinate alignment
# ax.imshow(image_np, extent=[0, image_np.shape[1], height, 0])

# # Draw OCR bounding boxes
# for i in range(len(ocr_data['text'])):
#     word = ocr_data['text'][i].strip()
#     if word != "":
#         x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
#                       ocr_data['width'][i], ocr_data['height'][i])
#         rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=1)
#         ax.add_patch(rect)
#         ax.text(x, y - 2, word, fontsize=6, color='blue')

# # Display coordinates on hover
# def format_coord(x, y):
#     return f'x={int(x)}, y={int(y)}'
# ax.format_coord = format_coord

# plt.title('PDF Page with OCR Coordinates (DPI 300)')
# plt.xlabel('X Pixels')
# plt.ylabel('Y Pixels')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.show()

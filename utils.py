import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
from PIL import Image

def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=300)

def extract_signature_regions(image_pil):
    image = np.array(image_pil)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    boxes = pytesseract.image_to_boxes(gray)
    mask = np.ones_like(gray) * 255

    for b in boxes.splitlines():
        b = b.split()
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(mask, (x1, gray.shape[0] - y2), (x2, gray.shape[0] - y1), 0, -1)

    diff = cv2.bitwise_and(gray, mask)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signature_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 100 < w < 800 and 30 < h < 300 and y > gray.shape[0] * 0.5:
            crop = gray[y:y+h, x:x+w]
            signature_regions.append(Image.fromarray(crop))

    return signature_regions

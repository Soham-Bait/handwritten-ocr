import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary image
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    # Horizontal kernel to detect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_images = []

    # Sort top → bottom
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small noise
        if w > 50 and h > 20:
            crop = image[y:y+h, x:x+w]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            line_images.append(crop)

    return line_images
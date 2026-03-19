import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ocr_service import extract_text


if __name__ == "__main__":
    image_path = r"C:\Users\soham\Downloads\10.jpg"  # <-- your image

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        exit()

    text = extract_text(image_path)

    print("\n===== OCR OUTPUT =====")
    print(text)
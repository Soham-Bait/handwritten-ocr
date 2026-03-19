from PIL import Image
import torch

from app.trocr_model import TrOCRModel
from app.config import DEVICE
from app.utils import load_image, detect_lines


def extract_text(image_path):
    image = load_image(image_path)

    # Detect lines
    line_images = detect_lines(image)

    if not line_images:
        return "No text detected"

    trocr = TrOCRModel.get_instance()
    final_text = []

    for line in line_images:
        pil_image = Image.fromarray(line).convert("RGB")

        pixel_values = trocr.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(DEVICE)

        with torch.no_grad():
            generated_ids = trocr.model.generate(
                pixel_values,
                max_new_tokens=100
            )

        text = trocr.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_text.append(text.strip())

    return "\n".join(final_text)
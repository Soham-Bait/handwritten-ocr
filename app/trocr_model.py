from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from app.config import MODEL_NAME, DEVICE


class TrOCRModel:
    _instance = None

    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TrOCRModel()
        return cls._instance
MODEL_NAME = "microsoft/trocr-base-handwritten"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
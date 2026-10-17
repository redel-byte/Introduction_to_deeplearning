import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_SEED = 22

CLASS_NAMES = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)

CLASS_TO_INDEX = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  
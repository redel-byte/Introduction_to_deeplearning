from pathlib import Path

# config.py is inside src/introduction_to_deeplearning, two folders below the project.
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_SEED = 42

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
IMAGE_CHANNELS = 1  # MRI images are read as grayscale in this project.

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0003

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

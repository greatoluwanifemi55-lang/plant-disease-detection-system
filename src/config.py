from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"
TEST_DIR = BASE_DIR / "dataset" / "test"

# Models
MODEL_DIR = BASE_DIR / "models"

# Results
RESULTS_DIR = BASE_DIR / "results"

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==================================================
# TRAINING SETTINGS
# ==================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 1e-5

# ==================================================
# MODEL SETTINGS
# ==================================================

MODEL_NAME = "MobileNetV2"

NUM_CLASSES = 15

USE_PRETRAINED = True

FINE_TUNE = True

TRAINABLE_LAYERS = 30

USE_DATA_AUGMENTATION = False
"""
==========================================================
FED-XAI V2

Module:
Configuration

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

Description:
Stores all global settings used throughout the project.

==========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

UPLOADS_DIR = BASE_DIR / "uploads"

STATIC_DIR = BASE_DIR / "static"

EXPLANATIONS_DIR = STATIC_DIR / "explanations"

TEMPLATES_DIR = BASE_DIR / "templates"

# Create required folders automatically

MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)

UPLOADS_DIR.mkdir(exist_ok=True)

EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# DATASET SETTINGS
# ==========================================================

IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224

IMAGE_CHANNELS = 3

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)

BATCH_SIZE = 32

NUM_CLASSES = 15

# ==========================================================
# MODEL SETTINGS
# ==========================================================

MODEL_NAME = "ResNet50"

USE_PRETRAINED = True

FINE_TUNE = True

TRAINABLE_LAYERS = 30

USE_DATA_AUGMENTATION = True

# ==========================================================
# TRAINING SETTINGS
# ==========================================================

EPOCHS = 15

LEARNING_RATE = 1e-4

EARLY_STOPPING_PATIENCE = 5

REDUCE_LR_PATIENCE = 2

REDUCE_LR_FACTOR = 0.2

MIN_LEARNING_RATE = 1e-7

# ==========================================================
# FEDERATED LEARNING SETTINGS
# ==========================================================

NUMBER_OF_CLIENTS = 3

COMMUNICATION_ROUNDS = 10

LOCAL_EPOCHS = 1

FEDERATED_BATCH_SIZE = 32

# ==========================================================
# EXPLAINABLE AI (LIME)
# ==========================================================

LIME_NUM_SAMPLES = 1000

LIME_NUM_FEATURES = 10

LIME_TOP_LABELS = 1

# ==========================================================
# APPLICATION SETTINGS
# ==========================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

MAX_FILE_SIZE = 10 * 1024 * 1024

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_SEED = 42
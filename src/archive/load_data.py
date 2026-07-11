import tensorflow as tf
from pathlib import Path

# Dataset paths
TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/validation")

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_dataset.class_names

print("=" * 50)
print("Classes Found:")
print("=" * 50)

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("=" * 50)
print(f"Total Classes: {len(class_names)}")
print("=" * 50)
import tensorflow as tf
from pathlib import Path

# ==================================================
# DATASET PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"

# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==================================================
# LOAD TRAINING DATA
# ==================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==================================================
# LOAD VALIDATION DATA
# ==================================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
 # Save class names before preprocessing
class_names = train_dataset.class_names
# ==================================================
# NORMALIZE IMAGES
# ==================================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# ==================================================
# IMPROVE PERFORMANCE
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)

# ==================================================
# DISPLAY INFORMATION
# ==================================================

print("=" * 50)
print("Dataset Prepared Successfully")
print("=" * 50)

print(f"Image Size : {IMAGE_SIZE}")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Number of Classes : {len(class_names)}")

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index + 1}. {class_name}")
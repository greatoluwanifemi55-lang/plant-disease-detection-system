from pathlib import Path
import shutil
import random

# Dataset location
SOURCE_DIR = Path("dataset/PlantVillage")

# Output folders
TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/validation")
TEST_DIR = Path("dataset/test")

# Split ratio
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

random.seed(42)

# Create folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Loop through every class
for class_folder in SOURCE_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    images = list(class_folder.glob("*"))

    random.shuffle(images)

    train_size = int(len(images) * TRAIN_SPLIT)
    val_size = int(len(images) * VAL_SPLIT)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (folder / class_folder.name).mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.copy(img, TRAIN_DIR / class_folder.name / img.name)

    for img in val_images:
        shutil.copy(img, VAL_DIR / class_folder.name / img.name)

    for img in test_images:
        shutil.copy(img, TEST_DIR / class_folder.name / img.name)

print("Dataset successfully split!")
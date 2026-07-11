from pathlib import Path

# Path to your dataset
DATASET_PATH = Path("dataset/PlantVillage")

# Check if dataset exists
if not DATASET_PATH.exists():
    print("Dataset folder not found!")
    exit()

# Get all class folders
classes = [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]

print("=" * 50)
print(f"Number of classes: {len(classes)}")
print("=" * 50)

# Count images in each class
for folder in sorted(classes):
    image_count = len(list(folder.glob("*.*")))
    print(f"{folder.name}: {image_count} images")
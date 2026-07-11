import json


from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


from config import (
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_DIR,
    MODEL_NAME,
    RESULTS_DIR
)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = tf.keras.models.load_model(
    MODEL_DIR / f"{MODEL_NAME}.keras"
)

print("=" * 50)
print("Model Loaded Successfully")
print("=" * 50)

# ==================================================
# LOAD TEST DATASET
# ==================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names
test_image_count = len(test_dataset.file_paths)

normalization = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(
    lambda x, y: (normalization(x), y)
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)



print(f"Test Images Loaded Successfully")
print(f"Number of Classes: {len(class_names)}")

# ==================================================
# EVALUATE MODEL
# ==================================================

print("\nEvaluating model on test dataset...\n")

loss, accuracy = model.evaluate(test_dataset)

print("=" * 50)
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")
print("=" * 50)

# ==================================================
# SAVE EVALUATION RESULTS
# ==================================================

metrics_dir = Path("..") / "results" / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

results_file = metrics_dir / "evaluation_results.txt"

with open(results_file, "w") as file:
    file.write("=" * 50 + "\n")
    file.write("MODEL EVALUATION\n")
    file.write("=" * 50 + "\n\n")
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Test Loss: {loss:.4f}\n")
    file.write(f"Number of Classes: {len(class_names)}\n")
    file.write(f"Number of Test Images: {test_image_count}\n")
print(f"\nEvaluation results saved to:\n{results_file}")
# ==================================================
# GENERATE PREDICTIONS
# ==================================================

true_labels = []

predicted_labels = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predictions = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(labels.numpy())

    predicted_labels.extend(predictions)

print("=" * 50)
print(f"True Labels      : {len(true_labels)}")
print(f"Predicted Labels : {len(predicted_labels)}")
print("=" * 50)

# ==================================================
# GENERATE CLASSIFICATION REPORT
# ==================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(report)

# ==================================================
# SAVE CLASSIFICATION REPORT
# ==================================================

reports_dir = Path("..") / "results" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

report_file = reports_dir / "classification_report.txt"

with open(report_file, "w") as file:
    file.write(report)

print("\nClassification report saved to:")
print(report_file)

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

fig, ax = plt.subplots(figsize=(16, 16))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    values_format=None,      # Hide numbers inside cells
    colorbar=False
)

plt.title(
    "Confusion Matrix",
    fontsize=18,
    pad=20
)

plt.tight_layout()

# ==================================================
# SAVE CONFUSION MATRIX
# ==================================================

figures_dir = Path("..") / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

confusion_path = figures_dir / "confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion Matrix saved to:")
print(confusion_path)

# ==================================================
# LOAD TRAINING HISTORY
# ==================================================

history_file = RESULTS_DIR / "metrics" / "training_history.json"

with open(history_file, "r") as file:
    history = json.load(file)

print("\nTraining history loaded successfully.")

# ==================================================
# PLOT ACCURACY CURVE
# ==================================================

plt.figure(figsize=(10,6))

plt.plot(
    history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)
# Save Accuracy Curve

accuracy_curve = figures_dir / "accuracy_curve.png"

plt.savefig(
    accuracy_curve,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAccuracy curve saved to:")
print(accuracy_curve)
# ==================================================
# PLOT LOSS CURVE
# ==================================================

plt.figure(figsize=(10,6))

plt.plot(
    history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)
# Save Loss Curve

loss_curve = figures_dir / "loss_curve.png"

plt.savefig(
    loss_curve,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nLoss curve saved to:")
print(loss_curve)
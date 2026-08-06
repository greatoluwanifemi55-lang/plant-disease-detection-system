from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = Path("models") / "best_global_model.keras"

DATASET_PATH = Path("dataset") / "validation"

RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Validation Dataset...")
print("=" * 60)

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = dataset.class_names

normalization = tf.keras.layers.Rescaling(1 / 255)

dataset = dataset.map(
    lambda x, y: (normalization(x), y)
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Global Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================================
# PREDICTIONS
# ==========================================================

print("\nRunning Predictions...")

y_true = []
y_pred = []

for images, labels in dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
        ],
    }
)

metrics.to_csv(
    RESULTS_DIR / "evaluation_metrics.csv",
    index=False,
)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
) as file:

    file.write(report)

print("\nClassification Report Saved!")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

cm_df.to_csv(
    RESULTS_DIR / "confusion_matrix.csv",
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.close()

print("Confusion Matrix Saved!")

# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("✔ evaluation_results/evaluation_metrics.csv")
print("✔ evaluation_results/classification_report.txt")
print("✔ evaluation_results/confusion_matrix.csv")
print("✔ evaluation_results/confusion_matrix.png")
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(
    r"C:\Users\Administrator\Desktop\FINAL YEAR PROJECT\fInal year system"
)

MODEL_PATH = BASE_DIR / "federated_learning" / "global_model.keras"

TEST_DIR = BASE_DIR / "dataset" / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 70)
print("FEDERATED RESNET50 MODEL EVALUATION")
print("=" * 70)

print("\nLoading global federated model...")
print(f"Model: {MODEL_PATH}")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Global ResNet50 model loaded successfully.")


# =====================================================
# LOAD TEST DATASET
# =====================================================

print("\nLoading test dataset...")
print(f"Test directory: {TEST_DIR}")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = test_dataset.class_names

print(f"\nNumber of classes: {len(class_names)}")
print("Classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# =====================================================
# GET TRUE LABELS
# =====================================================

print("\nCollecting true labels...")

y_true = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)


# =====================================================
# MODEL PREDICTIONS
# =====================================================

print("\nRunning predictions on test dataset...")

predictions = model.predict(
    test_dataset,
    verbose=1,
)

y_pred = np.argmax(
    predictions,
    axis=1,
)


# =====================================================
# CALCULATE METRICS
# =====================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


# =====================================================
# DISPLAY OVERALL RESULTS
# =====================================================

print("\n")
print("=" * 70)
print("FINAL FEDERATED MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy :  {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall   :  {recall * 100:.2f}%")
print(f"F1-score :  {f1 * 100:.2f}%")

print("=" * 70)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nCLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print(report)


# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nCONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred,
)

print(cm)


# =====================================================
# SAVE RESULTS
# =====================================================

RESULTS_DIR = BASE_DIR / "federated_learning" / "experiment_results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

report_path = RESULTS_DIR / "final_test_evaluation.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "FINAL FEDERATED RESNET50 MODEL EVALUATION\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Accuracy : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall   : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1-score : {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write("=" * 60 + "\n")

    file.write(report)

    file.write("\n\nCONFUSION MATRIX\n")

    file.write("=" * 60 + "\n")

    file.write(
        np.array2string(cm)
    )


print("\nEvaluation results saved to:")
print(report_path)

print("\nEvaluation completed successfully.")
"""
==========================================================
FED-XAI V2

Module:
Evaluator

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

Description:
Evaluates the trained model using the test dataset.

==========================================================
"""

import json
import tensorflow as tf

from src.config import (
    MODEL_NAME,
    MODELS_DIR,
    RESULTS_DIR,
)

from src.data_loader import load_datasets


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

def load_trained_model():

    model_path = MODELS_DIR / f"{MODEL_NAME}.keras"

    model = tf.keras.models.load_model(model_path)

    return model


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate():

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
        num_classes,
    ) = load_datasets()

    model = load_trained_model()

    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    results = {

        "model": MODEL_NAME,

        "test_accuracy": float(accuracy),

        "test_loss": float(loss)

    }

    results_path = RESULTS_DIR / f"{MODEL_NAME}_evaluation.json"

    with open(results_path, "w") as file:

        json.dump(results, file, indent=4)

    print("\n" + "=" * 60)
    print("Evaluation Completed")
    print("=" * 60)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Loss     : {loss:.4f}")

    print(f"\nResults Saved : {results_path}")

    return results


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluate()
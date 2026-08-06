"""
==========================================================
FED-XAI V2

Module:
Utilities

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

Description:
Generates training performance plots.

==========================================================
"""

import json

import matplotlib.pyplot as plt

from src.config import (
    MODEL_NAME,
    RESULTS_DIR,
)

# ==========================================================
# LOAD HISTORY
# ==========================================================

def load_history():

    history_file = RESULTS_DIR / f"{MODEL_NAME}_history.json"

    with open(history_file, "r") as file:

        history = json.load(file)

    return history


# ==========================================================
# ACCURACY CURVE
# ==========================================================

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

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

    plt.title(f"{MODEL_NAME} Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    save_path = RESULTS_DIR / f"{MODEL_NAME}_accuracy.png"

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    

    print(f"Accuracy graph saved to:\n{save_path}")


# ==========================================================
# LOSS CURVE
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(8,5))

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

    plt.title(f"{MODEL_NAME} Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    save_path = RESULTS_DIR / f"{MODEL_NAME}_loss.png"

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    

    print(f"Loss graph saved to:\n{save_path}")


# ==========================================================
# GENERATE ALL FIGURES
# ==========================================================

def generate_training_plots():

    history = load_history()

    plot_accuracy(history)

    plot_loss(history)

    print("\nTraining plots generated successfully.")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    generate_training_plots()
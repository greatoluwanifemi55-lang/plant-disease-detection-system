"""
==========================================================
FED-XAI V2

Module:
Trainer

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

Description:
Trains the deep learning model and saves the
best performing model together with the
training history.

==========================================================
"""

import json

import tensorflow as tf

from src.config import (
    MODEL_NAME,
    MODELS_DIR,
    RESULTS_DIR,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
)

from src.data_loader import load_datasets

from src.model_builder import build_model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train():

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
        num_classes,
    ) = load_datasets()

    model = build_model(num_classes)

    print("\n" + "=" * 60)
    print(f"Training {MODEL_NAME}")
    print("=" * 60)

    # ------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        filepath=MODELS_DIR / f"{MODEL_NAME}.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1,

    )

    early_stop = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=EARLY_STOPPING_PATIENCE,

        restore_best_weights=True,

        verbose=1,

    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=REDUCE_LR_FACTOR,

        patience=REDUCE_LR_PATIENCE,

        min_lr=MIN_LEARNING_RATE,

        verbose=1,

    )

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=[

            checkpoint,

            early_stop,

            reduce_lr,

        ],

    )

    # ------------------------------------------------------
    # SAVE TRAINING HISTORY
    # ------------------------------------------------------

    history_path = RESULTS_DIR / f"{MODEL_NAME}_history.json"

    with open(history_path, "w") as file:

        json.dump(history.history, file)

    print("\n" + "=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)

    print(f"Best Model Saved To : {MODELS_DIR}")
    print(f"History Saved To    : {history_path}")

    return model, history, class_names


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train()
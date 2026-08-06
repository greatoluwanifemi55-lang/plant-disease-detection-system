"""
==========================================================
FED-XAI V2

Module:
Data Loader

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

Description:
Loads and prepares the training, validation,
and testing datasets.

This module is shared by:
- Trainer
- Evaluator
- Federated Learning

==========================================================
"""

import tensorflow as tf

from src.config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
)

# ==========================================================
# AUTOTUNE
# ==========================================================

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================
# NORMALIZATION LAYER
# ==========================================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


# ==========================================================
# LOAD A SINGLE DATASET
# ==========================================================

def load_dataset(directory, shuffle):

    dataset = tf.keras.utils.image_dataset_from_directory(

        directory,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

    )

    class_names = dataset.class_names

    dataset = dataset.map(

        lambda images, labels: (

            normalization_layer(images),

            labels,

        ),

        num_parallel_calls=AUTOTUNE,

    )

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset, class_names


# ==========================================================
# LOAD ALL DATASETS
# ==========================================================

def load_datasets():

    print("\n" + "=" * 60)
    print("Loading Datasets...")
    print("=" * 60)

    train_dataset, class_names = load_dataset(

        TRAIN_DIR,

        shuffle=True,

    )

    validation_dataset, _ = load_dataset(

        VALIDATION_DIR,

        shuffle=False,

    )

    test_dataset, _ = load_dataset(

        TEST_DIR,

        shuffle=False,

    )

    num_classes = len(class_names)

    print("\nDatasets Loaded Successfully")

    print("-" * 60)

    print(f"Training Classes : {num_classes}")

    print(f"Batch Size       : {BATCH_SIZE}")

    print(f"Image Size       : {IMAGE_SIZE}")

    print("-" * 60)

    return (

        train_dataset,

        validation_dataset,

        test_dataset,

        class_names,

        num_classes,

    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    load_datasets()
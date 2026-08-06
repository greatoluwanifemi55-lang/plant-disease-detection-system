"""
==========================================================
FED-XAI V2

Module:
Model Builder

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

Description:
Builds and compiles the deep learning architecture used
throughout the project.

This module is shared by:

✓ Trainer
✓ Evaluator
✓ Predictor
✓ Federated Learning

==========================================================
"""

import tensorflow as tf

from src.config import (
    MODEL_NAME,
    IMAGE_SIZE,
    USE_PRETRAINED,
    FINE_TUNE,
    TRAINABLE_LAYERS,
    USE_DATA_AUGMENTATION,
    LEARNING_RATE,
)

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

# ==========================================================
# LOAD BACKBONE MODEL
# ==========================================================


def get_backbone():

    if MODEL_NAME == "ResNet50":

        backbone = tf.keras.applications.ResNet50(

            input_shape=IMAGE_SIZE + (3,),

            include_top=False,

            weights="imagenet" if USE_PRETRAINED else None,

        )

    elif MODEL_NAME == "MobileNetV2":

        backbone = tf.keras.applications.MobileNetV2(

            input_shape=IMAGE_SIZE + (3,),

            include_top=False,

            weights="imagenet" if USE_PRETRAINED else None,

        )

    elif MODEL_NAME == "EfficientNetB0":

        backbone = tf.keras.applications.EfficientNetB0(

            input_shape=IMAGE_SIZE + (3,),

            include_top=False,

            weights="imagenet" if USE_PRETRAINED else None,

        )

    else:

        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    # ======================================================
    # FINE-TUNING
    # ======================================================

    if FINE_TUNE:

        backbone.trainable = True

        for layer in backbone.layers[:-TRAINABLE_LAYERS]:

            layer.trainable = False

    else:

        backbone.trainable = False

    return backbone


# ==========================================================
# BUILD COMPLETE MODEL
# ==========================================================


def build_model(num_classes):

    backbone = get_backbone()

    inputs = tf.keras.Input(
        shape=IMAGE_SIZE + (3,),
        name="input_image",
    )

    x = inputs

    # ------------------------------------------------------

    if USE_DATA_AUGMENTATION:

        x = data_augmentation(x)

    # ------------------------------------------------------

    x = backbone(
        x,
        training=False,
    )

    # ------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_norm_1"
    )(x)

    x = tf.keras.layers.Dropout(
        0.5,
        name="dropout_1",
    )(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dense_256",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_norm_2"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    # ------------------------------------------------------

    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="FED_XAI_ResNet50",

    )

    # ======================================================
    # COMPILE MODEL
    # ======================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],

    )

    return model


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    model = build_model(num_classes=15)

    model.summary()
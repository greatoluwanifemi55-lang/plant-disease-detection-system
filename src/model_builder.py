import tensorflow as tf

from config import (
    MODEL_NAME,
    FINE_TUNE,
    TRAINABLE_LAYERS,
    USE_PRETRAINED,
    USE_DATA_AUGMENTATION
)


def get_base_model():

    if MODEL_NAME == "MobileNetV2":

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet" if USE_PRETRAINED else None
        )

    elif MODEL_NAME == "EfficientNetB0":

        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet" if USE_PRETRAINED else None
        )

    elif MODEL_NAME == "ResNet50":

        base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet" if USE_PRETRAINED else None
        )

    else:

        raise ValueError("Unknown Model")

    # Fine-tuning
    if FINE_TUNE:

        base_model.trainable = True

        for layer in base_model.layers[:-TRAINABLE_LAYERS]:
            layer.trainable = False

    else:

        base_model.trainable = False

    return base_model


# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])


def build_model(num_classes):

    base_model = get_base_model()

    # -----------------------------------------
    # Input Layer
    # -----------------------------------------

    inputs = tf.keras.layers.Input(
        shape=(224, 224, 3)
    )

    x = inputs

    # -----------------------------------------
    # Optional Data Augmentation
    # -----------------------------------------

    if USE_DATA_AUGMENTATION:

        x = data_augmentation(x)

    # -----------------------------------------
    # MobileNetV2 Backbone
    # -----------------------------------------

    x = base_model(x)

    # -----------------------------------------
    # Classification Head
    # -----------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    # -----------------------------------------
    # Build Functional Model
    # -----------------------------------------

    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="PlantDiseaseModel"

    )

    return model
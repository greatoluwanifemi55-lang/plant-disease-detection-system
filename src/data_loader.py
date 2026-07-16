import tensorflow as tf

from config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_dataset.class_names

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization(x), y)
    )

    validation_dataset = validation_dataset.map(
        lambda x, y: (normalization(x), y)
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, class_names
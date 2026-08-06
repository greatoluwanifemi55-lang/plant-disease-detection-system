import tensorflow as tf
import numpy as np
from pathlib import Path

IMAGE_SIZE = (224, 224)

model = tf.keras.models.load_model(
    "models/best_global_model.keras",
    compile=False
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

dataset = tf.keras.utils.image_dataset_from_directory(
    Path("dataset") / "validation",
    image_size=IMAGE_SIZE,
    batch_size=32,
    shuffle=False
)

class_names = dataset.class_names

normalization = tf.keras.layers.Rescaling(1/255)

dataset = dataset.map(
    lambda x, y: (normalization(x), y)
)

loss, accuracy = model.evaluate(dataset)

print("\nAccuracy:", accuracy)
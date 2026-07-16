from pathlib import Path

import tensorflow as tf


# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


BASE_DATASET = Path(
    r"C:\Users\Administrator\Desktop\FINAL YEAR PROJECT\fInal year system"
) / "federated_clients"


# =====================================================
# LOAD DATA
# =====================================================

def load_data(client_name: str):

    client_path = BASE_DATASET / client_name

    full_dataset = tf.keras.utils.image_dataset_from_directory(
        client_path,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    train_dataset, val_dataset = full_dataset

    train_count = tf.data.experimental.cardinality(train_dataset).numpy() * BATCH_SIZE

    class_names = train_dataset.class_names

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization(x), y)
    )

    val_dataset = val_dataset.map(
        lambda x, y: (normalization(x), y)
    )

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return train_dataset, val_dataset, class_names, train_count


# =====================================================
# BUILD MODEL
# =====================================================

def load_model(learning_rate: float = 3e-5):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    # Enable fine-tuning
    base_model.trainable = True

    # Freeze the earlier layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

    inputs = tf.keras.Input(
    shape=IMAGE_SIZE + (3,)
    )

    x = data_augmentation(inputs)

    x = base_model(
    x,
    training=False,
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(
        15,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="FederatedPlantDiseaseModel",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# =====================================================
# LOCAL TRAINING
# =====================================================

def train_model(model, dataset, epochs):

    history = model.fit(
        dataset,
        epochs=epochs,
        verbose=1,
    )

    return history


# =====================================================
# LOCAL EVALUATION
# =====================================================

def evaluate_model(model, dataset):

    loss, accuracy = model.evaluate(
        dataset,
        verbose=0,
    )

    return loss, accuracy
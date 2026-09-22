from pathlib import Path

import tensorflow as tf


# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 15

BASE_DIR = Path(
    r"C:\Users\Administrator\Desktop\FINAL YEAR PROJECT\fInal year system"
)

BASE_DATASET = BASE_DIR / "federated_clients"

MODEL_PATH = BASE_DIR / "models" / "ResNet50.keras"


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

    class_names = train_dataset.class_names

    train_batches = tf.data.experimental.cardinality(
        train_dataset
    ).numpy()

    train_count = train_batches * BATCH_SIZE

    # IMPORTANT:
    # Keep pixels in the 0-255 range.
    # This matches the current ResNet50 inference pipeline.

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_dataset,
        val_dataset,
        class_names,
        train_count,
    )


# =====================================================
# LOAD TRAINED RESNET50 MODEL
# =====================================================

def load_model(learning_rate: float = 1e-4):

    print("\nLoading trained ResNet50 model...")
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ResNet50 model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("ResNet50 model loaded successfully.")

    # -------------------------------------------------
    # Compile for federated local training
    # -------------------------------------------------

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

def train_model(
    model,
    dataset,
    epochs,
):

    history = model.fit(
        dataset,
        epochs=epochs,
        verbose=1,
    )

    return history


# =====================================================
# LOCAL EVALUATION
# =====================================================

def evaluate_model(
    model,
    dataset,
):

    loss, accuracy = model.evaluate(
        dataset,
        verbose=0,
    )

    return loss, accuracy
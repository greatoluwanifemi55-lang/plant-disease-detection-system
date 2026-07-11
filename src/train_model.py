import tensorflow as tf
from pathlib import Path

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 15
EPOCHS = 15

# ==================================================
# LOAD DATASETS
# ==================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

# ==================================================
# PREPROCESS
# ==================================================

normalization = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==================================================
# BUILD MODEL
# ==================================================
# ==================================================
# DATA AUGMENTATION
# ==================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True
# Freeze the lower layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==================================================
# SAVE THE BEST MODEL
# ==================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_DIR / "mobilenetv2_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ==================================================
# TRAIN
# ==================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("\nTraining Complete!")
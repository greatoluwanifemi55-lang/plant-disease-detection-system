import json

import tensorflow as tf

from config import (
    MODEL_DIR,
    RESULTS_DIR,
    MODEL_NAME,
    EPOCHS,
    LEARNING_RATE
)

from data_loader import load_datasets
from model_builder import build_model
train_dataset, validation_dataset, class_names = load_datasets()

model = build_model(len(class_names))
print("\n" + "=" * 50)
print("MODEL SUMMARY")
print("=" * 50)

model.summary()
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
print("\nModel compiled successfully.")
checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_DIR / f"{MODEL_NAME}_best.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

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

    min_lr=1e-7

)
history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[

        checkpoint,

        early_stop,

        reduce_lr

    ]

)

# ==================================================
# SAVE TRAINING HISTORY
# ==================================================

history_dict = history.history

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

metrics_dir = RESULTS_DIR / "metrics"

metrics_dir.mkdir(parents=True, exist_ok=True)

history_file = metrics_dir / "training_history.json"

with open(history_file, "w") as file:

    json.dump(
        history_dict,
        file,
        indent=4
    )

print("\nTraining history saved to:")
print(history_file)

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print(f"Best model saved to:")
print(MODEL_DIR / f"{MODEL_NAME}_best.keras")
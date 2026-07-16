import tensorflow as tf

from src.config import MODEL_DIR, MODEL_NAME

model = tf.keras.models.load_model(
    MODEL_DIR / f"{MODEL_NAME}_best.keras"
)

base_model = model.layers[0]

print("\nBASE MODEL NAME")
print(base_model.name)

print("\nLAST 20 LAYERS\n")

for layer in base_model.layers[-20:]:
    print(layer.name)
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/best_global_model.keras",
    compile=False
)

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)
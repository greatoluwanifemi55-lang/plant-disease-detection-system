import tensorflow as tf

MODEL_PATH = "models/ResNet50.keras"

print("\n==============================")
print("RESNET50 MODEL VERIFICATION")
print("==============================")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\nMODEL LAYERS:")
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)

print("\n==============================")
print("VERIFICATION COMPLETE")
print("==============================")
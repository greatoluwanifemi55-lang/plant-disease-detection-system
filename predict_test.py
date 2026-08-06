import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
import json

model = tf.keras.models.load_model(
    "models/best_global_model.keras",
    compile=False
)

with open("models/class_names.json") as f:
    class_names = json.load(f)

img = load_img(
    "test_images/test.jpg",
    target_size=(224,224)
)

img = img_to_array(img)

# IMPORTANT
img = img.astype("float32") / 255.0

img = np.expand_dims(img,0)

pred = model.predict(img,verbose=0)[0]

print()

for i,name in enumerate(class_names):
    print(f"{i:2d} {name:45} {pred[i]:.6f}")

print()

print("Prediction:",class_names[np.argmax(pred)])
print("Confidence:",np.max(pred))
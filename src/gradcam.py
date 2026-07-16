from tf_explain.core.grad_cam import GradCAM

import tensorflow as tf
import numpy as np

from keras.utils import load_img, img_to_array

from src.config import (
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_NAME
)

# ==================================================
# LOAD MODEL
# ==================================================

model = tf.keras.models.load_model(
    MODEL_DIR / f"{MODEL_NAME}_best.keras"
)

# ==================================================
# LAST CONVOLUTION LAYER
# ==================================================

LAST_CONV_LAYER = "Conv_1"

# ==================================================
# LOAD IMAGE
# ==================================================

def load_image(image_path):

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = img_to_array(image)

    image = image.astype("float32") / 255.0

    return np.expand_dims(image, axis=0)

# ==================================================
# GENERATE GRAD-CAM
# ==================================================

def generate_gradcam(image_path, output_path):

    image = load_image(image_path)

    explainer = GradCAM()

    grid = explainer.explain(

        validation_data=(image, None),

        model=model,

        class_index=None,

        layer_name=LAST_CONV_LAYER

    )

    tf.keras.utils.save_img(
        output_path,
        grid
    )

    return output_path

# ==================================================
# TEST GRAD-CAM
# ==================================================

if __name__ == "__main__":

    from pathlib import Path

    image_path = Path("test_images/test.jpg")

    output_path = Path("static/gradcam.jpg")

    generate_gradcam(
        image_path,
        output_path
    )

    print("=" * 50)
    print("Grad-CAM Generated Successfully!")
    print("=" * 50)
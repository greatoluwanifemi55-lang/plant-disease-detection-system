"""
==========================================================
FED-XAI V2

Explainable AI Module (LIME)

Authors:
- Okposio Great
- Adegbola Victor
==========================================================
"""

import matplotlib

# Force Matplotlib to use a non-GUI backend
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

from keras.utils import load_img, img_to_array

from keras.utils import (
    load_img,
    img_to_array,
)

from .config import (
    IMAGE_SIZE,
    MODELS_DIR,
)

# ==========================================================
# LOAD MODEL
# ==========================================================


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_path):

    image = load_img(

        image_path,

        target_size=IMAGE_SIZE,

    )

    image = img_to_array(image)

    return image

# ==========================================================
# GLOBAL MODEL
# ==========================================================

lime_model = None

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(images):

    images = np.array(images)

    return lime_model.predict(
        images,
        verbose=0
    )

# ==========================================================
# GENERATE LIME EXPLANATION
# ==========================================================

def generate_explanation(
    model,
    image_path,
    save_path,
):

    global lime_model
    lime_model = model

    image = preprocess_image(image_path)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype("double"),
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=30,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    explanation_image = mark_boundaries(
        temp / 255.0,
        mask,
    )
        # ======================================================
    # SAVE IMAGE
    # ======================================================

    save_path = Path(save_path)

    save_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    plt.figure(

        figsize=(5, 5)

    )

    plt.imshow(

        explanation_image

    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=120,

        bbox_inches="tight",

    )

    plt.close()

    del explanation
    del temp
    del mask
    del explanation_image

    print(

        f"LIME explanation saved to: {save_path}"

    )

    return save_path
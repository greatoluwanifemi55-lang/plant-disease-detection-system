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

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from lime import lime_image

from skimage.segmentation import (
    mark_boundaries,
    quickshift,
)

from keras.utils import (
    load_img,
    img_to_array,
)

from .config import IMAGE_SIZE


# ==========================================================
# GLOBAL MODEL
# ==========================================================

lime_model = None


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
# PREDICTION FUNCTION
# ==========================================================

def predict(images):

    images = np.array(images)

    return lime_model.predict(
        images,
        verbose=0,
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

    segments = quickshift(
        image,
        kernel_size=3,
        max_dist=6,
        ratio=0.5,
    )

    explanation = explainer.explain_instance(
        image.astype("double"),
        predict,
        segmentation_fn=lambda x: segments,
        top_labels=1,
        hide_color=0,
        num_samples=100,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=20,
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

    plt.figure(figsize=(5, 5))

    plt.imshow(explanation_image)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close("all")

    del explanation
    del temp
    del mask
    del explanation_image

    print(f"LIME explanation saved to: {save_path}")

    return save_path
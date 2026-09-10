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
    slic,
)

from skimage.color import rgb2hsv

from skimage.morphology import (
    closing,
    opening,
    remove_small_objects,
    disk,
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

    # Keep the same 0-255 range used by ResNet50
    image = image.astype("float32")

    return image


# ==========================================================
# CREATE FOREGROUND MASK
# ==========================================================

def create_foreground_mask(image):
    """
    Estimate the plant/leaf foreground using colour
    characteristics.

    This mask is used only to improve the LIME
    visualization. It does NOT change the classifier.
    """

    image_uint8 = image.astype("uint8")

    # Convert RGB image to HSV.
    hsv = rgb2hsv(image_uint8)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Green vegetation tends to have stronger saturation.
    green_mask = (
        (saturation > 0.12)
        &
        (value > 0.12)
    )

    # Also retain darker leaf regions.
    dark_leaf_mask = (
        (value > 0.05)
        &
        (value < 0.55)
        &
        (saturation > 0.08)
    )

    foreground = (
        green_mask
        |
        dark_leaf_mask
    )

    # Clean small isolated regions.
    foreground = opening(
        foreground,
        disk(2)
    )

    foreground = closing(
        foreground,
        disk(4)
    )

    foreground = remove_small_objects(
        foreground,
        max_size=150
    )

    return foreground


# ==========================================================
# CREATE LIME SEGMENTS
# ==========================================================

def create_segments(image):
    """
    Create SLIC superpixels and mark the estimated
    background as a separate region.
    """

    foreground = create_foreground_mask(
        image
    )

    segments = slic(
        image.astype("uint8"),
        n_segments=100,
        compactness=15,
        sigma=1,
        start_label=1,
    )

    # Background receives label 0.
    segments[~foreground] = 0

    return segments, foreground


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(images):

    images = np.array(
        images,
        dtype="float32"
    )

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

    image = preprocess_image(
        image_path
    )

    # ======================================================
    # FOREGROUND-AWARE SEGMENTATION
    # ======================================================

    segments, foreground = create_segments(
        image
    )

    print(
        "Foreground-aware SLIC segmentation created.",
        flush=True
    )

    explainer = lime_image.LimeImageExplainer()

    print(
        "Running LIME explanation...",
        flush=True
    )

    explanation = explainer.explain_instance(

        image.astype("double"),

        predict,

        top_labels=1,

        hide_color=0,

        num_samples=100,

        segmentation_fn=lambda x: segments,

    )

    # ======================================================
    # GET POSITIVE LIME FEATURES
    # ======================================================

    label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(

        label,

        positive_only=True,

        num_features=5,

        hide_rest=False,

    )

    # ======================================================
    # REMOVE BACKGROUND-ONLY FEATURES
    # ======================================================

    # Only retain highlighted pixels that overlap
    # with the estimated plant foreground.
    mask = mask * foreground.astype(
        mask.dtype
    )

    explanation_image = mark_boundaries(

        temp / 255.0,

        mask,

        color=(1, 1, 0),

        mode="thick",

    )

    # ======================================================
    # SAVE IMAGE
    # ======================================================

    save_path = Path(
        save_path
    )

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

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=120,

        bbox_inches="tight",

    )

    plt.close(
        "all"
    )

    # ======================================================
    # CLEANUP
    # ======================================================

    del explanation
    del temp
    del mask
    del explanation_image
    del segments
    del foreground

    print(
        f"LIME explanation saved to: {save_path}",
        flush=True
    )

    return save_path
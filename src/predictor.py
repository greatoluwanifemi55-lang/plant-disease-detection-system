"""
==========================================================
FED-XAI V2

Module:
Prediction Engine

Project:
Federated Explainable AI Framework for Plant Disease
Detection Using Transfer Learning

Authors:
- Okposio Great
- Adegbola Victor

==========================================================
"""

import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from keras.utils import (
    load_img,
    img_to_array,
)

from .config import (
    MODELS_DIR,
    MODEL_NAME,
    IMAGE_SIZE,
    TEST_DIR,
    EXPLANATIONS_DIR,
)

from .recommendations import recommendations
from .explainability import generate_explanation


# ==========================================================
# GLOBAL MODEL
# ==========================================================

GLOBAL_MODEL_PATH = MODELS_DIR / "ResNet50.keras"

model = None
class_names = None


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    global model

    if model is None:

        print("=" * 60)
        print("Loading Federated Model...", flush=True)

        print(
            f"Model path: {GLOBAL_MODEL_PATH}",
            flush=True
        )

        print(
            f"Model exists: {GLOBAL_MODEL_PATH.exists()}",
            flush=True
        )

        model = tf.keras.models.load_model(
            GLOBAL_MODEL_PATH,
            compile=False
        )

        print(
            "Model Loaded Successfully",
            flush=True
        )

        print(
            f"Model Architecture: {MODEL_NAME}",
            flush=True
        )

        print("=" * 60)

    return model


# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

def load_class_names():

    global class_names

    if class_names is None:

        class_file = MODELS_DIR / "class_names.json"

        with open(
            class_file,
            "r"
        ) as file:

            class_names = json.load(file)

    return class_names


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_path):

    print(
        "Loading image...",
        flush=True
    )

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    print(
        "Image loaded",
        flush=True
    )

    image = img_to_array(image)

    print(
        "Converted to array",
        flush=True
    )

    # Keep pixel values in the same 0-255 range
    # used during ResNet50 training.
    image = image.astype("float32")

    print(
    "Pixel values kept in 0-255 range",
    flush=True
    )

    image = np.expand_dims(
        image,
        axis=0,
    )

    print(
        "Expanded dimensions",
        flush=True
    )

    return image


# ==========================================================
# FORMAT DISEASE NAME
# ==========================================================

def format_disease_name(name):

    name = name.replace(
        "___",
        " "
    )

    name = name.replace(
        "__",
        " "
    )

    name = name.replace(
        "_",
        " "
    )

    words = name.split()

    formatted = []

    for word in words:

        if (
            formatted
            and word.lower()
            == formatted[0].lower()
        ):
            continue

        formatted.append(word)

    return " ".join(
        formatted
    ).title()


# ==========================================================
# PREDICT DISEASE
# ==========================================================

def predict_disease(image):

    print(
        "STEP A - Loading model...",
        flush=True
    )

    loaded_model = load_model()

    print(
        "STEP B - Model loaded",
        flush=True
    )

    print(
        "STEP C - Loading class names...",
        flush=True
    )

    classes = load_class_names()

    print(
        "STEP D - Class names loaded",
        flush=True
    )

    print(
        "STEP E - Running prediction...",
        flush=True
    )

    prediction = loaded_model.predict(
        image,
        verbose=0,
    )[0]

    print(
        "\n================ PREDICTIONS ================"
    )

    for i, score in enumerate(prediction):

        print(
            f"{i:2d} | "
            f"{classes[i]:40} | "
            f"{score:.6f}"
        )

    print(
        "============================================"
    )

    print(
        "STEP F - Prediction finished",
        flush=True
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = float(
        prediction[predicted_index]
    )

    disease = format_disease_name(
        classes[predicted_index]
    )

    top3_indices = np.argsort(
        prediction
    )[::-1][:3]

    top_predictions = []

    for index in top3_indices:

        top_predictions.append({

            "disease": format_disease_name(
                classes[index]
            ),

            "confidence": float(
                prediction[index] * 100
            ),

        })

    return (
        disease,
        confidence,
        top_predictions,
    )


# ==========================================================
# PREDICT IMAGE
# ==========================================================

def predict_image(image_path):

    print(
        "Predict image started",
        flush=True
    )

    start_time = time.time()

    # ======================================================
    # IMAGE PREPROCESSING
    # ======================================================

    image = preprocess_image(
        image_path
    )

    print(
        "Image preprocessed",
        flush=True
    )

    print("=" * 60)
    print(
        "STEP 1: Image Preprocessed"
    )
    print("=" * 60)

    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    model_start = time.time()

    disease, confidence, top_predictions = predict_disease(
        image
    )

    print("=" * 60)
    print(
        "STEP 2: Prediction Complete"
    )
    print("=" * 60)

    model_time = (
        time.time() - model_start
    )

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================

    recommendation = recommendations.get(

        disease,

        [
            "No recommendation available."
        ]

    )

    # ======================================================
    # LIME EXPLANATION
    # ======================================================

    EXPLANATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    explanation_filename = (
        f"{Path(image_path).stem}_lime.png"
    )

    explanation_path = (
        EXPLANATIONS_DIR /
        explanation_filename
    )

    print("=" * 60)
    print(
        "STEP 3: Starting LIME"
    )
    print("=" * 60)

    print(
        "Starting LIME",
        flush=True
    )

    generate_explanation(
        load_model(),
        image_path,
        explanation_path
    )

    print(
        "LIME complete",
        flush=True
    )

    print("=" * 60)
    print(
        "STEP 4: LIME Finished"
    )
    print("=" * 60)

    total_time = (
        time.time() - start_time
    )

    # ======================================================
    # BACKEND LOG
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FED-XAI V2"
    )

    print(
        "=" * 70
    )

    print(
        f"Image               : "
        f"{Path(image_path).name}"
    )

    print(
        f"Model               : "
        f"{MODEL_NAME}"
    )

    print(
        f"Disease             : "
        f"{disease}"
    )

    print(
        f"Confidence          : "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\nTop 3 Predictions"
    )

    print(
        "-" * 70
    )

    for i, prediction in enumerate(

        top_predictions,

        start=1

    ):

        print(

            f"{i}. "
            f"{prediction['disease']:<45}"
            f"{prediction['confidence']:.2f}%"

        )

    print(
        "\nRecommendations"
    )

    print(
        "-" * 70
    )

    for item in recommendation:

        print(
            f"• {item}"
        )

    print(
        "\nExplainability"
    )

    print(
        "-" * 70
    )

    print(
        f"LIME Output : "
        f"{explanation_filename}"
    )

    print(
        "\nFederated Learning"
    )

    print(
        "-" * 70
    )

    print(
        "Aggregation : FedAvg"
    )

    print(
        "Clients     : Oyo, Kaduna, Benue"
    )

    print(
        "Rounds      : 10"
    )

    print(
        "\nPerformance"
    )

    print(
        "-" * 70
    )

    print(
        f"Inference Time : "
        f"{model_time:.2f}s"
    )

    print(
        f"Total Time     : "
        f"{total_time:.2f}s"
    )

    print(
        "=" * 70
    )

    # ======================================================
    # RETURN TO FLASK
    # ======================================================

    return {

        "image": Path(
            image_path
        ).name,

        "disease": disease,

        "confidence": (
            confidence * 100
        ),

        "top_predictions":
            top_predictions,

        "recommendation":
            recommendation,

        "explanation":
            explanation_filename,

        "model":
            MODEL_NAME,

        "aggregation":
            "FedAvg",

        "clients":
            "Oyo, Kaduna, Benue",

        "rounds":
            10,

        "accuracy":
            "93.15%",

        "xai":
            "LIME"

    }


# ==========================================================
# TEST PREDICTOR
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "FED-XAI V2 Prediction Engine"
    )

    print("=" * 60)

    test_folder = TEST_DIR

    image_files = list(
        test_folder.rglob("*.jpg")
    )

    if len(image_files) == 0:

        image_files = list(
            test_folder.rglob("*.png")
        )

    if len(image_files) == 0:

        print(
            "No test images found."
        )

    else:

        result = predict_image(
            image_files[0]
        )

        print(
            "\nPrediction Summary"
        )

        print(
            "-" * 60
        )

        print(
            f"Disease : "
            f"{result['disease']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence']:.2f}%"
        )

        print(
            f"Model : "
            f"{result['model']}"
        )

        print(
            f"Accuracy : "
            f"{result['accuracy']}"
        )

        print(
            "=" * 60
        )
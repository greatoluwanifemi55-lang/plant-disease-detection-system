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

GLOBAL_MODEL_PATH = (
    MODELS_DIR /
    "best_global_model.keras"
)

model = None

class_names = None
# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    global model

    if model is None:

        print("=" * 60)

        print("Loading Federated Model...")

        model = tf.keras.models.load_model(
            GLOBAL_MODEL_PATH
        )

        print("Model Loaded Successfully")

        print("=" * 60)

    return model


# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

def load_class_names():

    global class_names

    if class_names is None:

        class_file = MODELS_DIR / "class_names.json"

        with open(class_file, "r") as file:

            class_names = json.load(file)

    return class_names
# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_path):

    print("Loading image...", flush=True)

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    print("Image loaded", flush=True)

    image = img_to_array(image)

    print("Converted to array", flush=True)

    image = np.expand_dims(
        image,
        axis=0,
    )

    print("Expanded dimensions", flush=True)

    return image

# ==========================================================
# FORMAT DISEASE NAME
# ==========================================================

def format_disease_name(name):

    name = name.replace("___", " ")

    name = name.replace("__", " ")

    name = name.replace("_", " ")

    words = name.split()

    formatted = []

    for word in words:

        if (

            formatted

            and word.lower() == formatted[0].lower()

        ):

            continue

        formatted.append(word)

    return " ".join(formatted).title()
# ==========================================================
# PREDICT DISEASE
# ==========================================================

def predict_disease(image):

    model = load_model()

    classes = load_class_names()

    prediction = model.predict(

        image,

        verbose=0,

    )[0]

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

        top_predictions.append(

            {

                "disease": format_disease_name(

                    classes[index]

                ),

                "confidence": float(

                    prediction[index] * 100

                ),

            }

        )

    return (

        disease,

        confidence,

        top_predictions,

    )
# ==========================================================
# PREDICT IMAGE
# ==========================================================

def predict_image(image_path):


    print("Predict image started")

    start_time = time.time()

    start_time = time.time()

    image = preprocess_image(
    image_path
)
    print("Image preprocessed")
    
    print("=" * 60)
    print("STEP 1: Image Preprocessed")
    print("=" * 60)

    model_start = time.time()

    disease, confidence, top_predictions = predict_disease(

        image

    )
    print("=" * 60)
    print("STEP 2: Prediction Complete")
    print("=" * 60)

    model_time = time.time() - model_start

    recommendation = recommendations.get(

        disease,

        ["No recommendation available."]

    )

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
    print("STEP 3: Starting LIME")
    print("=" * 60)

    print("Starting LIME")

    #generate_explanation(
    #image_path,
    #explanation_path
  #)

    #print("LIME complete")

    #print("=" * 60)
    #print("STEP 4: LIME Finished")
    #print("=" * 60)

    total_time = time.time() - start_time
        # ======================================================
    # BACKEND LOG
    # ======================================================

    print("\n" + "=" * 70)

    print("FED-XAI V2")

    print("=" * 70)

    print(f"Image               : {Path(image_path).name}")

    print(f"Disease             : {disease}")

    print(f"Confidence          : {confidence * 100:.2f}%")

    print("\nTop 3 Predictions")

    print("-" * 70)

    for i, prediction in enumerate(

        top_predictions,

        start=1

    ):

        print(

            f"{i}. "

            f"{prediction['disease']:<45}"

            f"{prediction['confidence']:.2f}%"

        )

    print("\nRecommendations")

    print("-" * 70)

    for item in recommendation:

        print(f"• {item}")

    print("\nExplainability")

    print("-" * 70)

    print(

        f"LIME Output : {explanation_filename}"

    )

    print("\nPerformance")

    print("-" * 70)

    print(

        f"Inference Time : {model_time:.2f}s"

    )

    print(

        f"Total Time     : {total_time:.2f}s"

    )

    print("=" * 70)
        # ======================================================
    # RETURN TO FLASK
    # ======================================================

    return {

        "image": Path(image_path).name,

        "disease": disease,

        "confidence": confidence * 100,

        "top_predictions": top_predictions,

        "recommendation": recommendation,

        "explanation": explanation_filename,

        "model": MODEL_NAME,

        "aggregation": "FedAvg",

        "clients": "Oyo, Kaduna, Benue",

        "rounds": 10,

        "accuracy": "95.09%",

        "xai": "LIME"

    }


# ==========================================================
# TEST PREDICTOR
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print("FED-XAI V2 Prediction Engine")

    print("=" * 60)

    test_folder = TEST_DIR

    image_files = list(test_folder.rglob("*.jpg"))

    if len(image_files) == 0:

        image_files = list(test_folder.rglob("*.png"))

    if len(image_files) == 0:

        print("No test images found.")

    else:

        result = predict_image(

            image_files[0]

        )

        print("\nPrediction Summary")

        print("-" * 60)

        print(

            f"Disease : {result['disease']}"

        )

        print(

            f"Confidence : {result['confidence']:.2f}%"

        )

        print("=" * 60)
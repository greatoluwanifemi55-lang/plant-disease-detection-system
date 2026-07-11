import tensorflow as tf
import numpy as np

from pathlib import Path
from keras.utils import load_img, img_to_array

from config import (
    MODEL_DIR,
    MODEL_NAME,
    IMAGE_SIZE,
    TEST_DIR
)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = tf.keras.models.load_model(
    MODEL_DIR / f"{MODEL_NAME}_best.keras"
)

print("=" * 50)
print("Model Loaded Successfully")
print("=" * 50)

# ==================================================
# LOAD CLASS NAMES
# ==================================================

dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False
)

class_names = dataset.class_names

print("Classes Loaded Successfully")
print("=" * 50)

# ==================================================
# IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image_path):

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = img_to_array(image)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==================================================
# SELECT TEST IMAGE
# ==================================================

def select_test_image():

    image_folder = Path("../test_images")

    image_files = sorted([
        file for file in image_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(image_files) == 0:
        raise FileNotFoundError(
            "No images found inside test_images folder."
        )

    print("\n" + "=" * 50)
    print("AVAILABLE TEST IMAGES")
    print("=" * 50)

    for index, file in enumerate(image_files, start=1):
        print(f"{index}. {file.name}")

    print("=" * 50)

    choice = int(input("Select image number: "))

    return image_files[choice - 1]

# ==================================================
# PREDICT DISEASE
# ==================================================

def predict_disease(image):

    prediction = model.predict(
        image,
        verbose=0
    )

    print(prediction)

    return prediction

# ==================================================
# PREDICT DISEASE
# ==================================================

def predict_disease(image):

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    disease = format_disease_name(
    class_names[predicted_index]
)

    top3_indices = np.argsort(prediction)[::-1][:3]

    return (
        disease,
        confidence,
        prediction,
        top3_indices
    )


# ==================================================
# PREDICT IMAGE
# ==================================================

def predict_image(image_path):

    # Preprocess the image
    image = preprocess_image(image_path)

    # Get prediction from the model
    disease, confidence, prediction, top3_indices = predict_disease(image)

    # Store the Top 3 predictions
    top_predictions = []

    for index in top3_indices:

        top_predictions.append(

            {
                "disease": format_disease_name(
                    class_names[index]
                ),

                "confidence": prediction[index] * 100
            }

        )

    # Return all prediction information
    return {

        "image": image_path.name,

        "disease": disease,

        "confidence": confidence * 100,

        "top_predictions": top_predictions

    }

# ==================================================
# FORMAT DISEASE NAME
# ==================================================

def format_disease_name(name):

    # Replace dataset separators
    name = name.replace("___", " ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")

    words = name.split()

    cleaned_words = []

    for word in words:

        if not cleaned_words:
            cleaned_words.append(word)
            continue

        if (
            word.lower() == cleaned_words[0].lower()
            and len(cleaned_words) == 1
        ):
            continue

        cleaned_words.append(word)

    return " ".join(cleaned_words).title()

# ==================================================
# MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    image_path = select_test_image()

    result = predict_image(image_path)

    print("\n" + "=" * 50)
    print("PLANT DISEASE PREDICTION")
    print("=" * 50)

    print(f"Image : {result['image']}")

    print("\nBest Prediction")
    print(f"Disease   : {result['disease']}")
    print(f"Confidence: {result['confidence']:.2f}%")

    print("\nTop 3 Predictions")
    print("-" * 50)

    for i, prediction in enumerate(result["top_predictions"], start=1):

     print(
        f"{i}. "
        f"{prediction['disease']:40}"
        f"{prediction['confidence']:.2f}%"
    )

    print("=" * 50)
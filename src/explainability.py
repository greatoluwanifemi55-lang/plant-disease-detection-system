import numpy as np
import tensorflow as tf

from lime import lime_image
from skimage.segmentation import mark_boundaries

from keras.utils import load_img, img_to_array
from PIL import Image

from src.config import MODEL_DIR, MODEL_NAME, IMAGE_SIZE


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    MODEL_DIR / f"{MODEL_NAME}_best.keras"
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict(images):

    images = np.array(images)

    images = images.astype("float32") / 255.0

    return model.predict(images)


# ==========================================
# GENERATE LIME EXPLANATION
# ==========================================

def generate_explanation(image_path, output_path):

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = img_to_array(image).astype(np.uint8)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(

    image,

    predict,

    top_labels=1,

    hide_color=0,

    num_samples=1500

)

    temp, mask = explanation.get_image_and_mask(

    explanation.top_labels[0],

    positive_only=True,

    num_features=5,

    hide_rest=False

)

    explained_image = mark_boundaries(temp / 255.0, mask)

    explained_image = (explained_image * 255).astype(np.uint8)

    Image.fromarray(explained_image).save(output_path)

    return output_path

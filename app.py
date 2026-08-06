from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)

from werkzeug.utils import secure_filename

from src.predictor import predict_image

from src.config import (
    UPLOADS_DIR,
    ALLOWED_EXTENSIONS,
)

# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)

UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# CHECK FILE TYPE
# ==========================================================

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(".", 1)[1].lower()

        in ALLOWED_EXTENSIONS

    )

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")

def home():

    return render_template(

        "index.html"

    )

from flask import send_from_directory

# ==========================================================
# SERVE UPLOADED IMAGES
# ==========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(

        UPLOADS_DIR,

        filename

    )

# ==========================================================
# PREDICT
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("STEP 1 - Request received")

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please upload an image."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="No image selected."
        )

    if not allowed_file(file.filename):

        return render_template(
            "index.html",
            error="Unsupported image format."
        )

    filename = secure_filename(
        file.filename
    )

    image_path = (
        UPLOADS_DIR /
        filename
    )

    file.save(image_path)

    print("STEP 2 - Image saved")

    print("STEP 3 - Starting prediction")

    result = predict_image(
        image_path
    )

    print("STEP 4 - Prediction finished")

    return render_template(
        "result.html",
        result=result
    )

# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )
    
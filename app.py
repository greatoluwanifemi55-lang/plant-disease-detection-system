from flask import (
    Flask,
    render_template,
    request
)

from pathlib import Path

from src.predictor import predict_image


app = Flask(__name__)

# ==========================================
# UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# PREDICT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Receive uploaded image
    image = request.files["image"]

    # Save image
    save_path = UPLOAD_FOLDER / image.filename

    image.save(save_path)

    # Show information in terminal
    print("=" * 50)
    print("IMAGE RECEIVED")
    print("=" * 50)
    print("Filename :", image.filename)
    print("Saved To :", save_path)
    print("Content Type :", image.content_type)

    # Run AI prediction
    result = predict_image(save_path)

    # Display result in browser
    return render_template(
    "result.html",
    result=result,
    image=image.filename
)


# ==========================================
# START FLASK
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
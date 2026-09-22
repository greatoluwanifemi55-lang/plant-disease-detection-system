from pathlib import Path
import json
import sqlite3
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    send_file,
    abort,
)

from werkzeug.utils import secure_filename

from src.predictor import predict_image

from src.config import (
    UPLOADS_DIR,
    ALLOWED_EXTENSIONS,
    BASE_DIR,
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
# HISTORY / REPORT STORAGE
# ==========================================================

HISTORY_DIR = BASE_DIR / "history"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_PATH = HISTORY_DIR / "diagnosis_history.db"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    """Open the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    """Create the diagnosis history table if it does not exist."""
    connection = get_db()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            top_predictions TEXT,
            recommendations TEXT,
            explanation_filename TEXT,
            model TEXT,
            aggregation TEXT,
            clients TEXT,
            rounds INTEGER,
            accuracy TEXT
        )
        """
    )

    connection.commit()
    connection.close()


init_database()

# ==========================================================
# CHECK FILE TYPE
# ==========================================================


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================================
# HOME PAGE
# ==========================================================


@app.route("/")
def home():
    return render_template("index.html")


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
# SAVE DIAGNOSIS TO HISTORY
# ==========================================================


def save_diagnosis(result):
    """Save one completed prediction to the SQLite history database."""

    connection = get_db()

    cursor = connection.execute(
        """
        INSERT INTO diagnoses (
            created_at,
            image_filename,
            disease,
            confidence,
            top_predictions,
            recommendations,
            explanation_filename,
            model,
            aggregation,
            clients,
            rounds,
            accuracy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result.get("image", ""),
            result.get("disease", "Unknown"),
            float(result.get("confidence", 0)),
            json.dumps(result.get("top_predictions", [])),
            json.dumps(result.get("recommendation", [])),
            result.get("explanation", ""),
            result.get("model", "ResNet50"),
            result.get("aggregation", "FedAvg"),
            result.get("clients", "Oyo, Kaduna, Benue"),
            int(result.get("rounds", 10)),
            result.get("accuracy", "95.04%"),
        ),
    )

    diagnosis_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return diagnosis_id


# ==========================================================
# PREDICT
# ==========================================================


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("STEP 1 - Request received", flush=True)

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

    filename = secure_filename(file.filename)

    image_path = UPLOADS_DIR / filename

    file.save(image_path)

    print("STEP 2 - Image saved", flush=True)

    try:
        print("STEP 3 - Calling predict_image()", flush=True)

        result = predict_image(
            image_path
        )

        print("STEP 4 - Prediction completed", flush=True)

        # Save completed diagnosis for future history/report access.
        diagnosis_id = save_diagnosis(result)
        result["diagnosis_id"] = diagnosis_id

        print(
            f"STEP 5 - Diagnosis saved to history: #{diagnosis_id}",
            flush=True
        )

        return render_template(
            "result.html",
            result=result
        )

    except Exception:
        import traceback

        traceback.print_exc()

        return (
            f"<pre>{traceback.format_exc()}</pre>",
            500
        )


# ==========================================================
# DIAGNOSIS HISTORY
# ==========================================================


@app.route("/history")
def history():

    connection = get_db()

    records = connection.execute(
        """
        SELECT *
        FROM diagnoses
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        records=records
    )


# ==========================================================
# DOWNLOAD PDF REPORT
# ==========================================================


@app.route("/report/<int:diagnosis_id>")
def report(diagnosis_id):

    connection = get_db()

    record = connection.execute(
        """
        SELECT *
        FROM diagnoses
        WHERE id = ?
        """,
        (diagnosis_id,)
    ).fetchone()

    connection.close()

    if record is None:
        abort(404)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image,
        )
    except ImportError:
        return (
            "<h3>PDF reporting is not installed.</h3>"
            "<p>Run: <code>pip install reportlab</code></p>",
            500
        )

    report_path = REPORTS_DIR / f"diagnosis_report_{diagnosis_id}.pdf"

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = styles["BodyText"]

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    story = []

    story.append(
        Paragraph(
            "FED-XAI V2 — Plant Disease Diagnosis Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Diagnosis ID: #{record['id']}<br/>"
            f"Date: {record['created_at']}",
            body_style
        )
    )

    story.append(Spacer(1, 10))

    # Add the uploaded image when it is available.
    image_path = UPLOADS_DIR / record["image_filename"]

    if image_path.exists():
        try:
            report_image = Image(
                str(image_path),
                width=75 * mm,
                height=75 * mm,
                kind="proportional",
            )
            story.append(report_image)
            story.append(Spacer(1, 10))
        except Exception:
            pass

    story.append(
        Paragraph("Diagnosis", heading_style)
    )

    diagnosis_table = Table(
        [
            ["Disease", record["disease"]],
            ["AI Confidence", f"{record['confidence']:.2f}%"],
            ["Model", record["model"] or "ResNet50"],
            ["Aggregation", record["aggregation"] or "FedAvg"],
            ["Clients", record["clients"] or "Oyo, Kaduna, Benue"],
            ["Communication Rounds", str(record["rounds"] or 10)],
            ["Model Test Accuracy", record["accuracy"] or "95.04%"],
        ],
        colWidths=[55 * mm, 105 * mm],
    )

    diagnosis_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(diagnosis_table)

    # Top predictions
    story.append(
        Paragraph("Top Predictions", heading_style)
    )

    top_predictions = json.loads(
        record["top_predictions"] or "[]"
    )

    prediction_rows = [
        ["Rank", "Disease", "Confidence"]
    ]

    for index, prediction in enumerate(top_predictions, start=1):
        prediction_rows.append(
            [
                str(index),
                prediction.get("disease", "Unknown"),
                f"{float(prediction.get('confidence', 0)):.2f}%",
            ]
        )

    predictions_table = Table(
        prediction_rows,
        colWidths=[20 * mm, 105 * mm, 35 * mm],
    )

    predictions_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(predictions_table)

    # Recommendations
    story.append(
        Paragraph("Recommended Actions", heading_style)
    )

    recommendations = json.loads(
        record["recommendations"] or "[]"
    )

    if recommendations:
        for recommendation in recommendations:
            story.append(
                Paragraph(
                    f"• {recommendation}",
                    body_style
                )
            )
            story.append(Spacer(1, 4))
    else:
        story.append(
            Paragraph(
                "No recommendation was recorded.",
                body_style
            )
        )

    story.append(
        Paragraph("Explainability", heading_style)
    )

    story.append(
        Paragraph(
            "The system generated a LIME explanation for this prediction. "
            "The highlighted regions represent image areas identified as "
            "important to the model's local prediction.",
            body_style
        )
    )

    # Add the LIME explanation image when available.
    explanation_filename = record["explanation_filename"]

    if explanation_filename:
        explanation_path = (
            BASE_DIR
            / "static"
            / "explanations"
            / explanation_filename
        )

        if explanation_path.exists():
            try:
                explanation_image = Image(
                    str(explanation_path),
                    width=75 * mm,
                    height=75 * mm,
                    kind="proportional",
                )
                story.append(Spacer(1, 8))
                story.append(explanation_image)
            except Exception:
                pass

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "This report was generated by the FED-XAI V2 plant disease "
            "detection system.",
            body_style
        )
    )

    document.build(story)

    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"FED-XAI_Diagnosis_{diagnosis_id}.pdf",
        mimetype="application/pdf",
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================


if __name__ == "__main__":

    app.run(
        debug=True
    )

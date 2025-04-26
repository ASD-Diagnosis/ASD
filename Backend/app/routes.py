import os
from flask import Blueprint, request, jsonify
from app.utils import predict_class
import logging

api = Blueprint("api", __name__)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "stack")

@api.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the ASD Diagnosis API"}), 200

@api.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    model_name = request.form.get("model_name", DEFAULT_MODEL)

    if not file.filename.endswith(".1D"):
        return jsonify({"error": "Only .1D files are supported"}), 400

    if not file or not model_name:
        return jsonify({"error": "File or model name missing"}), 400
    
    try:
        prediction, probabilities = predict_class(file, model_name)
        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        return jsonify({"error": "Invalid file or data format"}), 400
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
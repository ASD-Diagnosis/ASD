from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file

class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models/")
    DEFAULT_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    DEFAULT_PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
    if not DEFAULT_MODEL:
        raise ValueError("DEFAULT_MODEL is not set in the environment")
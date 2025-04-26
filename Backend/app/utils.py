import os
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import logging
from config import Config

MODEL_DIR = Config.MODEL_DIR
DEFAULT_SCALER_PATH = Config.DEFAULT_SCALER_PATH
DEFAULT_PCA_PATH = Config.DEFAULT_PCA_PATH


logger = logging.getLogger(__name__)

def compute_corr_from_1d(file):
    try:
        data = np.loadtxt(file)
        if data.shape[1] != 200:
            raise ValueError(f"File data must have 200 ROIs; got {data.shape[1]} instead.")
        corr = np.corrcoef(data.T)
        return corr[np.triu_indices(200, k=1)]
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {str(e)}")
        raise

def predict_class(file, model_name):
    # Define paths dynamically
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    scaler_path = DEFAULT_SCALER_PATH
    pca_path = DEFAULT_PCA_PATH

    # Check files exist
    required_files = [model_path, scaler_path, pca_path]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Compute features
    feat = compute_corr_from_1d(file).reshape(1, -1)

    # 1) Impute NaNs
    imp = SimpleImputer(strategy="mean")
    feat = imp.fit_transform(feat)

    if feat.shape[1] != 19900:
        raise ValueError(f"Expected 19900 features, got {feat.shape[1]}")

    # 2) Scale
    scaler = joblib.load(scaler_path)
    feat_scaled = scaler.transform(feat)

    # 3) PCA
    pca = joblib.load(pca_path)
    feat_pca = pca.transform(feat_scaled)

    # 4) Predict
    model = joblib.load(model_path)
    pred = model.predict(feat_pca)[0]
    proba = model.predict_proba(feat_pca)[0]
    label = "ASD" if pred == 1 else "TD"

    logger.info(f"Prediction complete: {label}")
    return label, {"ASD": proba[1], "TD": proba[0]}
# import numpy as np
# import joblib
# import os


# def load_1d_file_compute_corr(file_path):
#     try:
#         data = np.loadtxt(file_path)
#         if data.shape[1] != 200:
#             raise ValueError(f"❌ Expected 200 ROIs, got {data.shape[1]}")
#         corr = np.corrcoef(data.T)
#         return corr[np.triu_indices(200, k=1)]  # Flatten upper triangle
#     except Exception as e:
#         raise ValueError(f"🚫 Failed to load or process file: {e}")


# def predict_from_1d(file_path, model_path, pca_path):
#     # Load and flatten correlation matrix
#     feature = load_1d_file_compute_corr(file_path)

#     # Load PCA model and transform
#     pca = joblib.load(pca_path)
#     feature_pca = pca.transform(np.array(feature).reshape(1, -1))

#     # Load classifier and predict
#     model = joblib.load(model_path)
#     prediction = model.predict(feature_pca)[0]
#     return prediction


# def main():
#     file_path = "./data/CC200_data/Yale_0050628_rois_cc200.1D"  # 👈 CHANGE THIS TO YOUR FILE
#     model_path = "./models/xgboost.joblib"
#     pca_path = "./models/pca.joblib"  # 👈 Saved in prepare_data.py (see below)

#     if not os.path.exists(file_path):
#         print(f"❌ File not found: {file_path}")
#         return

#     pred = predict_from_1d(file_path, model_path, pca_path)
#     print(f"🧠 Predicted class: {'ASD' if pred == 1 else 'TD'}")


# if __name__ == "__main__":
#     main()



import numpy as np
import joblib
import os
import sys
from sklearn.impute import SimpleImputer


def load_1d_file_compute_corr(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] != 200:
            raise ValueError(f"❌ Expected 200 ROIs, got {data.shape[1]}")
        corr = np.corrcoef(data.T)
        return corr[np.triu_indices(200, k=1)]  # Flatten upper triangle
    except Exception as e:
        raise ValueError(f"🚫 Failed to load or process file: {e}")


def predict_from_1d(file_path, model_path, pca_path, scaler_path):
    feature = load_1d_file_compute_corr(file_path)

    # Handle NaN values (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    feature_imputed = imputer.fit_transform(feature.reshape(1, -1))

    # Load and apply scaler
    scaler = joblib.load(scaler_path)
    feature_scaled = scaler.transform(feature_imputed)

    # Load and apply PCA
    pca = joblib.load(pca_path)
    feature_pca = pca.transform(feature_scaled)

    # Load model and predict
    model = joblib.load(model_path)
    prediction = model.predict(feature_pca)[0]
    prob = model.predict_proba(feature_pca)[0]
    return prediction, prob


def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py path/to/file.1D")
        return

    file_path = sys.argv[1]
    model_path = "models/lightgbm.joblib"
    pca_path = "models/pca.joblib"
    scaler_path = "models/scaler.joblib"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    pred, prob = predict_from_1d(file_path, model_path, pca_path, scaler_path)
    label = "ASD" if pred == 1 else "TD"
    print(f"🧠 Predicted class: {label}")
    print(f"🔍 Confidence: ASD={prob[1]:.3f}, TD={prob[0]:.3f}")


if __name__ == "__main__":
    main()

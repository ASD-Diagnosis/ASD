import os, sys
import numpy as np
import joblib
from sklearn.impute import SimpleImputer



def load_1d_file_compute_corr(fp):
    data = np.loadtxt(fp)
    if data.shape[1]!=200:
        raise ValueError("Expected 200 ROIs")
    corr = np.corrcoef(data.T)
    return corr[np.triu_indices(200, k=1)]

def main():
    if len(sys.argv)!=3:
        print("Usage: python predict.py model_name path/to/file.1D")
        print("Example: python predict.py stack data/cc200_data/1234567_rois_cc200.1D")
        return

    model_name, file_path = sys.argv[1], sys.argv[2]
    model_path  = f"models/{model_name}.pkl"
    scaler_path = "models/scaler.pkl"
    pca_path    = "models/pca.pkl"

    if not os.path.exists(file_path):
        print("File not found:", file_path); return
    if not os.path.exists(model_path):
        print("Model not found:", model_path); return

    feat = load_1d_file_compute_corr(file_path).reshape(1,-1)
    # 1) impute NaNs
    imp = SimpleImputer(strategy='mean')
    feat = imp.fit_transform(feat)

    if feat.shape[1] != 19900:
        print(f"Skipping {file_path}: Expected 19900 features, got {feat.shape[1]}")
        return

    # 2) scale
    scaler = joblib.load(scaler_path)
    feat_scaled = scaler.transform(feat)
    # 3) PCA
    pca = joblib.load(pca_path)
    feat_pca = pca.transform(feat_scaled)
    # 4) predict
    model = joblib.load(model_path)
    pred = model.predict(feat_pca)[0]
    proba = model.predict_proba(feat_pca)[0]
    label = "ASD" if pred==1 else "TD"
    print(f"Predicted: {label}")
    print(f"Probability → ASD: {proba[1]:.3f}, TD: {proba[0]:.3f}")

if __name__=="__main__":
    main()

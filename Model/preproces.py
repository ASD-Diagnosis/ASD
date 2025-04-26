import os
import re
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

def load_1d_file_compute_corr(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] != 200:
            return None
        corr = np.corrcoef(data.T)
        return corr if corr.shape == (200, 200) else None
    except:
        return None

def extract_features(data_dir):
    features = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.1D'):
                m = re.search(r'(\d{7})_rois_cc200\.1D$', file)
                if not m: 
                    continue
                subj = m.group(1)
                mat = load_1d_file_compute_corr(os.path.join(root, file))
                if mat is not None and not np.isnan(mat).any():
                    vec = mat[np.triu_indices(200, k=1)]
                    features.append((subj, vec))
    return features

def load_phenotypes(csv_path):
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        sid = str(row['SUB_ID']).zfill(7)
        labels[sid] = 1 if row.get('DX_GROUP', 0)==1 else 0
    return labels

def apply_pca_and_scale(features, n_components=100):
    subj_ids = [s for s,_ in features]
    X = np.vstack([vec for _,vec in features])

    # 1) Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    # 2) PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    # Save transformers
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl", compress=3)
    joblib.dump(pca,    "models/pca.pkl",    compress=3)

    # build DataFrame
    df = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(n_components)])
    df.insert(0, "Subject_ID", subj_ids)
    return df

def main():
    cc200_dir = "./data/cc200_data"
    pheno_csv  = "./data/ABIDE_summary.csv"
    out_features_pkl = "features/cc200_features.pkl"

    os.makedirs("features", exist_ok=True)

    features = extract_features(cc200_dir)
    with open(out_features_pkl, "wb") as f:
        pickle.dump(features, f)

    labels = load_phenotypes(pheno_csv)
    pca_df = apply_pca_and_scale(features, n_components=100)

    X, y = [], []
    for _, row in pca_df.iterrows():
        sid = row["Subject_ID"]
        if sid in labels:
            X.append(row.values[1:].astype(float))
            y.append(labels[sid])

    np.save("features/X.npy", np.array(X))
    np.save("features/y.npy", np.array(y))
    print("✅ Prepared features (X.npy, y.npy) and saved scaler/pca to models/")

if __name__=="__main__":
    main()

# import os
# import re
# import joblib
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.decomposition import PCA


# def load_1d_file_compute_corr(file_path):
#     try:
#         data = np.loadtxt(file_path)
#         if data.shape[1] != 200:
#             return None
#         corr = np.corrcoef(data.T)
#         return corr if corr.shape == (200, 200) else None
#     except:
#         return None


# def extract_features(data_dir):
#     features = []
#     for root, _, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith('.1D'):
#                 match = re.search(r'(\d{7})_rois_cc200\.1D$', file)
#                 if match:
#                     subject_id = match.group(1)
#                     file_path = os.path.join(root, file)
#                     mat = load_1d_file_compute_corr(file_path)
#                     if mat is not None and not np.isnan(mat).any():
#                         features.append((subject_id, mat[np.triu_indices(200, k=1)]))
#     return features


# def save_features(features, out_path):
#     with open(out_path, "wb") as f:
#         pickle.dump(features, f)


# def load_phenotypes(csv_path):
#     df = pd.read_csv(csv_path)
#     subject_labels = {}
#     for _, row in df.iterrows():
#         try:
#             subj_id = str(row['SUB_ID']).zfill(7)
#             subject_labels[subj_id] = 1 if row['DX_GROUP'] == 1 else 0
#         except:
#             continue
#     return subject_labels


# def apply_pca(features, n_components=100):
#     subject_ids = [sid for sid, _ in features]
#     X = np.array([vec for _, vec in features])
#     pca = PCA(n_components=n_components)
#     X_reduced = pca.fit_transform(X)
#     df = pd.DataFrame(X_reduced)
#     df.insert(0, "Subject_ID", subject_ids)

#     pca_df, pca = apply_pca(features)

#     # Save PCA model
#     os.makedirs("models", exist_ok=True)
#     joblib.dump(pca, "models/pca.joblib")

#     return df, pca


# def main():
#     cc200_dir = "./data/cc200_data"
#     pheno_csv = "./data/ABIDE_summary.csv"
#     output_pkl = "features/cc200_features.pkl"

#     os.makedirs("features", exist_ok=True)

#     features = extract_features(cc200_dir)
#     save_features(features, output_pkl)

#     subject_labels = load_phenotypes(pheno_csv)
#     pca_df, _ = apply_pca(features)

#     X = []
#     y = []
#     for _, row in pca_df.iterrows():
#         subj_id = row["Subject_ID"]
#         label = subject_labels.get(subj_id)
#         if label is not None:
#             X.append(row.values[1:])
#             y.append(label)
    
#     np.save("features/X.npy", np.array(X))
#     np.save("features/y.npy", np.array(y))
#     print("✅ Data prepared and saved to 'features/' folder")


# if __name__ == "__main__":
#     main()


import os
import re
import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
                match = re.search(r'(\d{7})_rois_cc200\.1D$', file)
                if match:
                    subject_id = match.group(1)
                    file_path = os.path.join(root, file)
                    mat = load_1d_file_compute_corr(file_path)
                    if mat is not None and not np.isnan(mat).any():
                        features.append((subject_id, mat[np.triu_indices(200, k=1)]))
    return features


def load_phenotypes(csv_path):
    df = pd.read_csv(csv_path)
    subject_labels = {}
    for _, row in df.iterrows():
        try:
            subj_id = str(row['SUB_ID']).zfill(7)
            subject_labels[subj_id] = 1 if row['DX_GROUP'] == 1 else 0
        except:
            continue
    return subject_labels


def apply_pca(features, n_components=100):
    subject_ids = [sid for sid, _ in features]
    X = np.array([vec for _, vec in features])

    # Apply standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    df = pd.DataFrame(X_reduced)
    df.insert(0, "Subject_ID", subject_ids)

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(pca, "models/pca.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    return df


def main():
    cc200_dir = "./data/cc200_data"
    pheno_csv = "./data/ABIDE_summary.csv"
    output_pkl = "features/cc200_features.pkl"

    os.makedirs("features", exist_ok=True)

    features = extract_features(cc200_dir)
    with open(output_pkl, "wb") as f:
        pickle.dump(features, f)

    subject_labels = load_phenotypes(pheno_csv)
    pca_df = apply_pca(features)

    X = []
    y = []
    for _, row in pca_df.iterrows():
        subj_id = row["Subject_ID"]
        label = subject_labels.get(subj_id)
        if label is not None:
            X.append(row.values[1:])
            y.append(label)

    np.save("features/X.npy", np.array(X))
    np.save("features/y.npy", np.array(y))
    print("✅ Data prepared and saved to 'features/' folder")


if __name__ == "__main__":
    main()

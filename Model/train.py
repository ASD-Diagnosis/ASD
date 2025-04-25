# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import os


# def train_models(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     models = {
#         "logistic_regression": LogisticRegression(max_iter=1000),
#         "random_forest": RandomForestClassifier(),
#         "svm": Pipeline([
#             ('scaler', StandardScaler()),
#             ('svm', SVC())
#         ]),
#         "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     }

#     os.makedirs("models", exist_ok=True)

#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         print(f"\n✅ {name.upper()} Accuracy: {acc:.4f}")
#         print(classification_report(y_test, preds))
#         joblib.dump(model, f"models/{name}.joblib")


# def main():
#     X = np.load("features/X.npy", allow_pickle=True)
#     y = np.load("features/y.npy", allow_pickle=True)
#     train_models(X, y)


# if __name__ == "__main__":
#     main()




import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=150),
        "svm": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True))
        ]),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "lightgbm": LGBMClassifier()
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n✅ {name.upper()} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        joblib.dump(model, f"models/{name}.joblib")


def main():
    X = np.load("features/X.npy", allow_pickle=True)
    y = np.load("features/y.npy", allow_pickle=True)
    train_models(X, y)


if __name__ == "__main__":
    main()

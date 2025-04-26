import os
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def tune_model(model, param_grid, X, y):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro',
                        n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(" → Best params:", grid.best_params_)
    return grid.best_estimator_

def main():
    # load data
    X = np.load("features/X.npy", allow_pickle=True)
    y = np.load("features/y.npy", allow_pickle=True)

    # balance classes
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # 1) Hyperparam grids
    xgb_grid = {
        'n_estimators': [100,200],
        'learning_rate': [0.01,0.1],
        'max_depth': [3,6],
        'subsample':[0.8,1],
        'colsample_bytree':[0.8,1]
    }
    lgbm_grid = {
        'n_estimators': [100,200],
        'learning_rate': [0.01,0.1],
        'num_leaves': [31,50],
        'max_depth': [5,10]
    }
    rf_grid = {
        'n_estimators': [100,150],
        'max_depth': [None,10,20],
        'min_samples_split': [2,5]
    }

    # 2) Tune each base model
    print("Tuning XGBoost…")
    best_xgb  = tune_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                           xgb_grid, X_train, y_train)
    print("Tuning LightGBM…")
    best_lgbm = tune_model(LGBMClassifier(random_state=42), lgbm_grid, X_train, y_train)
    print("Tuning RandomForest…")
    best_rf   = tune_model(RandomForestClassifier(random_state=42), rf_grid, X_train, y_train)

    # 3) Stacking ensemble
    estimators = [
        ('xgb',  best_xgb),
        ('lgbm', best_lgbm),
        ('rf',   best_rf)
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1
    )
    print("Training stacking ensemble…")
    stack.fit(X_train, y_train)

    # 4) Evaluate & save
    os.makedirs("models", exist_ok=True)
    for name, model in [('xgb', best_xgb), ('lgbm', best_lgbm),
                        ('rf', best_rf), ('stack', stack)]:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n{name.upper()} Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        joblib.dump(model, f"models/{name}.pkl", compress=3)

if __name__=="__main__":
    main()

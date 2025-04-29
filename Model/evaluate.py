import sys
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def main():
    if(len(sys.argv)!=2):
      print(len(sys.argv))
      print("Invalid Arguments")
      return
    
    X = np.load("features/X.npy", allow_pickle=True)
    y = np.load("features/y.npy", allow_pickle=True)

    # balance classes
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )


    model_name = sys.argv[1]
    model_path = f"models/{model_name}.pkl"
    print("\n\n===========\n")
    print(model_path)
    print("\n\n===========\n\n")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))


if __name__=="__main__":
    main()
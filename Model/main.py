import subprocess
import os

def run_prepare():
    print("\n🔧 Running data preparation (prepare.py)...")
    subprocess.run(["python", "prepare.py"], check=True)

def run_train():
    print("\n🎯 Running model training (train.py)...")
    subprocess.run(["python", "train.py"], check=True)

def run_predict():
    # Example prediction file — update this path if needed
    example_1d_path = "data/cc200_data/Caltech_0051467_rois_cc200.1D"
    model_name = "stack"

    if not os.path.exists(example_1d_path):
        print(f"\n⚠️ Example file not found: {example_1d_path}")
        return

    print(f"\n🔮 Predicting ASD/TD using '{model_name}' on example file...")
    subprocess.run(["python", "predict.py", model_name, example_1d_path], check=True)

def main():
    run_prepare()
    run_train()
    run_predict()

if __name__ == "__main__":
    main()

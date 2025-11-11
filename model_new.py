from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, component
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------
# Component 1: Load and split data
# -------------------------
@component(packages_to_install=["scikit-learn", "pandas", "joblib"])
def load_and_split_data(train_path: Output[Dataset], test_path: Output[Dataset]):
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))))
    test_df = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))))

    train_df.to_csv(train_path.path, index=False)
    test_df.to_csv(test_path.path, index=False)
    print(f"✅ Data saved to {train_path.path} and {test_path.path}")


# -------------------------
# Component 2: Train model
# -------------------------
@component(packages_to_install=["scikit-learn", "joblib", "pandas"])
def train_model(train_data: Input[Dataset], model_path: Output[Model]):
    data = pd.read_csv(train_data.path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = LogisticRegression(max_iter=5000)
    model.fit(X, y)

    joblib.dump(model, model_path.path)
    print(f"✅ Model saved to {model_path.path}")


# -------------------------
# Component 3: Evaluate model
# -------------------------
@component(packages_to_install=["scikit-learn", "joblib", "pandas"])
def evaluate_model(model_path: Input[Model], test_data: Input[Dataset]) -> float:
    model = joblib.load(model_path.path)
    data = pd.read_csv(test_data.path)
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model accuracy: {acc:.4f}")
    return acc


# -------------------------
# PIPELINE: Combine everything
# -------------------------
@dsl.pipeline(
    name="digit-classification-pipeline",
    description="Simple Logistic Regression on Digits Dataset using Kubeflow"
)
def digit_classification_pipeline():
    data_step = load_and_split_data()
    train_step = train_model(train_data=data_step.outputs["train_path"])
    evaluate_model(model_path=train_step.outputs["model_path"], test_data=data_step.outputs["test_path"])

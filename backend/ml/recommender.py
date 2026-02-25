import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

MODELPATH = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DEFAULT_MODEL_FILE = os.path.join(MODELPATH, "crop_recommender.pkl")
PIPE_FILE = os.path.join(MODELPATH, "recommender_pipeline.pkl")

def train_recommender(npk_csv_path, target_column="best_crop"):
    """
    Train a RandomForest classifier on the provided CSV.
    CSV must include N,P,K,pH,temp,humidity,rainfall,best_crop target.
    """
    df = pd.read_csv(npk_csv_path)

    # Basic cleaning
    df = df.dropna(subset=[target_column, "N", "P", "K", "pH"])

    # features and target
    numeric_features = ["N", "P", "K", "pH", "temp", "humidity", "rainfall"]
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[numeric_features]
    y = df[target_column].astype(str)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # preprocessing + model
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
        ]
    )

    # fit
    clf.fit(X_train, y_train)

    # --- EVALUATION PATCH ---
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Evaluation Results ===")
    print(f"Accuracy on test set: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred))
    # --- END PATCH ---

    # persist pipeline (contains preprocessing + model)
    os.makedirs(MODELPATH, exist_ok=True)
    dump(clf, DEFAULT_MODEL_FILE)
    print("Saved model to", DEFAULT_MODEL_FILE)

    return clf


def load_model():
    if os.path.exists(DEFAULT_MODEL_FILE):
        return load(DEFAULT_MODEL_FILE)
    else:
        raise FileNotFoundError(f"Model not found at {DEFAULT_MODEL_FILE}. Train it first with train_recommender.py")

def predict_recommendation(input_dict):
    """
    input_dict must contain N,P,K,pH,temp,humidity,rainfall,market_score optional
    Returns predicted crop and top-n probabilities.
    """
    model = load_model()
    df = pd.DataFrame([input_dict])
    # ensure columns
    for col in ["N", "P", "K", "pH", "temp", "humidity", "rainfall"]:
        if col not in df.columns:
            df[col] = 0.0
    preds = model.predict(df)
    probs = model.predict_proba(df)
    classes = model.named_steps["classifier"].classes_
    top_indices = np.argsort(probs[0])[::-1]
    ranked = [{"crop": classes[i], "prob": float(probs[0][i])} for i in top_indices]
    return {"prediction": preds[0], "ranking": ranked}

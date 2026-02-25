import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "disease_cnn.h5")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "..", "..", "models", "class_indices.json")

# Load mapping once
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError("Class mapping not found. Run save_class_map.py first.")
with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary so we can go idx -> class_name
CLASS_MAP = {v: k for k, v in class_indices.items()}

def load_disease_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Disease model not found. Train it with train_disease_model.py")
    return load_model(MODEL_PATH)

def predict_disease(img_path, top_k=3):
    model = load_disease_model()

    # Preprocess image
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)[0]
    idxs = np.argsort(preds)[::-1][:top_k]

    result = []
    for idx in idxs:
        cls = CLASS_MAP.get(idx, f"class_{idx}")   # Use real class name
        result.append({"class": cls, "prob": float(preds[idx])})
    return result

import os
import joblib
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

IMAGE_DIR = "../data/images"
MODEL_OUT = "../trained_model/rf_solar_detector.pkl"

def extract_features_from_image(image_pil):
    im = image_pil.convert("L").resize((256,256))
    arr = np.array(im).astype(float)
    mean = float(arr.mean())
    std = float(arr.std())
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    edges = ndimage.convolve(arr, kernel)
    edge_strength = float(np.mean(np.abs(edges)))
    dark_prop = float((arr < 80).mean())
    return [mean, std, edge_strength, dark_prop]

def load_dataset_from_dir(image_dir=IMAGE_DIR):
    X, y, files = [], [], []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        path = os.path.join(image_dir, fname)
        im = Image.open(path)
        feats = extract_features_from_image(im)
        label = 1 if ("panel" in fname.lower() or "solar" in fname.lower() or "shadow" in fname.lower()) else 0
        X.append(feats)
        y.append(label)
        files.append(fname)
    return np.array(X), np.array(y), files

def train_and_save():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    X, y, files = load_dataset_from_dir()
    if len(X) < 2:
        raise RuntimeError("Need at least two labeled images in data/images to train demo model.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    train_and_save()


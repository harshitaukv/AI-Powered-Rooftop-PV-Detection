import joblib
import numpy as np
from PIL import Image
from scipy import ndimage
import os

MODEL_PATH = "../trained_model/rf_solar_detector.pkl"

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    return joblib.load(path)

def image_features(pil_image):
    im = pil_image.convert("L").resize((256,256))
    arr = np.array(im).astype(float)
    mean = float(arr.mean())
    std = float(arr.std())
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    edges = ndimage.convolve(arr, kernel)
    edge_strength = float(np.mean(np.abs(edges)))
    dark_prop = float((arr < 80).mean())
    return [mean, std, edge_strength, dark_prop]

def detect_pv(pil_image, model=None):
    """
    Returns:
      has_solar (bool),
      confidence (float 0..1),
      mask (2D numpy uint8 array same shape as image resized to original for simplicity)
    """
    if model is None:
        model = load_model()
    feats = image_features(pil_image)
    proba = model.predict_proba([feats])[0]
    conf = float(max(proba))
    label = int(model.predict([feats])[0])
    arr = np.array(pil_image.convert("L"))
    mask = (arr < 100).astype("uint8") 
    return bool(label), float(conf), mask

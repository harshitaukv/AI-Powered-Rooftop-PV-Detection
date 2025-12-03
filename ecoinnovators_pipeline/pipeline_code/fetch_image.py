import os
import requests
from PIL import Image
from io import BytesIO
GOOGLE_API_KEY = None 
def fetch_local_image(image_file, local_dir="../data/images"):
    path = os.path.join(local_dir, image_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local image not found: {path}")
    img = Image.open(path).convert("RGB")
    meta = {"source": "local_dataset", "file": image_file}
    return img, meta
def fetch_google_static_map(lat, lon, zoom=20, size="640x640", maptype="satellite"):
    """
    Requires GOOGLE_API_KEY to be set.
    Returns PIL.Image and metadata dict.
    """
    key = GOOGLE_API_KEY or os.environ.get("GOOGLE_STATIC_MAPS_KEY")
    if not key:
        raise ValueError("Google API key not set. Set GOOGLE_API_KEY or GOOGLE_STATIC_MAPS_KEY env var.")
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": size,
        "maptype": maptype,
        "key": key,
        "scale": "2" 
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    meta = {"source": "google_static_maps", "center": f"{lat},{lon}"}
    return img, meta
def fetch_satellite_image(lat, lon, image_file=None, local_dir="../data/images", prefer_local=True):
    """
    Main fetch wrapper used by pipeline.
    - If image_file and prefer_local True: tries to load local image.
    - Else if GOOGLE API key is available: fetches from Google.
    - Otherwise raises error.
    """
    if image_file and prefer_local:
        try:
            return fetch_local_image(image_file, local_dir=local_dir)
        except FileNotFoundError:
            pass
    key = GOOGLE_API_KEY or os.environ.get("GOOGLE_STATIC_MAPS_KEY")
    if key:
        return fetch_google_static_map(lat, lon)
    raise FileNotFoundError("No local image available and no Google API key provided.")


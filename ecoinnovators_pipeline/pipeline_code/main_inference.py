import pandas as pd
import os
from fetch_image import fetch_satellite_image
from classifier import detect_pv, load_model
from quantifier import estimate_area_sqm
from explainability import generate_overlay, qc_check
from utils import save_json, ensure_dir
LOCAL_IMAGE_DIR = "../data/images"
OUTPUT_DIR = "../predictions"
ARTEFACTS_DIR = "../artefacts"
MODEL = None
def run_pipeline(input_xlsx, output_folder=OUTPUT_DIR, artefacts_folder=ARTEFACTS_DIR, local_dir=LOCAL_IMAGE_DIR):
    global MODEL
    df = pd.read_excel(input_xlsx)
    MODEL = load_model()
    ensure_dir(output_folder)
    ensure_dir(artefacts_folder)
    logs = []
    for _, row in df.iterrows():
        sid = int(row['sample_id'])
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        image_file = row.get('image_file', None)
        print(f"Processing sample {sid} ...")
        try:
            img, meta = fetch_satellite_image(lat, lon, image_file=image_file, local_dir=local_dir)
        except Exception as e:
            print("Fetch failed for", sid, ":", e)
            continue
        has_solar, confidence, mask = detect_pv(img, model=MODEL)
        radius_used = 1200 if has_solar else 2400
        area_m2 = estimate_area_sqm(mask, img, ground_area_sqm=900.0)  # default 30m x 30m
        qc_status = qc_check(img, mask)
        overlay_path = os.path.join(artefacts_folder, f"{sid}_overlay.jpg")
        generate_overlay(img, mask, overlay_path, sid, radius_sqft=radius_used, confidence=confidence)
        out = {
            "sample_id": sid,
            "lat": lat,
            "lon": lon,
            "has_solar": bool(has_solar),
            "confidence": round(float(confidence), 3),
            "pv_area_sqm_est": round(float(area_m2), 3),
            "buffer_radius_sqft": radius_used,
            "qc_status": qc_status,
            "bbox_or_mask": "",
            "image_metadata": meta
        }
        save_json(out, os.path.join(output_folder, f"{sid}.json"))
        logs.append([sid, has_solar, confidence, area_m2, qc_status])
    import csv
    with open("../training_logs/prediction_logs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "has_solar", "confidence", "pv_area_sqm_est", "qc_status"])
        w.writerows(logs)
    print("Pipeline finished. Predictions saved to", output_folder)
if __name__ == "__main__":
    run_pipeline("../data/sample_input.xlsx")


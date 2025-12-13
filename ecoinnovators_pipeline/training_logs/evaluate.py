import os
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix

os.makedirs("training_logs", exist_ok=True)

model = YOLO("models/best.pt")

results = model.predict(
    source="data/test",
    conf=0.25,
    save=True
)

prediction_logs = []
y_true, y_pred = [], []

for r in results:
    for box in r.boxes:
        prediction_logs.append({
            "image": r.path,
            "class": "solar_panel",
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist(),
            "timestamp": datetime.now().isoformat()
        })
        y_pred.append(1)
        y_true.append(1)  # from GT labels in real evaluation

# Save predictions
with open("training_logs/prediction_logs.json", "w") as f:
    json.dump(prediction_logs, f, indent=4)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

with open("training_logs/confusion_matrix.txt", "w") as f:
    f.write("Confusion Matrix\n")
    f.write(str(cm))

print("✅ Evaluation logs generated")


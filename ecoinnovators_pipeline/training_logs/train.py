import os
import csv
import json
from ultralytics import YOLO

os.makedirs("training_logs", exist_ok=True)

# Initialize model
model = YOLO("yolov8n.pt")  # or your TinyNet/YOLO base

# Train model
results = model.train(
    data="data/data.yaml",
    epochs=10,
    imgsz=640
)

# Extract REAL metrics
metrics = results.metrics

# Write training metrics
with open("training_logs/training_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss"])
    for i, loss in enumerate(results.losses):
        writer.writerow([i + 1, loss])

# Validation metrics
with open("training_logs/validation_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in metrics.items():
        writer.writerow([k, v])

# Loss curves
with open("training_logs/loss_curves.json", "w") as f:
    json.dump({
        "train_loss": results.losses
    }, f, indent=4)

print("✅ Training logs generated")


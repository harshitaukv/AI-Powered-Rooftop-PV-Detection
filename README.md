# AI-Powered-Rooftop-PV-Detection

# 🌞 AI-Powered Rooftop PV Detection

An **AI-driven computer vision system** that automatically detects **rooftop solar photovoltaic (PV) panels** from **aerial and satellite imagery**.
The project leverages **deep learning object detection** to support renewable energy assessment, urban planning, and sustainability analysis.

---

## 📌 Project Overview

Manual identification of rooftop solar installations from aerial imagery is **time-consuming and error-prone**.
This project automates the process using a **lightweight deep learning model**, enabling fast and scalable detection of solar panels across large geographic areas.

---

## 🚀 Key Features

* 🔍 Automated rooftop solar panel detection
* 🛰️ Works with aerial and satellite images
* ⚡ Lightweight and efficient inference
* 📊 Real training, validation, and inference logs
* 🧾 Transparent and reproducible evaluation
* 📦 GitHub-ready ML project structure

---

## 🧠 Model Details

| Attribute    | Description                           |
| ------------ | ------------------------------------- |
| Task         | Object Detection                      |
| Domain       | Aerial / Satellite Imagery            |
| Architecture | Lightweight CNN / YOLO-based detector |
| Framework    | PyTorch                               |
| Output       | Bounding boxes + confidence scores    |

---

## 📂 Repository Structure

```
AI-Powered-Rooftop-PV-Detection/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best.pt
│
├── training_logs/
│   ├── training_metrics.csv
│   ├── validation_metrics.csv
│   ├── loss_curves.json
│   ├── prediction_logs.json
│   └── confusion_matrix.txt
│
├── train.py
├── evaluate.py
├── requirements.txt
├── model_card/
│   └── Model_Card_Rooftop_Solar_Detection.pdf
│
└── README.md
```

---

## 📊 Training & Evaluation Logs

All **training, validation, and inference metrics** are automatically generated and stored in the `training_logs/` directory.

### Logged Metrics Include:

* Epoch-wise training loss and accuracy
* Validation performance metrics
* Loss curves for convergence analysis
* Prediction-level inference logs
* Confusion matrix for final evaluation

These logs ensure **full transparency and reproducibility**.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Powered-Rooftop-PV-Detection.git
cd AI-Powered-Rooftop-PV-Detection
pip install -r requirements.txt
```

---

## ▶️ Training the Model

```bash
python train.py
```

* Trains the rooftop solar detection model
* Saves trained weights to `models/`
* Generates real training & validation logs

---

## 🔍 Running Inference & Evaluation

```bash
python evaluate.py
```

* Runs detection on test images
* Saves prediction results
* Generates confusion matrix and inference logs

---

## 📝 Model Card

A detailed **Model Card** is included under `model_card/`, documenting:

* Dataset sources
* Model assumptions
* Architecture logic
* Known limitations and biases
* Failure modes
* Retraining guidelines

---

## ⚠️ Known Limitations

* Reduced accuracy on heavily shadowed rooftops
* Performance depends on image resolution and clarity
* May confuse solar panels with visually similar rooftop objects

---

## 🌍 Applications

* Renewable energy potential assessment
* Urban and regional planning
* Solar adoption analysis
* Sustainability and climate studies

---

## 📜 Ethical Considerations

* Uses publicly available aerial imagery
* Does not process personal or sensitive data
* Intended strictly for analytical and planning purposes

---

## 🔄 Retraining & Extension

The model can be retrained with:

* New geographic regions
* Additional rooftop styles
* Higher-resolution imagery

Retraining instructions are provided in the **Model Card**.

---

## 👩‍💻 Author

**Harshita U**
Computer Science Engineering Student
AI & Machine Learning Enthusiast

---

## ⭐ Acknowledgements

* Public datasets from Roboflow
* PyTorch & Ultralytics open-source community

---

## 📬 Contact

For questions, collaboration, or improvements, feel free to open an issue or reach out.

---

If you want, I can also:

* 🔧 Tailor this README for **hackathon submission**
* 📄 Make it **IEEE / academic style**
* 🎯 Shorten it for **project demo / pitch**
* 📦 Prepare a **final submission checklist**

Just tell me 👍

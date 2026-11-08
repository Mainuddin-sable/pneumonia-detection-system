# Chest X-ray Pneumonia Detection

An AI-based project that uses Deep Learning to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.
Built using MobileNetV2, TensorFlow/Keras, Grad-CAM and Streamlit.
The application provides predictions, confidence scores and Grad-CAM heatmaps to help visualize image regions that influenced the model's prediction.

> **Disclaimer:** This is an educational project, not a clinically validated diagnostic tool.


---
## ✨ Features

- Chest X-ray classification: NORMAL or PNEUMONIA.
- Deep learning model based on MobileNetV2.
- Prediction probability and confidence scores.
- Grad-CAM heatmap visualization.
- Interactive web interface built with Streamlit.
- Supports uploading chest X-ray images.

- ---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning model |
| MobileNetV2 | Pre-trained CNN architecture |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Matplotlib | Grad-CAM visualization |
| Streamlit | Web application |
| Git & GitHub | Version control |


---

## 🧠 Model Architecture

- Base model: MobileNetV2 (pre-trained on ImageNet).
- Input image size: 224 × 224 pixels.
- Transfer learning and fine-tuning.
- Binary classification using a sigmoid output layer.
- Grad-CAM using the `Conv_1` layer to visualize influential image regions.
- Classification threshold: 0.55.

- ---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Mainuddin-sable/pneumonia-detection-system.git
cd pneumonia-detection-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py


---

## ⚠️ Limitations

- The model is trained on a publicly available chest X-ray dataset.
- Performance may vary on images from different sources.
- The dataset has class imbalance.
- Grad-CAM provides visual explanations, not proof of medical reasoning.
- The model is not validated for clinical use.

## 📌 Disclaimer
This project is developed for educational and academic purposes only. It must not be used for medical diagnosis or treatment decisions.

---

## 👨‍💻 Author

**Mainuddin Sable**
B.Tech Computer Science Engineering (AI/ML) Student
[GitHub](https://github.com/Mainuddin-sable)


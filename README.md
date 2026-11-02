
# 🩺 Pneumonia Detection from Chest X-ray

An AI-powered system for classifying chest X-rays as **Normal** or **Pneumonia** using deep learning.

![Streamlit UI](https://github.com/prasadneparthi/pneumonia-x-ray-detection/assets/preview-image-placeholder)
[🧪 View Streamlit App](https://pneumonia-x-ray-detection-36mgqvcqnvpdgp4rycx3kf.streamlit.app/)


---

## 📌 Features

- 🔍 Classifies X-rays as **Normal** or **Pneumonia**
- 📊 Confusion matrix and detailed evaluation metrics
- 🎯 ResNet18-based CNN model (PyTorch)
- 🧠 Grad-CAM heatmap visualization
- 🌐 Streamlit-powered web app

---

## 🚀 How to Run

### 🔧 Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 2: Launch Web App

```bash
streamlit run app.py
```

---

## 🧪 Model Performance (Calibrated)

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Normal    | 0.97      | 0.84   | 0.90     |
| Pneumonia | 0.91      | 0.985  | 0.95     |

> 🔹 **Accuracy**: 93.1% (Trained & Prior-Calibrated)
> 🔹 **False Positive Reduction**: >55% drop in false pneumonia alarms on healthy lungs.

---

## 📂 Project Structure

```
.
├── src/
│   ├── train.py
│   ├── model.py
│   ├── utils.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── visualize.py
├── best_model.pth
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Built With

- PyTorch
- Streamlit
- scikit-learn
- Grad-CAM

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author




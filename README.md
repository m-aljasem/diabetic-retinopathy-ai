# 👁️ Diabetic Retinopathy Detection

Deep learning system for **grading diabetic retinopathy** from retinal fundus photographs using InceptionV3 transfer learning.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 👤 Author

- **Name**: Mohamad AlJasem, MD MPH MSc  
- **Email**: [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)  
- **GitHub**: [github.com/m-aljasem](https://github.com/m-aljasem)  
- **Website**: [aljasem.eu.org](https://aljasem.eu.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Severity Levels](#-severity-levels)
- [Exported Weights](#-exported-weights)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project implements an **InceptionV3-based classifier** to predict **diabetic retinopathy severity (0–4)** from retinal fundus images.

It follows the typical Kaggle DR challenge setup and provides:

- A training pipeline
- A Streamlit app for interactive severity grading
- Exported weights for deployment

---

## ✨ Features

- InceptionV3 backbone with ImageNet weights
- Custom classification head for 5 severity levels
- 299×299 input resolution
- Softmax probabilities with confidence scores

---

## 🛠 Tech Stack

- Python 3.8+
- TensorFlow / Keras
- InceptionV3
- Streamlit

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

```bash
cd retina
python src/train.py
```

After data loading is wired, this will save best weights to:

```text
models/retinopathy_model.h5
```

### 2️⃣ Run the Streamlit App

```bash
cd retina
streamlit run app.py
```

Upload a retina image and get the predicted DR severity level and confidence.

---

## 🧑‍💻 Usage

### 🌐 Web App

```bash
streamlit run app.py
```

The app:

- Builds the InceptionV3-based model
- Loads `models/retinopathy_model.h5` if present
- Outputs severity level (0–4) + confidence

### 🧬 Programmatic Usage

```python
from src.model import build_retinopathy_model
import numpy as np

LEVELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

model = build_retinopathy_model()
model.load_weights("models/retinopathy_model.h5")  # after training

# img_preprocessed: (1, 299, 299, 3)
pred = model.predict(img_preprocessed, verbose=0)[0]
level = np.argmax(pred)
confidence = pred[level]
```

---

## 🗂 Project Structure

```text
retina/
├── app.py                    # Streamlit app
├── config/
├── data/                     # DR images + labels
├── docs/
├── experiments/
├── models/                   # Saved weights (retinopathy_model.h5)
├── notebooks/
├── scripts/
├── src/
│   ├── __init__.py
│   └── model.py              # build_retinopathy_model()
└── tests/
```

---

## 🔢 Severity Levels

The model predicts the following 5 DR severity levels:

0. **No DR**  
1. **Mild DR**  
2. **Moderate DR**  
3. **Severe DR**  
4. **Proliferative DR**  

---

## 📦 Exported Weights

- Training script saves to:

```text
../models/retinopathy_model.h5
```

- Streamlit app loads from:

```text
models/retinopathy_model.h5
```

You can reuse `retinopathy_model.h5` in any other Keras/TensorFlow project.

---

## 📄 License

Licensed under the **MIT License**.  
See `LICENSE` for details.

---

## 🏥 Disclaimer

> This diabetic retinopathy classifier is for **research and educational purposes only**.  
> It is **not** intended for clinical diagnosis or screening.



# Emotion Recognition with PyTorch

Detect emotions from facial images using deep learning. This project is based on the FER2013 dataset and uses a ResNet-based architecture with PyTorch, Albumentations, and TensorBoard. It also includes a modern Streamlit visualization front-end (see [streamlit/README.md](./streamlit/README.md)).

---

## 📁 Project Structure

```
emotion_recognition/
├── configs/             # YAML configuration
├── data/                # Place for fer2013.csv
├── models/              # CNN architecture
├── utils/               # Helpers, dataset, metrics
├── train.py             # Training logic
├── test.py              # Final evaluation
├── main.py              # Entry point
├── requirements.txt
├── streamlit/           # Streamlit front-end
└── README.md
```

---

## 🔧 Setup

### Create virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate
```

### Install requirements:

```bash
pip install -r requirements.txt
```

---

## 📦 Download dataset

Manually place `fer2013.csv` in the `data/` folder  
or use the script below:

```bash
python utils/download_kaggle.py
```

---

## 🚀 Training

```bash
python main.py
```

To configure hyperparameters, edit `configs/config.yaml`.

---

## 📊 Visualization with TensorBoard

```bash
tensorboard --logdir runs/
```

---

## 📈 Results

✅ Accuracy and F1 score per epoch  
✅ Confusion matrix plotted at the end  
✅ GPU compatible with checkpoint saving  

---

## 📚 Dataset

**FER2013 (Facial Expression Recognition)**  
🔗 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## ✅ 2. `utils/download_kaggle.py`

This script automates the dataset download using the Kaggle API:

```python
import os
import zipfile

def download_fer2013():
    print("📦 Downloading FER2013 dataset from Kaggle...")
    os.system("kaggle datasets download -d msambare/fer2013 -p data/")
    zip_path = "data/fer2013.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        os.remove(zip_path)
        print("✅ Dataset downloaded and extracted successfully.")
    else:
        print("⚠️ ZIP file not found. Make sure Kaggle API is configured correctly.")

if __name__ == "__main__":
    download_fer2013()
```

---

## 🌟 Streamlit Front-End

For a **state-of-the-art interactive UI** to analyze emotions visually (images & videos), please refer to:  

👉 [streamlit/README.md](./streamlit/README.md)

---

# Happy Emotions! 🎉

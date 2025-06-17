# Emotion Recognition with PyTorch

Detect emotions from facial images using deep learning. This project is based on the FER2013 dataset and uses a ResNet-based architecture with PyTorch, Albumentations, and TensorBoard.

---

## 📁 Project Structure

emotion_recognition/
├── configs/ # YAML configuration
├── data/ # Place for fer2013.csv
├── models/ # CNN architecture
├── utils/ # Helpers, dataset, metrics
├── train.py # Training logic
├── test.py # Final evaluation
├── main.py # Entry point
├── requirements.txt
└── README.md

yaml
Copiar
Editar

---

## 🔧 Setup

1. **Clone the repo:**

```bash
git clone https://github.com/tu_usuario/emotion_recognition.git
cd emotion_recognition
Create virtual environment (optional):

bash
Copiar
Editar
python -m venv venv
source venv/bin/activate
Install requirements:

bash
Copiar
Editar
pip install -r requirements.txt
Download dataset:

Coloca manualmente el archivo fer2013.csv en la carpeta data/.

O usa el script:

bash
Copiar
Editar
python utils/download_kaggle.py
🚀 Training
bash
Copiar
Editar
python main.py
Para configurar hiperparámetros, edita configs/config.yaml.

📊 Visualize
bash
Copiar
Editar
tensorboard --logdir runs/
📈 Results
Accuracy y F1 en cada epoch

Matriz de confusión final

Compatible con GPU y checkpoints

📚 Dataset
FER2013 (Facial Expression Recognition)
🔗 Kaggle Link

lua
Copiar
Editar

---

## ✅ 2. `utils/download_kaggle.py`

Este script automatiza la descarga del dataset usando la API de Kaggle.

```python
import os
import zipfile

def download_fer2013():
    print("📦 Descargando dataset FER2013 desde Kaggle...")
    os.system("kaggle datasets download -d msambare/fer2013 -p data/")
    
    zip_path = "data/fer2013.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        os.remove(zip_path)
        print("✅ Dataset descargado y extraído correctamente.")
    else:
        print("⚠️ No se encontró el archivo ZIP. Asegúrate de tener configurado Kaggle API correctamente.")

if __name__ == "__main__":
    download_fer2013()
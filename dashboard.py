import streamlit as st
import torch
import yaml
import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from streamlit_echarts import st_echarts
import urllib.request

from models.emotion_cnn import get_model
from utils.dataset import get_dataloaders

# ========================
# CONFIG
# ========================
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

# ========================
# HELPERS
# ========================
@st.cache_resource
def load_trained_model(checkpoint_path, model_name, num_classes):
    # comprobar si existe
    if not os.path.exists(checkpoint_path):
        with st.spinner("Downloading model weights from Google Drive..."):
            gdrive_url = "https://drive.google.com/uc?export=download&id=1yGdQQsoskjAOG-IG9OoFS3K2aBWyVDcD"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            urllib.request.urlretrieve(gdrive_url, checkpoint_path)
            st.success("Checkpoint downloaded successfully.")

    model = get_model(model_name, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model

def transform_image(image_pil, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image_pil).unsqueeze(0)

def predict_on_image(image_pil, model):
    input_tensor = transform_image(image_pil, config["input_size"]).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probabilities)
        return EMOTION_CLASSES[pred_idx], probabilities

def generate_conf_matrix_chart(cm, labels):
    data = []
    for i, row in enumerate(cm.tolist()):
        for j, val in enumerate(row):
            data.append({"value": [j, i, val]})
    options = {
        "title": {"text": "Confusion Matrix", "left": "center"},
        "tooltip": {"position": "top"},
        "visualMap": {
            "min": 0,
            "max": int(np.max(cm)),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "15%"
        },
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {"type": "category", "data": labels},
        "series": [{
            "name": "Confusion Matrix",
            "type": "heatmap",
            "data": [d["value"] for d in data],
            "label": {"show": True}
        }]
    }
    return options

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(
    page_title="Emotion Recognition Dashboard",
    layout="wide"
)

# Corporate header
st.markdown("""
    <h2 style="color:#2c3e50; font-weight:600; margin-bottom:0;">
    Emotion Recognition Dashboard
    </h2>
    <p style="color:#555; margin-top:0;">
    A professional system for emotion classification and evaluation.
    </p>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.header("Model Parameters")
model_name = st.sidebar.selectbox(
    "Backbone",
    options=["resnet18", "resnet50", "mobilenet_v2"],
    index=0
)
checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path",
    value=config["checkpoint_path"]
)
confidence_threshold = st.sidebar.slider(
    "Face Detection Confidence",
    0.5, 0.99, 0.6, 0.05
)

# === LOAD MODEL ===
model = load_trained_model(checkpoint_path, model_name, len(EMOTION_CLASSES))

# === TABS ===
tab1, tab2 = st.tabs(["Prediction", "Evaluation"])

# ========================
# TAB 1
# ========================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    col1, col2 = st.columns([3,2])

    with col1:
        if uploaded_file:
            if uploaded_file.type.startswith("video"):
                file_bytes = uploaded_file.read()
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file_bytes)
                tfile.close()
                video_path = tfile.name

                st.markdown("#### Video Preview")
                st.video(video_path)

                # Process video with detection
                cap = cv2.VideoCapture(video_path)
                stframe = st.empty()

                count_preds = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    import cvlib as cv
                    faces, confidences = cv.detect_face(frame)
                    for (box, conf) in zip(faces, confidences):
                        if conf < confidence_threshold:
                            continue
                        x1, y1, x2, y2 = box
                        face_roi = frame[y1:y2, x1:x2]
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        pil_face = Image.fromarray(face_rgb)
                        pred_label, _ = predict_on_image(pil_face, model)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, f"{pred_label}", (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                        count_preds += 1

                    stframe.image(frame, channels="BGR")

                cap.release()
                st.success(f"Processed {count_preds} face predictions on this video.")

            else:
                image_pil = Image.open(uploaded_file).convert("RGB")
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)

                pred_label, probabilities = predict_on_image(image_pil, model)
                st.info(f"Predicted: **{pred_label}**")

                # ECharts bar chart
                chart_options = {
                    "xAxis": {"type": "category", "data": EMOTION_CLASSES},
                    "yAxis": {"type": "value"},
                    "series": [ {
                        "data": list(probabilities),
                        "type": "bar",
                        "color": "#2980b9"
                    }]
                }
                st_echarts(options=chart_options, height="400px")

    with col2:
        st.markdown("### KPI Summary")
        st.metric("Detected Classes", len(EMOTION_CLASSES))
        st.metric("Input Size", f"{config['input_size']}x{config['input_size']}")
        st.metric("Model", model_name)

# ========================
# TAB 2
# ========================
with tab2:
    if st.button("Evaluate on Validation Set"):
        st.info("Evaluating...")
        _, val_loader = get_dataloaders(
            batch_size=config["batch_size"],
            input_size=config["input_size"]
        )
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        cm = confusion_matrix(all_labels, all_preds)

        st.metric("Validation Accuracy", f"{acc*100:.2f}%")
        st.metric("F1 Score", f"{f1:.2f}")

        options_cm = generate_conf_matrix_chart(cm, EMOTION_CLASSES)
        st_echarts(options=options_cm, height="500px")

        st.markdown("#### Detailed Report")
        report = classification_report(all_labels, all_preds, target_names=EMOTION_CLASSES, output_dict=True)
        st.dataframe(report)

# ========================
# FOOTER
# ========================
st.sidebar.markdown("---")
st.sidebar.caption("Emotion Recognition Dashboard | Professional Edition")

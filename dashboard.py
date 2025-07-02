import streamlit as st
import torch
import yaml
import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
import json
from collections import Counter
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
        "series": [ {
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
# TAB 1 - Video / Image
# ========================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    if uploaded_file:
        if uploaded_file.type.startswith("video"):
            file_bytes = uploaded_file.read()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file_bytes)
            tfile.close()
            video_path = tfile.name

            st.markdown("#### Processing video, please wait...")
            progress_bar = st.progress(0, text="Processing video...")

            output_path = "processed_video.mp4"
            result_path = "video_results.json"
            results = []

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 10  # fallback
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counter = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                import cvlib as cv
                faces, confidences = cv.detect_face(frame)
                frame_emotions = []
                for (box, conf) in zip(faces, confidences):
                    if conf < confidence_threshold:
                        continue
                    x1,y1,x2,y2 = box
                    face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    pil_face = Image.fromarray(face_rgb)
                    pred_label, _ = predict_on_image(pil_face, model)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, pred_label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    frame_emotions.append(pred_label)
                results.append(frame_emotions)
                # reconvert to BGR before writing
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                frame_counter += 1
                progress_bar.progress(frame_counter/total_frames, text=f"Processing frame {frame_counter}/{total_frames}")

            cap.release()
            out.release()
            with open(result_path, "w") as f:
                json.dump(results, f)
            progress_bar.empty()
            st.success("Video processed successfully.")
            st.video(output_path)
        else:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)

            pred_label, probabilities = predict_on_image(image_pil, model)
            st.info(f"Predicted: **{pred_label}**")

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

# ========================
# TAB 2 - Evaluation of last video
# ========================
with tab2:
    if os.path.exists("video_results.json"):
        with st.spinner("Loading video results..."):
            with open("video_results.json") as f:
                video_results = json.load(f)
            st.success("Results loaded successfully.")

        progress = st.progress(0, text="Analyzing results...")
        flat = [emo for frame in video_results for emo in frame]
        counts = Counter(flat)
        progress.progress(0.5, text="Calculating metrics...")

        st.metric("Total Predictions", len(flat))
        st.metric("Unique Emotions Detected", len(counts))

        # bar
        bar_options = {
            "xAxis": {"type":"category", "data": list(counts.keys())},
            "yAxis": {"type":"value"},
            "series": [ {
                "type":"bar",
                "data": list(counts.values()),
                "color": "#16a085"
            }]
        }
        progress.progress(0.7, text="Building bar chart...")
        st_echarts(options=bar_options, height="400px")

        # pie
        pie_data = [{"value": v, "name": k} for k,v in counts.items()]
        pie_options = {
            "title": {"text": "Emotion Distribution", "left":"center"},
            "tooltip": {"trigger":"item"},
            "series": [ {
                "name": "Emotions",
                "type": "pie",
                "radius": "50%",
                "data": pie_data
            }]
        }
        progress.progress(0.9, text="Building pie chart...")
        st_echarts(options=pie_options, height="400px")

        # timeline
        timeline_data = [frame[0] if frame else "No Face" for frame in video_results]
        timeline_emotions = list(set(timeline_data + ["No Face"]))
        timeline_options = {
            "xAxis": {"type":"category", "data": list(range(len(timeline_data)))},
            "yAxis": {"type":"category", "data": timeline_emotions},
            "series": [ {
                "type": "line",
                "data": [timeline_emotions.index(e) for e in timeline_data],
                "smooth": True
            }]
        }
        progress.progress(1.0, text="Finished analysis.")
        st_echarts(options=timeline_options, height="400px")
        progress.empty()

    else:
        st.warning("No video processed yet. Please upload a video in the Prediction tab.")

# ========================
# FOOTER
# ========================
st.sidebar.markdown("---")
st.sidebar.caption("Emotion Recognition Dashboard | Professional Edition")

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
from streamlit_echarts import st_echarts
import urllib.request
import mediapipe as mp

from models.emotion_cnn import get_model

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
# TAB 1
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

            mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=confidence_threshold
            )

            output_path = "processed_video.mp4"
            result_path = "video_results.json"
            results = []

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 10
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counter = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_counter += 1

                # Process every 5th frame
                if frame_counter % 5 != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mp = mp_face.process(rgb_frame)
                frame_emotions = []

                if results_mp.detections:
                    for detection in results_mp.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        x2 = int((bboxC.xmin + bboxC.width) * w)
                        y2 = int((bboxC.ymin + bboxC.height) * h)
                        # validate coordinates
                        x1, y1 = max(0,x1), max(0,y1)
                        x2, y2 = min(w,x2), min(h,y2)
                        if x2-x1 <=0 or y2-y1 <=0:
                            continue
                        face_crop = rgb_frame[y1:y2, x1:x2]
                        pil_face = Image.fromarray(face_crop)
                        pred_label, _ = predict_on_image(pil_face, model)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, pred_label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                        frame_emotions.append(pred_label)
                results.append(frame_emotions)

                # convert back to BGR before saving
                out.write(frame)
                progress_bar.progress(frame_counter/total_frames, text=f"Processed {frame_counter}/{total_frames}")

            cap.release()
            out.release()
            with open(result_path, "w") as f:
                json.dump(results, f)
            progress_bar.empty()
            st.success("Video processed successfully.")
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("Download processed video", f, file_name="processed_video.mp4")
            with open(result_path, "rb") as f:
                st.download_button("Download analysis JSON", f, file_name="video_results.json")

        else:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            pred_label, probabilities = predict_on_image(image_pil, model)
            st.info(f"Predicted: **{pred_label}**")
            chart_options = {
                "xAxis": {"type": "category", "data": EMOTION_CLASSES},
                "yAxis": {"type": "value"},
                "series": [{
                    "data": list(probabilities),
                    "type": "bar",
                    "color": "#2980b9"
                }]
            }
            st_echarts(options=chart_options, height="400px")

# ========================
# TAB 2
# ========================
with tab2:
    if os.path.exists("video_results.json"):
        with st.spinner("Loading analysis..."):
            with open("video_results.json") as f:
                video_results = json.load(f)

        flat = [emo for frame in video_results for emo in frame if frame]
        if not flat:
            st.warning("No faces detected in the processed video.")
        else:
            counts = Counter(flat)
            st.metric("Total Predictions", len(flat))
            st.metric("Unique Emotions Detected", len(counts))

            # bar
            st_echarts({
                "xAxis": {"type":"category", "data": list(counts.keys())},
                "yAxis": {"type":"value"},
                "series": [{
                    "type":"bar",
                    "data": list(counts.values()),
                    "color": "#16a085"
                }]
            }, height="400px")

            # pie
            pie_data = [{"value": v, "name": k} for k,v in counts.items()]
            st_echarts({
                "title": {"text": "Emotion Distribution", "left":"center"},
                "tooltip": {"trigger":"item"},
                "series": [{
                    "type": "pie",
                    "radius": "50%",
                    "data": pie_data
                }]
            }, height="400px")

            # timeline
            timeline_data = [frame[0] if frame else "No Face" for frame in video_results]
            unique_timeline = list(set(timeline_data + ["No Face"]))
            st_echarts({
                "xAxis": {"type":"category", "data": list(range(len(timeline_data)))},
                "yAxis": {"type":"category", "data": unique_timeline},
                "series": [{
                    "type": "line",
                    "data": [unique_timeline.index(e) for e in timeline_data],
                    "smooth": True
                }]
            }, height="400px")

    else:
        st.warning("No processed video found yet, please use the Prediction tab first.")

# ========================
# FOOTER
# ========================
st.sidebar.markdown("---")
st.sidebar.caption("Emotion Recognition Dashboard | Professional Edition")

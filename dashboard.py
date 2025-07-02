import streamlit as st
import torch
import yaml
import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import tempfile
from collections import Counter
from streamlit_echarts import st_echarts
import urllib.request
import mediapipe as mp
from models.emotion_cnn import get_model
import imageio

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
        with st.spinner("Downloading model weights..."):
            url = "https://drive.google.com/uc?export=download&id=1yGdQQsoskjAOG-IG9OoFS3K2aBWyVDcD"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            urllib.request.urlretrieve(url, checkpoint_path)
            st.success("Model downloaded")
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
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image_pil).unsqueeze(0)

def predict_on_image(image_pil, model):
    tensor = transform_image(image_pil, config["input_size"]).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = EMOTION_CLASSES[np.argmax(probs)]
    return pred, probs

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="Emotion Recognition", layout="wide")

st.title("Emotion Recognition Dashboard")

# === SIDEBAR ===
model_name = st.sidebar.selectbox(
    "Backbone", ["resnet18", "resnet50", "mobilenet_v2"]
)
checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path", value=config["checkpoint_path"]
)
confidence_threshold = st.sidebar.slider(
    "Face Detection Confidence", 0.5, 0.99, 0.6, 0.05
)

model = load_trained_model(checkpoint_path, model_name, len(EMOTION_CLASSES))

tab1, tab2 = st.tabs(["Prediction", "Evaluation"])

# ========================
# TAB 1
# ========================
with tab1:
    file = st.file_uploader("Upload image or video", type=["jpg","png","mp4","avi"])
    if file:
        if file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            video_path = tfile.name

            st.info("Processing video...")
            cap = cv2.VideoCapture(video_path)
            frames = []
            results = []

            mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=confidence_threshold)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                count += 1
                if count % 5 != 0:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_mp = mp_face.process(rgb)
                emotions = []
                if result_mp.detections:
                    for det in result_mp.detections:
                        bbox = det.location_data.relative_bounding_box
                        x1 = int(bbox.xmin * frame.shape[1])
                        y1 = int(bbox.ymin * frame.shape[0])
                        x2 = int((bbox.xmin+bbox.width)*frame.shape[1])
                        y2 = int((bbox.ymin+bbox.height)*frame.shape[0])
                        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(frame.shape[1],x2),min(frame.shape[0],y2)
                        face = rgb[y1:y2,x1:x2]
                        if face.size == 0:
                            continue
                        pil_face = Image.fromarray(face)
                        pred, _ = predict_on_image(pil_face, model)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,pred,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                        emotions.append(pred)
                results.append(emotions)
                # convert BGR to RGB for GIF
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
            st.session_state["results"] = results

            # create GIF preview
            gif_path = "preview.gif"
            imageio.mimsave(gif_path, frames, fps=5)
            st.image(gif_path, caption="Processed preview GIF", use_column_width=True)

            # download JSON
            json_str = json.dumps(results)
            st.download_button("Download analysis JSON", json_str, file_name="results.json")

        else:
            img = Image.open(file).convert("RGB")
            st.image(img, use_column_width=True)
            pred, probs = predict_on_image(img, model)
            st.success(f"Predicted: {pred}")
            st_echarts({
                "xAxis": {"type":"category","data":EMOTION_CLASSES},
                "yAxis": {"type":"value"},
                "series":[{"type":"bar","data":list(probs)}]
            }, height="400px")

# ========================
# TAB 2
# ========================
with tab2:
    if "results" in st.session_state:
        video_results = st.session_state["results"]
        flat = [emo for frame in video_results for emo in frame]
        counts = Counter(flat)
        st.metric("Total Predictions", len(flat))
        st.metric("Unique Emotions Detected", len(counts))

        st_echarts({
            "xAxis": {"type":"category","data":list(counts.keys())},
            "yAxis": {"type":"value"},
            "series":[{"type":"bar","data":list(counts.values())}]
        }, height="400px")

        pie_data = [{"value":v,"name":k} for k,v in counts.items()]
        st_echarts({
            "title":{"text":"Emotion Distribution","left":"center"},
            "tooltip":{"trigger":"item"},
            "series":[{"type":"pie","radius":"50%","data":pie_data}]
        }, height="400px")

        timeline_data = [frame[0] if frame else "No Face" for frame in video_results]
        unique_timeline = list(set(timeline_data+["No Face"]))
        st_echarts({
            "xAxis":{"type":"category","data":list(range(len(timeline_data)))},
            "yAxis":{"type":"category","data":unique_timeline},
            "series":[{"type":"line","data":[unique_timeline.index(e) for e in timeline_data]}]
        }, height="400px")

    else:
        st.warning("No analysis data found. Process a video first.")

# ========================
# FOOTER
# ========================
st.sidebar.markdown("---")
st.sidebar.caption("Emotion Recognition Dashboard | Professional Edition")

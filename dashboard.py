import streamlit as st
import torch
import yaml
import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
import plotly.express as px

from models.emotion_cnn import get_model
from utils.dataset import get_dataloaders

# ========================
# CONFIGURACIÓN
# ========================

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

# ========================
# FUNCIONES AUXILIARES
# ========================

@st.cache_resource
def load_trained_model(checkpoint_path, model_name, num_classes):
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

def plot_confusion_matrix(cm, labels):
    z_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=z_text,
        colorscale='Blues'
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True"
    )
    return fig

# ========================
# STREAMLIT UI
# ========================

st.set_page_config(
    page_title="Emotion Recognition Dashboard",
    layout="wide"
)

st.title("🎭 Emotion Recognition Dashboard")

# === SIDEBAR ===
st.sidebar.header("Model Options")
model_name = st.sidebar.selectbox(
    "Select Backbone",
    options=["resnet18", "resnet50", "mobilenet_v2"],
    index=0
)
checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path",
    value=config["checkpoint_path"]
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold for Face Detection",
    0.5, 0.99, 0.6, 0.05
)

# === LOAD MODEL ===
model = load_trained_model(checkpoint_path, model_name, len(EMOTION_CLASSES))

# === TABS ===
tab1, tab2 = st.tabs(["Upload & Predict", "Model Evaluation"])

# ========================
# TAB 1: UPLOAD & PREDICT
# ========================
with tab1:
    st.subheader("Upload an Image or Video")

    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    if uploaded_file:
        file_bytes = uploaded_file.read()

        # Determine if video or image
        if uploaded_file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file_bytes)
            tfile.close()
            video_path = tfile.name

            st.video(video_path)

            # Process video
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

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
                    cv2.putText(frame, f"{pred_label} ({conf:.0%})", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

                resized_frame = cv2.resize(frame, (960,540))
                stframe.image(resized_frame, channels="BGR")

            cap.release()

        else:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)

            pred_label, probabilities = predict_on_image(image_pil, model)
            st.success(f"**Predicted Emotion: {pred_label}**")

            # Plot probabilities
            fig = px.bar(
                x=EMOTION_CLASSES,
                y=probabilities,
                labels={"x": "Emotion", "y": "Probability"},
                title="Class Probabilities"
            )
            st.plotly_chart(fig, use_container_width=True)

# ========================
# TAB 2: EVALUATION
# ========================
with tab2:
    st.subheader("Model Validation Results")

    if st.button("Run Evaluation on Validation Set"):
        st.info("Evaluating on validation set, please wait...")

        train_loader, val_loader = get_dataloaders(
            batch_size=config["batch_size"],
            input_size=config["input_size"]
        )

        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=EMOTION_CLASSES, output_dict=True)
        st.plotly_chart(plot_confusion_matrix(cm, EMOTION_CLASSES), use_container_width=True)

        st.write("### Classification Report")
        st.dataframe(report)

# ========================
# FOOTER
# ========================

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")


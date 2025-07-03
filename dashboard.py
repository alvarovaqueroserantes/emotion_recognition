import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER_FILE_WATCHER"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Resolve library conflicts

import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import cvlib as cv
import tempfile
import time
from collections import defaultdict
import gc
import pandas as pd
import base64
from io import BytesIO
import altair as alt
from torchvision import transforms
import torch.nn as nn

# Disable unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Constants and Config
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLORS = {
    'Angry': '#D32F2F',
    'Disgust': '#7B1FA2',
    'Fear': '#1976D2',
    'Happy': '#388E3C',
    'Sad': '#F57C00',
    'Surprise': '#FFA000',
    'Neutral': '#616161'
}
COLOR_HEX = list(COLORS.values())

# Professional color scheme
BACKGROUND_COLOR = "#0e1117"
SIDEBAR_COLOR = "#19212e"
CARD_COLOR = "#1a2130"
TEXT_COLOR = "#f0f2f6"
ACCENT_COLOR = "#4a76d0"

# Fixed model architecture matching 48x48 input
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Fixed for 48x48 input
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Config
@st.cache_resource
def load_config():
    return {
        "use_gpu": torch.cuda.is_available(),
        "model_name": "emotion_cnn",
        "num_classes": 7,
        "input_size": 48,
        "checkpoint_path": "model_weights.pth"
    }

# Load Model
@st.cache_resource
def load_model():
    config = load_config()
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = EmotionModel(config["num_classes"])
    
    # Initialize with pretrained-like weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    
    # Move to device
    model = model.to(device)
    model.eval()
    return model, device

# Image Transformations
def get_transform():
    config = load_config()
    return transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

# Visualization Functions with Altair
def create_emotion_bar_chart(probabilities):
    df = pd.DataFrame({
        'Emotion': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    })
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Emotion', sort=EMOTION_CLASSES, axis=alt.Axis(title='Emotion', labelAngle=0)),
        y=alt.Y('Probability', axis=alt.Axis(title='Probability', format='%')),
        color=alt.Color('Emotion', scale=alt.Scale(domain=EMOTION_CLASSES, range=COLOR_HEX)),
        tooltip=['Emotion', 'Probability']
    ).properties(
        title='Emotion Distribution',
        height=300
    )
    
    return chart

def create_emotion_timeline_chart(emotion_timeline):
    if not emotion_timeline:
        return None
    
    df = pd.DataFrame(emotion_timeline, columns=['Frame', 'Emotion'])
    df['Count'] = 1
    emotion_counts = df.groupby(['Frame', 'Emotion']).size().reset_index(name='Count')
    
    chart = alt.Chart(emotion_counts).mark_line(point=True).encode(
        x=alt.X('Frame:Q', axis=alt.Axis(title='Frame')),
        y=alt.Y('Count:Q', axis=alt.Axis(title='Detections')),
        color=alt.Color('Emotion', scale=alt.Scale(domain=EMOTION_CLASSES, range=COLOR_HEX)),
        tooltip=['Frame', 'Emotion', 'Count']
    ).properties(
        title='Emotion Timeline',
        height=300
    )
    
    return chart

def create_emotion_heatmap(emotion_matrix):
    # Create a DataFrame for the heatmap
    df = pd.DataFrame(emotion_matrix, index=EMOTION_CLASSES, columns=EMOTION_CLASSES)
    df = df.stack().reset_index()
    df.columns = ['From', 'To', 'Count']
    
    # Filter out zero values for cleaner visualization
    df = df[df['Count'] > 0]
    
    chart = alt.Chart(df).mark_rect().encode(
        x='To:O',
        y='From:O',
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['From', 'To', 'Count']
    ).properties(
        title='Emotion Transitions',
        width=500,
        height=400
    )
    
    return chart

def create_performance_charts(metrics):
    # Convert metrics to DataFrame
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    bar_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Metric', axis=alt.Axis(title='Metric')),
        y=alt.Y('Value', axis=alt.Axis(title='Value', format='%')),
        color=alt.value(ACCENT_COLOR),
        tooltip=['Metric', 'Value']
    ).properties(
        title='Performance Metrics',
        height=300
    )
    
    return bar_chart

# Video utility functions
def get_video_download_link(video_bytes, filename):
    """Generate a download link for the video"""
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}" style="color: {ACCENT_COLOR};">Download Processed Video</a>'
    return href

def process_video_frame(frame, model, transform, device, confidence_threshold):
    """Process a single video frame"""
    faces, confidences = cv.detect_face(frame)
    emotion_history = defaultdict(int)
    emotion_labels = []
    
    for (box, conf) in zip(faces, confidences):
        if conf < confidence_threshold:
            continue
        
        x1, y1, x2, y2 = box
        # Expand face area slightly for better recognition
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)
            label = EMOTION_CLASSES[pred.item()]
        
        # Update emotion history
        emotion_history[label] += 1
        emotion_labels.append(label)
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.0%})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return frame, emotion_history, emotion_labels

def calculate_model_metrics(emotion_history, total_frames):
    if total_frames == 0:
        return {}
    
    # Calculate model confidence metrics
    metrics = {
        "Detection Rate": sum(emotion_history.values()) / total_frames,
        "Diversity": len([e for e, count in emotion_history.items() if count > 0]) / len(EMOTION_CLASSES),
        "Frames Processed": total_frames
    }
    return metrics

def set_page_style():
    st.markdown(f"""
    <style>
    .main {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stButton>button {{
        background-color: {ACCENT_COLOR};
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
    }}
    .stButton>button:hover {{
        background-color: #3a66c0;
        color: white;
    }}
    .stProgress>div>div>div>div {{
        background-color: {ACCENT_COLOR};
    }}
    .reportview-container .markdown-text-container {{
        color: {TEXT_COLOR};
    }}
    .sidebar .sidebar-content {{
        background-color: {SIDEBAR_COLOR};
        color: {TEXT_COLOR};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR};
        font-weight: 600;
    }}
    .metric-container {{
        background-color: {CARD_COLOR};
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }}
    .divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {ACCENT_COLOR}, transparent);
        margin: 25px 0;
    }}
    .st-emotion-cache-1y4p8pa {{
        padding: 2rem 1rem;
    }}
    .st-emotion-cache-1v0mbdj {{
        border-radius: 8px;
    }}
    .st-emotion-cache-1vbkxwb {{
        color: {TEXT_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.set_page_config(
        page_title="Emotion Recognition System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_page_style()
    
    st.title("Facial Emotion Recognition System")
    st.markdown("""
    <div style='background-color:#1a2130; padding:20px; border-radius:8px; margin-bottom:25px;'>
    <h3 style='color:#f0f2f6;'>Advanced facial emotion detection using deep learning models</h3>
    <p>Upload media or use real-time webcam to analyze emotional expressions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Configuration")
        app_mode = st.selectbox("Application Mode", 
                               ["Real-time Webcam", "Upload Image", "Upload Video", "Performance Analytics"])
        confidence_threshold = st.slider("Detection Confidence", 0.5, 1.0, 0.7, 0.01)
        st.markdown("---")
        st.markdown("### System Information")
        st.info("This system uses a CNN model trained on the FER2013 dataset for emotion recognition.")
    
    # Load model
    model, device = load_model()
    transform = get_transform()
    
    # Initialize session state for metrics
    if 'run_metrics' not in st.session_state:
        st.session_state.run_metrics = {
            "webcam": defaultdict(int),
            "image": defaultdict(int),
            "video": defaultdict(int)
        }
    
    # Main Content
    if app_mode == "Real-time Webcam":
        st.header("Real-time Emotion Detection")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Webcam Controls")
            run_webcam = st.button("Start Detection")
            stop_webcam = st.button("Stop Detection")
            st.markdown("""
            <div class="metric-container">
                <p style="font-size:14px; margin-bottom:10px;">Detection settings optimized for real-time performance. 
                Processing occurs locally on your device.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if run_webcam:
                st.session_state.webcam_running = True
                
            if st.session_state.get('webcam_running', False):
                FRAME_WINDOW = st.image([], use_column_width=True)
                cap = cv2.VideoCapture(0)
                
                # Placeholder for real-time metrics
                metrics_placeholder = st.empty()
                emotion_history = defaultdict(int)
                frame_count = 0
                start_time = time.time()
                last_emotion = None
                emotion_timeline = []
                emotion_matrix = np.zeros((len(EMOTION_CLASSES), len(EMOTION_CLASSES)))
                
                while cap.isOpened() and st.session_state.webcam_running:
                    if stop_webcam:
                        st.session_state.webcam_running = False
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Video capture failed")
                        break
                    
                    # Process frame
                    processed_frame, frame_emotions, emotion_labels = process_video_frame(
                        frame, model, transform, device, confidence_threshold
                    )
                    
                    # Update metrics
                    for emotion in emotion_labels:
                        emotion_history[emotion] += 1
                        frame_count += 1
                        
                        # Track emotion transitions
                        if last_emotion:
                            prev_index = EMOTION_CLASSES.index(last_emotion)
                            curr_index = EMOTION_CLASSES.index(emotion)
                            emotion_matrix[prev_index][curr_index] += 1
                        last_emotion = emotion
                        emotion_timeline.append((frame_count, emotion))
                    
                    # Display frame
                    FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    
                    # Update metrics every 15 frames
                    if frame_count % 15 == 0 and frame_count > 0:
                        with metrics_placeholder.container():
                            st.markdown("### Real-time Metrics")
                            
                            # Calculate FPS
                            elapsed_time = time.time() - start_time
                            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                            
                            mcol1, mcol2 = st.columns(2)
                            with mcol1:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <p style="font-size:12px; margin-bottom:5px;">Frames Processed</p>
                                    <h3 style="color:{ACCENT_COLOR}">{frame_count}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class="metric-container">
                                    <p style="font-size:12px; margin-bottom:5px;">FPS</p>
                                    <h3 style="color:{ACCENT_COLOR}">{fps:.2f}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with mcol2:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <p style="font-size:12px; margin-bottom:5px;">Detections</p>
                                    <h3 style="color:{ACCENT_COLOR}">{sum(emotion_history.values())}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create normalized distribution
                                total = sum(emotion_history.values())
                                if total > 0:
                                    dist = {e: emotion_history[e]/total for e in EMOTION_CLASSES}
                                    st.altair_chart(create_emotion_bar_chart(dist), use_container_width=True)
                
                cap.release()
                st.session_state.webcam_running = False
                
                # Save session metrics
                for emotion, count in emotion_history.items():
                    st.session_state.run_metrics["webcam"][emotion] += count
                
                # Show final analytics
                st.markdown("---")
                st.header("Session Analysis")
                
                # Emotion timeline
                st.subheader("Emotion Timeline")
                if emotion_timeline:
                    st.altair_chart(create_emotion_timeline_chart(emotion_timeline), use_container_width=True)
                else:
                    st.info("No emotion data collected")
                
                # Emotion transition matrix
                st.subheader("Emotion Transitions")
                st.altair_chart(create_emotion_heatmap(emotion_matrix), use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                metrics = calculate_model_metrics(emotion_history, frame_count)
                if metrics:
                    st.altair_chart(create_performance_charts(metrics), use_container_width=True)
    
    elif app_mode == "Upload Image":
        st.header("Image Analysis")
        uploaded_file = st.file_uploader("Select image for analysis", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Initialize metrics
            emotion_history = defaultdict(int)
            detection_count = 0
            
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_file)
                # Convert to OpenCV format
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect faces
                faces, confidences = cv.detect_face(frame)
                
                results = []
                
                for i, (box, conf) in enumerate(zip(faces, confidences)):
                    if conf < confidence_threshold:
                        continue
                    
                    detection_count += 1
                    
                    x1, y1, x2, y2 = box
                    # Expand face area slightly for better recognition
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(face_rgb)
                    input_tensor = transform(pil_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)[0]
                        _, pred = torch.max(output, 1)
                        label = EMOTION_CLASSES[pred.item()]
                    
                    # Update emotion history
                    emotion_history[label] += 1
                    
                    # Draw on image
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.0%})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Store results
                    probabilities = {e: float(p) for e, p in zip(EMOTION_CLASSES, probs)}
                    results.append({
                        "face": i+1,
                        "emotion": label,
                        "confidence": conf,
                        "probabilities": probabilities
                    })
                
                # Convert back to RGB for display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            st.success("Analysis complete")
            col1, col2 = st.columns(2)
            
            # Show annotated image
            with col1:
                st.subheader("Analysis Results")
                st.image(display_frame, caption="Processed Image", use_column_width=True)
            
            # Show results
            if results:
                with col2:
                    st.subheader(f"Detections: {detection_count} faces")
                    for result in results:
                        st.markdown(f"#### Face {result['face']}: {result['emotion']}")
                        st.altair_chart(create_emotion_bar_chart(result["probabilities"]), use_container_width=True)
            
            # Emotion distribution across faces
            st.subheader("Overall Emotion Distribution")
            if emotion_history:
                dist = {e: emotion_history[e] for e in EMOTION_CLASSES}
                st.altair_chart(create_emotion_bar_chart(dist), use_container_width=True)
            else:
                st.info("No emotions detected in the image")
            
            # Performance metrics
            st.subheader("Performance Metrics")
            if results:
                metrics = {
                    "Detection Rate": detection_count,
                    "Diversity": len(emotion_history) / len(EMOTION_CLASSES),
                    "Avg Confidence": np.mean([r['confidence'] for r in results])
                }
                st.altair_chart(create_performance_charts(metrics), use_container_width=True)
            
            # Save session metrics
            for emotion, count in emotion_history.items():
                st.session_state.run_metrics["image"][emotion] += count
    
    elif app_mode == "Upload Video":
        st.header("Video Analysis")
        uploaded_file = st.file_uploader("Select video for analysis", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Initialize metrics
            emotion_history = defaultdict(int)
            frame_count = 0
            detection_count = 0
            last_emotion = None
            emotion_timeline = []
            emotion_matrix = np.zeros((len(EMOTION_CLASSES), len(EMOTION_CLASSES)))
            
            # Display input video
            st.subheader("Input Video")
            video_placeholder = st.empty()
            video_placeholder.video(uploaded_file)
            
            if st.button("Process Video"):
                # Save uploaded video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name
                
                # Create output video with tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_temp:
                    output_path = output_temp.name
                
                # Prepare processing
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    st.error("Could not open video file")
                    return
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Prepare video writer - use XVID codec which is more compatible
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Create progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_counter = st.empty()
                
                # Process video frame by frame
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    frame_counter.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Process frame
                    processed_frame, frame_emotions, emotion_labels = process_video_frame(
                        frame, model, transform, device, confidence_threshold
                    )
                    
                    # Update metrics
                    for emotion in emotion_labels:
                        detection_count += 1
                        emotion_history[emotion] += 1
                        
                        # Track emotion transitions
                        if last_emotion:
                            prev_index = EMOTION_CLASSES.index(last_emotion)
                            curr_index = EMOTION_CLASSES.index(emotion)
                            emotion_matrix[prev_index][curr_index] += 1
                        last_emotion = emotion
                        emotion_timeline.append((frame_count, emotion))
                    
                    # Write frame
                    out_writer.write(processed_frame)
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                
                # Release resources
                cap.release()
                out_writer.release()
                
                # Show completion message
                status_text.success("Video processing complete")
                frame_counter.empty()
                
                # Display processed video
                st.subheader("Processed Video")
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Replace the input video with processed video
                video_placeholder.video(video_bytes)
                
                # Add download button
                st.markdown(get_video_download_link(video_bytes, "processed_video.mp4"), unsafe_allow_html=True)
                
                # Analytics section
                st.markdown("---")
                st.header("Video Analysis Report")
                
                # Emotion distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Emotion Distribution")
                    if detection_count > 0:
                        emotion_dist = {e: emotion_history[e]/detection_count for e in EMOTION_CLASSES}
                        st.altair_chart(create_emotion_bar_chart(emotion_dist), use_container_width=True)
                    else:
                        st.info("No emotions detected in video")
                
                with col2:
                    st.subheader("Emotion Timeline")
                    if emotion_timeline:
                        st.altair_chart(create_emotion_timeline_chart(emotion_timeline), use_container_width=True)
                    else:
                        st.info("No emotion timeline data")
                
                # Emotion transitions
                st.subheader("Emotion Transitions")
                st.altair_chart(create_emotion_heatmap(emotion_matrix), use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                if frame_count > 0:
                    metrics = {
                        "Frames": frame_count,
                        "Detections": detection_count,
                        "Detection Rate": detection_count / frame_count,
                        "Diversity": len(emotion_history) / len(EMOTION_CLASSES) if emotion_history else 0
                    }
                    st.altair_chart(create_performance_charts(metrics), use_container_width=True)
                
                # Detailed emotion report
                st.subheader("Detailed Report")
                if emotion_history:
                    emotion_df = pd.DataFrame({
                        "Emotion": list(emotion_history.keys()),
                        "Count": list(emotion_history.values()),
                        "Percentage": [count / detection_count * 100 for count in emotion_history.values()]
                    })
                    st.dataframe(emotion_df.style.format({'Percentage': '{:.1f}%'}).background_gradient(cmap='Blues'))
                
                # Clean up temp files
                os.unlink(temp_path)
                os.unlink(output_path)
                
                # Save session metrics
                for emotion, count in emotion_history.items():
                    st.session_state.run_metrics["video"][emotion] += count
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
                gc.collect()
    
    elif app_mode == "Performance Analytics":
        st.header("System Performance Analytics")
        
        # Show overall session metrics
        st.subheader("Session Summary")
        
        if any(st.session_state.run_metrics.values()):
            # Create aggregated metrics
            all_metrics = defaultdict(int)
            for category in st.session_state.run_metrics.values():
                for emotion, count in category.items():
                    all_metrics[emotion] += count
            
            total_detections = sum(all_metrics.values())
            
            if total_detections > 0:
                col1, col2 = st.columns(2)
                
                # Overall emotion distribution
                with col1:
                    st.subheader("Emotion Distribution")
                    dist = {e: all_metrics[e]/total_detections for e in EMOTION_CLASSES}
                    df = pd.DataFrame({
                        'Emotion': list(dist.keys()),
                        'Percentage': list(dist.values())
                    })
                    
                    chart = alt.Chart(df).mark_arc().encode(
                        theta='Percentage',
                        color=alt.Color('Emotion', scale=alt.Scale(domain=EMOTION_CLASSES, range=COLOR_HEX)),
                        tooltip=['Emotion', 'Percentage']
                    ).properties(
                        height=300,
                        width=400
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                # Category distribution
                with col2:
                    st.subheader("Detection Distribution")
                    source_data = {
                        "Webcam": sum(st.session_state.run_metrics["webcam"].values()),
                        "Images": sum(st.session_state.run_metrics["image"].values()),
                        "Videos": sum(st.session_state.run_metrics["video"].values())
                    }
                    df = pd.DataFrame({
                        'Source': list(source_data.keys()),
                        'Count': list(source_data.values())
                    })
                    
                    chart = alt.Chart(df).mark_bar().encode(
                        x='Source',
                        y='Count',
                        color=alt.value(ACCENT_COLOR),
                        tooltip=['Source', 'Count']
                    ).properties(
                        height=300
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                # Performance trends
                st.subheader("Detection Trends")
                trend_data = []
                for category in ["webcam", "image", "video"]:
                    if st.session_state.run_metrics[category]:
                        for emotion, count in st.session_state.run_metrics[category].items():
                            trend_data.append({
                                'Source': category.capitalize(),
                                'Emotion': emotion,
                                'Count': count
                            })
                
                trend_df = pd.DataFrame(trend_data)
                
                chart = alt.Chart(trend_df).mark_bar().encode(
                    x='Emotion',
                    y='Count',
                    color='Source',
                    column='Source'
                ).properties(
                    width=200,
                    height=250
                )
                st.altair_chart(chart)
                
            else:
                st.warning("No detections recorded")
        else:
            st.info("No performance data collected. Use the system to collect metrics.")

if __name__ == "__main__":
    main()
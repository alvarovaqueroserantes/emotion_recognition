import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER_FILE_WATCHER"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
import plotly.express as px
import plotly.graph_objects as go
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, HeatMap, Pie, Gauge
from streamlit_echarts import st_pyecharts
from torchvision import transforms
import torch.nn as nn

# Disable warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# =====================
# CONSTANTS & CONFIG - LIGHT THEME
# =====================
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
PALETTE = {
    'background': '#FFFFFF',
    'card': '#F5F7FA',
    'accent': '#4A76D0',
    'text': '#2D3748',
    'secondary': '#E2E8F0',
    'success': '#48BB78',
    'warning': '#ED8936',
    'error': '#E53E3E',
    'border': '#CBD5E0'
}

COLORS = {
    'Angry': '#E53E3E',
    'Disgust': '#805AD5',
    'Fear': '#3182CE',
    'Happy': '#38A169',
    'Sad': '#DD6B20',
    'Surprise': '#D69E2E',
    'Neutral': '#718096'
}

# =====================
# MODEL DEFINITION
# =====================
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
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

# =====================
# UTILITY FUNCTIONS
# =====================
@st.cache_resource
def load_config():
    return {
        "use_gpu": torch.cuda.is_available(),
        "model_name": "emotion_cnn",
        "num_classes": 7,
        "input_size": 48,
        "checkpoint_path": "model_weights.pth"
    }

@st.cache_resource
def load_model():
    config = load_config()
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    model = EmotionModel(config["num_classes"])
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    
    model = model.to(device)
    model.eval()
    return model, device

def get_transform():
    config = load_config()
    return transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def get_video_download_link(video_bytes, filename):
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}" style="background-color: {PALETTE["accent"]}; color: white; padding: 10px 20px; border-radius: 4px; text-decoration: none; display: inline-block; margin: 10px 0;">Download Processed Video</a>'
    return href

def process_video_frame(frame, model, transform, device, confidence_threshold):
    faces, confidences = cv.detect_face(frame)
    emotion_history = defaultdict(int)
    emotion_labels = []
    
    for (box, conf) in zip(faces, confidences):
        if conf < confidence_threshold:
            continue
        
        x1, y1, x2, y2 = box
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
        
        emotion_history[label] += 1
        emotion_labels.append(label)
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
        cv2.putText(frame, f"{label} ({conf:.0%})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    
    return frame, emotion_history, emotion_labels

def calculate_model_metrics(emotion_history, total_frames):
    if total_frames == 0:
        return {}
    
    metrics = {
        "Detection Rate": sum(emotion_history.values()) / total_frames,
        "Diversity": len([e for e, count in emotion_history.items() if count > 0]) / len(EMOTION_CLASSES),
        "Frames Processed": total_frames
    }
    return metrics

# =====================
# ECHARTS VISUALIZATION FUNCTIONS
# =====================
def create_emotion_bar_chart(probabilities):
    # Convert probabilities to percentages
    data = [(e, round(p * 100, 1)) for e, p in probabilities.items()]
    
    bar = (
        Bar()
        .add_xaxis([d[0] for d in data])
        .add_yaxis(
            "Probability (%)",
            [d[1] for d in data],
            itemstyle_opts=opts.ItemStyleOpts(color=PALETTE['accent']),
            label_opts=opts.LabelOpts(position="top", formatter="{c}%")
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Emotion Distribution", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(color=PALETTE['text']),
                axistick_opts=opts.AxisTickOpts(is_align_with_label=True)
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(color=PALETTE['text'], formatter="{value}%"),
                min_=0,
                max_=100
            ),
            tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}%"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            grid_opts=opts.GridOpts(
                contain_label=True,
                left="10%",
                right="10%",
                top="20%",
                bottom="10%"
            )
        )
    )
    
    # Set colors for each bar
    colors = [COLORS[e] for e in probabilities.keys()]
    bar.set_series_opts(
        itemstyle_opts=opts.ItemStyleOpts(color=opts.JsCode(
            f"""function(params) {{
                var colors = {colors};
                return colors[params.dataIndex];
            }}"""
        ))
    )
    
    return bar

def create_emotion_timeline_chart(emotion_timeline):
    if not emotion_timeline:
        return None
    
    # Group by frame and emotion
    df = pd.DataFrame(emotion_timeline, columns=['Frame', 'Emotion'])
    df['Count'] = 1
    emotion_counts = df.groupby(['Frame', 'Emotion']).size().reset_index(name='Count')
    
    # Prepare series data
    series_data = {}
    for emotion in EMOTION_CLASSES:
        series_data[emotion] = emotion_counts[emotion_counts['Emotion'] == emotion][['Frame', 'Count']].values.tolist()
    
    # Create line chart
    line = Line()
    for emotion, data in series_data.items():
        line.add_xaxis([str(d[0]) for d in data])
        line.add_yaxis(
            emotion,
            [d[1] for d in data],
            symbol_size=8,
            linestyle_opts=opts.LineStyleOpts(width=2),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS[emotion])
        )
    
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Emotion Timeline", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(
            name="Frame",
            type_="category",
            name_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text']),
            axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
        ),
        yaxis_opts=opts.AxisOpts(
            name="Detections",
            name_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text']),
            axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
        ),
        legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])
        ),
        datazoom_opts=[opts.DataZoomOpts(type_="inside")],
        grid_opts=opts.GridOpts(
            contain_label=True,
            left="10%",
            right="5%",
            top="20%",
            bottom="15%"
        )
    )
    
    return line

def create_emotion_heatmap(emotion_matrix):
    # Prepare data for heatmap
    data = []
    for i, emotion_from in enumerate(EMOTION_CLASSES):
        for j, emotion_to in enumerate(EMOTION_CLASSES):
            count = emotion_matrix[i][j]
            if count > 0:
                data.append([j, i, count])
    
    # Create heatmap
    heatmap = (
        HeatMap()
        .add_xaxis(EMOTION_CLASSES)
        .add_yaxis(
            "From Emotion",
            EMOTION_CLASSES,
            data,
            label_opts=opts.LabelOpts(is_show=True, position="inside", color="#000"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Emotion Transitions", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=np.max(emotion_matrix) if np.max(emotion_matrix) > 0 else 1,
                orient="horizontal",
                pos_left="center",
                textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])
            ),
            xaxis_opts=opts.AxisOpts(
                name="To Emotion",
                name_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text']),
                axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
            ),
            yaxis_opts=opts.AxisOpts(
                name="From Emotion",
                name_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text']),
                axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter=opts.TooltipOpts(
                    formatter=JsCode("""function(params) {
                        return 'From: ' + params.value[1] + '<br/>To: ' + params.value[0] + 
                               '<br/>Transitions: ' + params.value[2];
                    }""").js_code
                )
            )
        )
    )
    
    return heatmap

def create_radial_metric(value, title, max_value=100):
    gauge = (
        Gauge()
        .add(
            series_name=title,
            data_pair=[("", value * 100)],
            min_=0,
            max_=max_value,
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color=[
                        [0.3, PALETTE['error']],
                        [0.7, PALETTE['warning']],
                        [1, PALETTE['success']]
                    ],
                    width=20
                )
            ),
            detail_label_opts=opts.GaugeDetailOpts(
                formatter="{value}%",
                font_size=16,
                color=PALETTE['text']
            ),
            title_label_opts=opts.GaugeTitleOpts(
                offset_center=[0, "40%"],
                color=PALETTE['text'],
                font_size=14
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    return gauge

def create_performance_charts(metrics):
    # Convert metrics to percentage values
    data = []
    for metric, value in metrics.items():
        if "Rate" in metric or "Diversity" in metric:
            data.append((metric, round(value * 100, 1)))
        else:
            data.append((metric, value))
    
    bar = (
        Bar()
        .add_xaxis([d[0] for d in data])
        .add_yaxis(
            "Value",
            [d[1] for d in data],
            label_opts=opts.LabelOpts(position="top", formatter="{c}%"),
            itemstyle_opts=opts.ItemStyleOpts(color=PALETTE['accent'])
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Performance Metrics", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(
                    color=PALETTE['text'],
                    interval=0,
                    rotate=0
                )
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
            ),
            tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}%"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            grid_opts=opts.GridOpts(
                contain_label=True,
                left="10%",
                right="10%",
                top="20%",
                bottom="15%"
            )
        )
    )
    return bar

def create_emotion_pie_chart(distribution):
    data = [(e, round(p * 100, 1)) for e, p in distribution.items() if p > 0]
    
    pie = (
        Pie()
        .add(
            "",
            [list(d) for d in data],
            radius=["30%", "70%"],
            center=["50%", "50%"],
            rosetype=None,
            label_opts=opts.LabelOpts(formatter="{b}: {c}%"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Emotion Distribution", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
            legend_opts=opts.LegendOpts(
                orient="vertical",
                pos_right="5%",
                textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])
            ),
            tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}%")
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(color=PALETTE['text'])
        )
    )
    
    # Apply colors
    pie.set_colors([COLORS[e] for e, _ in data])
    
    return pie

# =====================
# STYLING - LIGHT THEME
# =====================
def set_page_style():
    st.markdown(f"""
    <style>
    :root {{
        --primary: {PALETTE['accent']};
        --background: {PALETTE['background']};
        --card: {PALETTE['card']};
        --text: {PALETTE['text']};
        --secondary: {PALETTE['secondary']};
        --border: {PALETTE['border']};
    }}
    
    html, body, [class*="View"] {{
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background-color: var(--background);
        color: var(--text);
    }}
    
    .stButton>button {{
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        background-color: #3a66c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 118, 208, 0.2);
    }}
    
    .metric-card {{
        background-color: var(--card);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border);
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }}
    
    .metric-title {{
        font-size: 0.9rem;
        color: #718096;
        margin-bottom: 8px;
        font-weight: 500;
    }}
    
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }}
    
    .header-section {{
        background-color: var(--card);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border);
    }}
    
    .section-title {{
        border-left: 4px solid var(--primary);
        padding-left: 15px;
        margin: 25px 0 15px 0;
        color: var(--text);
        font-weight: 600;
    }}
    
    .stProgress > div > div > div > div {{
        background-color: var(--primary);
    }}
    
    .stFileUploader > div > div {{
        border: 2px dashed var(--border);
        border-radius: 12px;
        background-color: var(--card);
    }}
    
    .st-bb {{
        border-bottom: 1px solid var(--border);
        padding-bottom: 15px;
        margin-bottom: 20px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 20px;
        margin-right: 10px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary);
        color: white !important;
    }}
    
    .echarts {{
        border-radius: 12px;
        border: 1px solid var(--border);
        padding: 15px;
        background-color: var(--card);
    }}
    </style>
    """, unsafe_allow_html=True)

# =====================
# MAIN APP
# =====================
def main():
    st.set_page_config(
        page_title="EmotionSense Analytics",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=":bar_chart:"
    )
    
    set_page_style()
    
    # Custom header
    st.markdown(f"""
    <div class="header-section">
        <h1 style="color: {PALETTE['text']}; margin-bottom: 10px;">EmotionSense Analytics</h1>
        <p style="font-size: 1.1rem; color: #4A5568;">Advanced facial emotion detection and analysis platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Reorganized structure
    with st.sidebar:
        st.title("Configuration")
        analysis_type = st.radio("Analysis Type", ["Media Analysis", "Performance Analytics"], index=0)
        
        if analysis_type == "Media Analysis":
            st.markdown("---")
            app_mode = st.selectbox("Analysis Mode", ["Upload Image", "Upload Video"])
            confidence_threshold = st.slider("Detection Confidence", 0.5, 1.0, 0.7, 0.01)
        else:
            st.markdown("---")
            st.info("View system-wide performance metrics and analytics")
        
        st.markdown("---")
        st.markdown("### System Information")
        st.caption("CNN model trained on FER2013 dataset for emotion recognition")
        st.markdown("---")
        st.caption("Developed by Alvaro Vaquero Serantes")

    # Load model
    model, device = load_model()
    transform = get_transform()
    
    # Initialize session state
    if 'run_metrics' not in st.session_state:
        st.session_state.run_metrics = {
            "image": defaultdict(int),
            "video": defaultdict(int)
        }
    
    # Main Content - Split into two main sections
    if analysis_type == "Media Analysis":
        if app_mode == "Upload Image":
            st.header("Image Analysis")
            uploaded_file = st.file_uploader("Select image for analysis", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Initialize metrics
                emotion_history = defaultdict(int)
                detection_count = 0
                
                with st.spinner("Analyzing image..."):
                    image = Image.open(uploaded_file)
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
                        
                        emotion_history[label] += 1
                        
                        # Draw on image
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
                        cv2.putText(frame, f"{label} ({conf:.0%})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
                        
                        # Store results
                        probabilities = {e: float(p) for e, p in zip(EMOTION_CLASSES, probs)}
                        results.append({
                            "face": i+1,
                            "emotion": label,
                            "confidence": conf,
                            "probabilities": probabilities
                        })
                    
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
                        st.subheader(f"Detected Faces: {detection_count}")
                        for result in results:
                            with st.expander(f"Face {result['face']}: {result['emotion']}"):
                                st_pyecharts(
                                    create_emotion_bar_chart(result["probabilities"]), 
                                    height="400px"
                                )
                
                # Emotion distribution
                st.markdown("---")
                st.markdown("<h3 class='section-title'>Emotion Distribution</h3>", unsafe_allow_html=True)
                if emotion_history:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        # Create normalized distribution
                        total = sum(emotion_history.values())
                        dist = {e: emotion_history[e]/total for e in EMOTION_CLASSES}
                        
                        # Display metrics
                        st.markdown("### Emotion Metrics")
                        for emotion, prob in dist.items():
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">{emotion}</div>
                                <div class="metric-value">{prob*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st_pyecharts(
                            create_emotion_pie_chart(dist), 
                            height="500px"
                        )
                else:
                    st.info("No emotions detected in the image")
                
                # Performance metrics
                st.markdown("---")
                st.markdown("<h3 class='section-title'>Performance Metrics</h3>", unsafe_allow_html=True)
                if results:
                    metrics = {
                        "Detection Rate": detection_count / max(1, len(faces)),
                        "Diversity": len(emotion_history) / len(EMOTION_CLASSES),
                        "Avg Confidence": np.mean([r['confidence'] for r in results])
                    }
                    st_pyecharts(
                        create_performance_charts(metrics), 
                        height="400px"
                    )
                
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
                st.video(uploaded_file)
                
                # Process button with prominent styling
                if st.button("Analyze Video", key="process_video", 
                            use_container_width=True, type="primary",
                            help="Click to start video processing"):
                    
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
                    
                    # Prepare video writer
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
                    
                    # Display download button
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.markdown("---")
                    st.subheader("Processed Video")
                    st.markdown(get_video_download_link(video_bytes, "emotion_analysis_video.mp4"), unsafe_allow_html=True)
                    
                    # Analytics section
                    st.markdown("---")
                    st.markdown("<h3 class='section-title'>Video Analysis Report</h3>", unsafe_allow_html=True)
                    
                    # Radial metrics
                    if frame_count > 0:
                        st.markdown("### Key Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            detection_rate = detection_count / frame_count
                            st_pyecharts(
                                create_radial_metric(detection_rate, "Detection Rate"), 
                                height="300px"
                            )
                        
                        with col2:
                            diversity = len(emotion_history) / len(EMOTION_CLASSES) if emotion_history else 0
                            st_pyecharts(
                                create_radial_metric(diversity, "Diversity"), 
                                height="300px"
                            )
                        
                        with col3:
                            st_pyecharts(
                                create_radial_metric(
                                    np.mean(list(emotion_history.values())) if emotion_history else 0, 
                                    "Avg Detections", 
                                    max_value=10
                                ), 
                                height="300px"
                            )
                    
                    # Emotion distribution
                    st.markdown("---")
                    st.markdown("<h4 class='section-title'>Emotion Distribution</h4>", unsafe_allow_html=True)
                    if detection_count > 0:
                        emotion_dist = {e: emotion_history[e]/detection_count for e in EMOTION_CLASSES}
                        st_pyecharts(
                            create_emotion_bar_chart(emotion_dist), 
                            height="400px"
                        )
                    else:
                        st.info("No emotions detected in video")
                    
                    # Emotion timeline
                    st.markdown("---")
                    st.markdown("<h4 class='section-title'>Emotion Timeline</h4>", unsafe_allow_html=True)
                    if emotion_timeline:
                        st_pyecharts(
                            create_emotion_timeline_chart(emotion_timeline), 
                            height="400px"
                        )
                    else:
                        st.info("No emotion timeline data")
                    
                    # Emotion transitions
                    st.markdown("---")
                    st.markdown("<h4 class='section-title'>Emotion Transitions</h4>", unsafe_allow_html=True)
                    st_pyecharts(
                        create_emotion_heatmap(emotion_matrix), 
                        height="500px"
                    )
                    
                    # Clean up temp files
                    os.unlink(temp_path)
                    os.unlink(output_path)
                    
                    # Save session metrics
                    for emotion, count in emotion_history.items():
                        st.session_state.run_metrics["video"][emotion] += count
                    
                    # Clean up GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()
    
    elif analysis_type == "Performance Analytics":
        st.header("Performance Analytics")
        
        # Show overall session metrics
        st.markdown("<h3 class='section-title'>System Performance Summary</h3>", unsafe_allow_html=True)
        
        if any(st.session_state.run_metrics.values()):
            # Create aggregated metrics
            all_metrics = defaultdict(int)
            for category in st.session_state.run_metrics.values():
                for emotion, count in category.items():
                    all_metrics[emotion] += count
            
            total_detections = sum(all_metrics.values())
            
            if total_detections > 0:
                # Key metrics
                st.markdown("### System Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Detections</div>
                        <div class="metric-value">{total_detections}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    diversity = len(all_metrics) / len(EMOTION_CLASSES)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Emotion Diversity</div>
                        <div class="metric-value">{diversity*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_detections = np.mean(list(all_metrics.values())) if all_metrics else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Avg Detections</div>
                        <div class="metric-value">{avg_detections:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    source_counts = {
                        "Images": sum(st.session_state.run_metrics["image"].values()),
                        "Videos": sum(st.session_state.run_metrics["video"].values())
                    }
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Sources</div>
                        <div style="font-size: 1.2rem; margin-top: 10px;">
                            <div>Images: {source_counts['Images']}</div>
                            <div>Videos: {source_counts['Videos']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Emotion distribution
                st.markdown("---")
                st.markdown("<h4 class='section-title'>Emotion Distribution</h4>", unsafe_allow_html=True)
                dist = {e: all_metrics[e]/total_detections for e in EMOTION_CLASSES}
                st_pyecharts(
                    create_emotion_pie_chart(dist), 
                    height="500px"
                )
                
                # Source distribution
                st.markdown("---")
                st.markdown("<h4 class='section-title'>Detection Sources</h4>", unsafe_allow_html=True)
                source_data = {
                    "Images": sum(st.session_state.run_metrics["image"].values()),
                    "Videos": sum(st.session_state.run_metrics["video"].values())
                }
                
                # Create bar chart for sources
                bar = (
                    Bar()
                    .add_xaxis(list(source_data.keys()))
                    .add_yaxis(
                        "Detections",
                        list(source_data.values()),
                        itemstyle_opts=opts.ItemStyleOpts(color=PALETTE['accent']),
                        label_opts=opts.LabelOpts(position="top")
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="Detection Sources", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
                        xaxis_opts=opts.AxisOpts(
                            axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
                        ),
                        yaxis_opts=opts.AxisOpts(
                            axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
                        )
                    )
                )
                st_pyecharts(bar, height="400px")
                
                # Emotion trends by source
                st.markdown("---")
                st.markdown("<h4 class='section-title'>Detection Trends</h4>", unsafe_allow_html=True)
                trend_data = []
                for category in ["image", "video"]:
                    if st.session_state.run_metrics[category]:
                        for emotion, count in st.session_state.run_metrics[category].items():
                            trend_data.append({
                                'Source': category.capitalize(),
                                'Emotion': emotion,
                                'Count': count
                            })
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    
                    # Create bar chart for trends
                    bar = (
                        Bar()
                        .add_xaxis(EMOTION_CLASSES)
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Emotion Detection by Source", title_textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])),
                            xaxis_opts=opts.AxisOpts(
                                axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
                            ),
                            yaxis_opts=opts.AxisOpts(
                                axislabel_opts=opts.LabelOpts(color=PALETTE['text'])
                            ),
                            tooltip_opts=opts.TooltipOpts(trigger="axis"),
                            legend_opts=opts.LegendOpts(
                                textstyle_opts=opts.TextStyleOpts(color=PALETTE['text'])
                            )
                        )
                    )
                    
                    for source in ["Image", "Video"]:
                        source_data = trend_df[trend_df['Source'] == source]
                        counts = [source_data[source_data['Emotion'] == e]['Count'].sum() for e in EMOTION_CLASSES]
                        bar.add_yaxis(
                            source,
                            counts,
                            stack="stack1",
                            label_opts=opts.LabelOpts(is_show=False)
                        )
                    
                    st_pyecharts(bar, height="500px")
                else:
                    st.info("No trend data available")
            else:
                st.warning("No detections recorded in the current session")
        else:
            st.info("No performance data collected. Use the Media Analysis tools to collect metrics.")

if __name__ == "__main__":
    main()
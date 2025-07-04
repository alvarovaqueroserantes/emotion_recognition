# C:\Users\alvar\Documents\GitHub\emotion_recognition\streamlit\app.py

"""
EmotionSense – stakeholder-grade Streamlit front-end
• pydantic settings (& dark / corp themes)
• MediaPipe + batched inference pipeline
• Modern visuals: bullet KPIs, rolling-share line, log heat-map
• Per-asset metrics with CSV / Excel export
• Enhanced UI/UX with Streamlit Elements and custom theming
"""

import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER_FILE_WATCHER"] = "false"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import base64
import gc
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_echarts import st_pyecharts
from streamlit_elements import elements, dashboard, mui, html

from config import cfg
from styles import build_theme
from model import load_emotion_model, _infer_resnet_variant
from vision import EmotionDetector
from viz import (
    emotion_bar,
    emotion_pie,
    bullet_metric,
    rolling_share_line,
    transition_heatmap,
    metrics_to_dataframe,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    emotion_radar,
    sentiment_gauge,
    emotion_timeline,
)

# ────────── UI bootstrap ─────────────────────────────────────────────
st.set_page_config(
    page_title="EmotionSense Analytics",
    layout="wide",
    page_icon="🧠",
)
st.markdown(build_theme(cfg.palette), unsafe_allow_html=True)

# Persistent header
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #4e4376 0%, #2b5876 100%);
                padding: 1rem 2rem;
                border-radius: 0.5rem;
                color: white;
                margin-bottom: 1rem;">
        <h1 style="margin:0; font-size:1.8rem;">EmotionSense Analytics Platform</h1>
        <p style="margin:0.2rem 0 0; opacity:0.8;">Unlock deeper insights into audience sentiment and engagement with state-of-the-art AI.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ────────── detector (cached) ───────────────────────────────────────
@st.cache_resource
def init_detector() -> EmotionDetector:
    model, device = load_emotion_model(cfg)
    return EmotionDetector(model, device)

detector = init_detector()

# ────────── helpers ─────────────────────────────────────────────────
def video_download_link(data: bytes, filename: str) -> str:
    b64 = base64.b64encode(data).decode()
    return (
        f'<a href="data:video/mp4;base64,{b64}" download="{filename}" '
        f'style="background:{cfg.palette["accent"]};color:#fff;'
        f'padding:10px 20px;border-radius:4px;text-decoration:none;'
        f'display:inline-block; margin-top: 1rem; width: 100%; text-align: center;">Download Processed Video</a>'
    )

def torch_gc() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def calculate_sentiment(emotions_counts: dict) -> float:
    weights = {
        "Sad": -1.0,
        "Angry": -0.5,
        "Neutral": 0.0,
        "Surprise": 0.7,
        "Happy": 1.0,
        "Fear": -0.7,
        "Disgust": 0.5
    }
    total = sum(emotions_counts.values())
    if total == 0:
        return 0.0
    weighted_sum = sum(weights.get(e, 0) * c for e, c in emotions_counts.items())
    return weighted_sum / total

# ────────── main application layout ────────────────────────────────────────────
def main() -> None:
    st.session_state.setdefault("process_triggered", False)
    st.session_state.setdefault("download_report", False)
    st.session_state.setdefault("metrics", defaultdict(int))
    st.session_state.setdefault("all_metrics", [])
    st.session_state.setdefault("timeline", [])
    st.session_state.setdefault("last_video_name", "video")
    st.session_state.setdefault("calculated_fps", "N/A")

    with elements("dashboard"):
        layout = [
            dashboard.Item("header", 0, 0, 12, 1),
            dashboard.Item("controls", 0, 1, 3, 11),
            dashboard.Item("main_content_area", 3, 1, 9, 11),
        ]

        with dashboard.Grid(layout):
            # ────────── Header panel
            with mui.Paper(
                key="header",
                sx={"p": 2, "borderRadius": "16px", "boxShadow": "0 4px 12px rgba(0,0,0,0.05)"}
            ):
                with mui.Box(sx={"textAlign": "center"}):
                    mui.Typography(
                        "EmotionSense Analytics Platform",
                        variant="h3",
                        fontWeight="700",
                        color="primary"
                    )
                    mui.Typography(
                        "Unlock deeper insights into audience sentiment and engagement using advanced AI.",
                        variant="subtitle1",
                        color="text.secondary"
                    )

            # ────────── Controls sidebar
            with mui.Card(
                key="controls",
                sx={
                    "p": 2,
                    "borderRadius": "16px",
                    "boxShadow": "0 6px 18px rgba(0,0,0,0.1)",
                    "overflowY": "auto",
                    "maxHeight": "90vh"
                }
            ):
                with mui.Grow(in_=True, timeout=800):
                    mui.CardHeader(title="Analysis Configuration", sx={"textAlign": "center"})
                    mui.Divider()
                    analysis_mode = st.radio(
                        "Select Analysis Type",
                        ["🖼️ Image Analysis", "🎥 Video Analysis", "📊 Overall Performance"],
                        index=0,
                        key="analysis_mode_radio"
                    )
                    detector.confidence_threshold = st.slider(
                        "Detection Confidence",
                        0.5, 1.0, cfg.confidence, 0.01,
                        help="Faces detected with a probability below this threshold are ignored."
                    )
                    st.divider()
                    if st.button("Process Media", type="primary", use_container_width=True):
                        st.session_state["process_triggered"] = True
                    else:
                        st.session_state["process_triggered"] = False

                    st.divider()
                    mui.CardHeader(title="Data Export", sx={"textAlign": "center"})
                    if st.button("Prepare Data Report", use_container_width=True):
                        st.session_state["download_report"] = True

                    if st.session_state["download_report"]:
                        records = st.session_state.get("all_metrics", [])
                        if records:
                            df = metrics_to_dataframe(records)
                            excel_bytes = dataframe_to_excel_bytes(df)
                            csv_bytes = dataframe_to_csv_bytes(df)
                            st.success("✅ Data ready to download below:")
                            st.download_button(
                                label="⬇ CSV Format",
                                data=csv_bytes,
                                file_name="emotion_metrics.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            if excel_bytes:
                                st.download_button(
                                    label="⬇ Excel Format",
                                    data=excel_bytes,
                                    file_name="emotion_metrics.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            else:
                                st.info("Install `openpyxl` to enable Excel export.")
                        else:
                            st.warning("No metrics available yet — run an analysis first.")

            # ────────── Main content
            with mui.Paper(
                key="main_content_area",
                sx={"p": 2, "borderRadius": "16px", "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "overflowY": "auto"}
            ):
                if analysis_mode == "🖼️ Image Analysis":
                    image_mode_dashboard()
                elif analysis_mode == "🎥 Video Analysis":
                    video_mode_dashboard()
                else:
                    performance_mode_dashboard()

    if st.session_state["download_report"]:
        records = st.session_state.get("all_metrics", [])
        if records:
            df = metrics_to_dataframe(records)
            st.success("Report data prepared for download!")
            st.download_button(
                label="⬇ Download Full Metrics (CSV)",
                data=dataframe_to_csv_bytes(df),
                file_name="emotion_metrics.csv",
                mime="text/csv"
            )
            excel_bytes = dataframe_to_excel_bytes(df)
            if excel_bytes:
                st.download_button(
                    label="⬇ Download Full Metrics (Excel)",
                    data=excel_bytes,
                    file_name="emotion_metrics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Install `openpyxl` to enable Excel export.")
        else:
            st.info("No data available to export. Please process some media first.")
        st.session_state["download_report"] = False

# ====================================================================
# IMAGE MODE DASHBOARD
# ====================================================================
def image_mode_dashboard() -> None:
    st.header("Single Image Emotion Analysis")
    file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if file:
        if file.size > 10 * 1024 * 1024:
            st.error("Image too large (max 10 MB). Please upload a smaller image.")
            return
    else:
        st.info("Please upload an image to begin the analysis. Click 'Process Media' after uploading.")
        st.session_state["metrics_current_image"] = defaultdict(int)
        st.session_state["sentiment_current_image"] = 0.0
        return

    if st.session_state.get("process_triggered"):
        with st.spinner("Analyzing image..."):
            bgr = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
            detections = detector.detect(bgr)
            rgb_preview = cv2.cvtColor(detector.draw(bgr, detections), cv2.COLOR_BGR2RGB)
            
            # Ajustar el contenedor para mantener altura constante
            with st.container(border=True, height=450):
                st.markdown(
                    """
                    <style>
                    div[data-testid="stImage"] img {
                        width: 100% !important;
                        height: 100% !important;
                        object-fit: contain;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.image(rgb_preview, caption=f"Detected faces in {file.name}")

            if not detections:
                st.warning("No faces detected in the uploaded image.")
                st.session_state["metrics_current_image"] = defaultdict(int)
                st.session_state["sentiment_current_image"] = 0.0
                return

            history = defaultdict(int)
            for det in detections:
                history[det.label] += 1
                with st.expander(f"{det.label} ({det.confidence:.0%})"):
                    st_pyecharts(emotion_bar(dict(zip(cfg.emotion_labels, det.probabilities))), height="320px")
            
            st.session_state["metrics_current_image"] = history
            st.session_state["sentiment_current_image"] = calculate_sentiment(history)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st_pyecharts(emotion_pie(
                {e: history.get(e, 0) / sum(history.values()) if sum(history.values()) > 0 else 0
                 for e in cfg.emotion_labels}), height="350px")
        with col2:
            st_pyecharts(emotion_radar(
                {e: history.get(e, 0) / sum(history.values()) if sum(history.values()) > 0 else 0
                 for e in cfg.emotion_labels}), height="350px")
        with col3:
            st_pyecharts(sentiment_gauge(st.session_state["sentiment_current_image"]), height="350px")

        st.session_state["all_metrics"].extend(
            {"source": file.name, "emotion": e, "count": c} for e, c in history.items()
        )
        st.session_state["timeline"] = [history]

# ====================================================================
# VIDEO MODE DASHBOARD
# ====================================================================
def video_mode_dashboard() -> None:
    st.header("Video Emotion Analysis")
    file = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if file:
        if file.size > 100 * 1024 * 1024:
            st.error("Video too large (max 100 MB). Please upload a smaller video.")
            return
    else:
        st.info("Please upload a video to begin the analysis.")
        st.session_state["video_hist"] = defaultdict(int)
        st.session_state["video_timeline"] = []
        st.session_state["video_matrix"] = np.zeros((len(cfg.emotion_labels), len(cfg.emotion_labels)), dtype=int)
        return
    
    st.session_state["last_video_name"] = file.name
    with st.container(border=True, height=400):
        st.markdown(
            """
            <style>
            div[data-testid="stVideo"] {
                width: 100% !important;
                height: 100% !important;
                overflow: hidden;
                position: relative;
            }
            video {
                width: 100% !important;
                height: 100% !important;
                object-fit: contain;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.video(file)

    if st.session_state.get("process_triggered"):
        with st.status("Processing video, please wait...", expanded=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as src:
                src.write(file.getbuffer())
                src_path = Path(src.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst:
                dst_path = Path(dst.name)

            hist, timeline, matrix, calculated_fps = _process_video(src_path, dst_path)

            st.markdown(video_download_link(dst_path.read_bytes(), f"processed_{file.name}"), unsafe_allow_html=True)
            os.unlink(src_path)
            os.unlink(dst_path)

            st.session_state["video_hist"] = hist
            st.session_state["video_timeline"] = timeline
            st.session_state["video_matrix"] = matrix
            st.session_state["calculated_fps"] = calculated_fps

        _display_video_dashboard(hist, timeline, matrix)

    elif st.session_state.get("video_hist"):
        st.info("Video loaded from session. Click 'Process Media' to re-analyze.")
        _display_video_dashboard(
            st.session_state["video_hist"],
            st.session_state["video_timeline"],
            st.session_state["video_matrix"]
        )


def _process_video(src_path: Path, dst_path: Path) -> tuple[dict, list, np.ndarray, str]:
    """Internal function to handle video processing and return aggregated data."""
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        st.error("Could not open video file. Please ensure it's a valid format.")
        return defaultdict(int), [], np.zeros((len(cfg.emotion_labels), len(cfg.emotion_labels)), dtype=int), "N/A"

    fps_video = cap.get(cv2.CAP_PROP_FPS) # Original video FPS
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))

    hist = defaultdict(int)
    timeline: list[dict[str, int]] = []
    matrix = np.zeros((len(cfg.emotion_labels), len(cfg.emotion_labels)), dtype=int)
    prev_label: str | None = None
    
    inference_times = [] # To store time taken for each detection batch

    # Using st.progress within st.status for a cleaner UI
    progress_bar = st.progress(0.0)
    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        
        t_start_inference = perf_counter() # Start timing for inference
        current_detections = detector.detect(frame) # Renamed 'dets' to 'current_detections' for clarity
        t_end_inference = perf_counter() # End timing
        inference_times.append(t_end_inference - t_start_inference)

        frame_counter = defaultdict(int)
        current_frame_emotions_in_labels = [] # Collect labels for this frame that passed confidence
        
        for det in current_detections: # Iterate over the results of detector.detect
            hist[det.label] += 1
            frame_counter[det.label] += 1
            current_frame_emotions_in_labels.append(det.label)
        
        # Update transition matrix based on detected faces in the current frame
        if prev_label is not None and current_frame_emotions_in_labels:
            # We will consider transitions from the single 'prev_label' to all detected emotions in the current frame
            for current_label in current_frame_emotions_in_labels:
                i = cfg.emotion_labels.index(prev_label)
                j = cfg.emotion_labels.index(current_label)
                matrix[i, j] += 1
        
        # Set prev_label for the next iteration.
        # If faces were detected and emotions assigned, pick the most frequent emotion.
        # Otherwise, reset prev_label to None to indicate no continuous emotion.
        if current_frame_emotions_in_labels:
            prev_label = max(frame_counter, key=frame_counter.get) # Pick the most frequent emotion in current frame
        else: # No faces detected or no emotions passed confidence in this frame
            prev_label = None # Reset prev_label

        timeline.append(frame_counter)
        writer.write(detector.draw(frame, current_detections)) # Use current_detections for drawing
        progress_bar.progress((idx + 1) / total_frames)

    writer.release(); cap.release(); torch_gc()
    
    # Calculate average FPS for inference
    total_inference_duration = sum(inference_times)
    calculated_fps = round(total_frames / total_inference_duration, 1) if total_inference_duration > 0 else "N/A"

    # Store total metrics for "Overall Performance" tab
    st.session_state.setdefault("metrics", defaultdict(int)) # Aggregated for all analyses
    for k, v in hist.items():
        st.session_state.metrics[k] += v

    st.session_state.setdefault("all_metrics", []) # Raw detections for export
    st.session_state["all_metrics"].extend(
        {
            "source": st.session_state.get("last_video_name", "video"),
            "emotion": e,
            "count": c,
        }
        for e, c in hist.items()
    )
    # Store full timeline for "Overall Performance" as well
    st.session_state["timeline"] = timeline 

    return hist, timeline, matrix, calculated_fps

def _display_video_dashboard(hist, timeline, matrix):
    """Helper function to display video analysis charts."""
    total_det = sum(hist.values())
    if not total_det:
        st.warning("No faces detected in the video with the current confidence settings.")
        return

    st.markdown("---")
    st.markdown("### Video Analysis Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        dist = {e: hist[e] / total_det for e in cfg.emotion_labels}
        st_pyecharts(emotion_pie(dist), height="350px")
    with col2:
        st_pyecharts(emotion_radar(dist), height="350px")

    st_pyecharts(sentiment_gauge(calculate_sentiment(hist)), height="300px")

    st.markdown("---")
    st.markdown("### Temporal Emotion Dynamics")
    st_pyecharts(rolling_share_line(timeline, window=60), height="480px")

    st.markdown("---")
    st.markdown("### Emotion Transition Patterns")
    st_pyecharts(transition_heatmap(matrix), height="500px")

    st.markdown("---")
    st.markdown("### Key Video Performance Indicators")
    col1, col2, col3 = st.columns(3)
    
    # Calculate metrics for the current video display
    detection_rate = (total_det / len(timeline) * 100) if len(timeline) > 0 else 0
    diversity_val = round(len(hist) / len(cfg.emotion_labels) * 100, 1) if len(cfg.emotion_labels) > 0 else 0

    with col1:
        st.markdown(f'<div class="metric-card"><p class="metric-title">Detection Rate</p><p class="metric-value">{detection_rate:.1f}%</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><p class="metric-title">Diversity</p><p class="metric-value">{diversity_val:.1f}%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><p class="metric-title">Inference FPS</p><p class="metric-value">{st.session_state.get("calculated_fps", "N/A")}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Frame-by-Frame Emotion Timeline (Interactive)")
    st_pyecharts(emotion_timeline(timeline), height="500px")


# ====================================================================
#                          PERFORMANCE MODE DASHBOARD
# ====================================================================
def performance_mode_dashboard() -> None:
    st.header("Overall Performance Analytics")

    records: list[dict] = st.session_state.get("all_metrics", [])
    if not records:
        st.info("Analyze at least one image or video first to populate comprehensive performance data.")
        return

    df = metrics_to_dataframe(records)
    
    st.markdown("### Aggregated Emotion Data Table")
    st.dataframe(df, use_container_width=True, hide_index=True) # Hide DataFrame index

    st.markdown("---")
    st.markdown("### Overall Emotion Distribution Across All Analyses")
    total_overall_detections = int(df["count"].sum())
    overall_dist = df.groupby("emotion")["count"].sum().apply(lambda x: x / total_overall_detections).to_dict()
    
    col1, col2 = st.columns(2)
    with col1:
        st_pyecharts(emotion_pie(overall_dist), height="400px")
    with col2:
        st_pyecharts(emotion_radar(overall_dist), height="400px")

    st.markdown("---")
    st.markdown("### Overall Sentiment Score")
    overall_sentiment_score = calculate_sentiment(df.groupby("emotion")["count"].sum().to_dict())
    st_pyecharts(sentiment_gauge(overall_sentiment_score), height="300px")

    st.markdown("---")
    st.markdown("### Historical Emotion Timeline (Aggregated Video Data)")
    # This timeline specifically uses the 'timeline' stored from the last video analysis,
    # or it could be an aggregation of *all* timelines if stored granularly.
    # For simplicity, currently it re-uses the last video's timeline or is empty for image-only sessions.
    # If a user processes multiple videos, you might need to combine their timelines here.
    if st.session_state.get("timeline"):
        st_pyecharts(rolling_share_line(st.session_state["timeline"], window=60), height="480px")
    else:
        st.info("No video timeline data available for historical view.")


    st.markdown("---")
    st.markdown("### Model & System Information")
    st.info(f"**Model Architecture:** ResNet-{_infer_resnet_variant(detector.model.state_dict())} (auto-detected)")
    st.info(f"**Processing Device:** {detector.device.type.upper()}")
    st.info(f"**Half Precision (FP16):** {'Enabled' if cfg.half_precision else 'Disabled'} (CUDA only)")
    st.info(f"**Batch Size for Inference:** {cfg.batch_size}")
    
    # Placeholder for static model performance metrics if you have them
    st.markdown("##### Model Performance (Pre-trained Benchmarks)")
    st.write("- **Overall Accuracy (Example):** 90.5% on FER2013 dataset")
    st.write("- **Average Inference Latency (Example):** 15ms per frame")
    st.caption("These metrics are indicative of the model's general performance and are not calculated dynamically by the app.")


# ────────── run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    with ThreadPoolExecutor():
        main()
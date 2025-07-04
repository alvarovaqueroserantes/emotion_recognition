
# EmotionSense – Streamlit Front-End

🚀 An advanced Streamlit-based front-end to visualize, analyze, and monitor facial emotion recognition interactively.  

This app complements the PyTorch backend and uses its trained models to run inference over images and videos.

---

## 🌟 Features

✅ Modern corporate UI with [streamlit-elements](https://github.com/okld/streamlit-elements)  
✅ Dark/corporate themes  
✅ Video and image support  
✅ Real-time metrics:  
  - Pie chart  
  - Radar  
  - Sentiment gauge  
  - Rolling-share line  
  - Transition heatmap  
✅ CSV / Excel export  
✅ Aggregated historical performance  
✅ Responsive dashboard layout  
✅ GPU acceleration compatible

---

## 📁 Structure

```
streamlit/
├── app.py                 # Streamlit app entry
├── config.py              # Front-end config
├── styles.py              # Theming & CSS
├── viz.py                 # ECharts visualizations
└── README.md
```

---

## 🔧 Setup

From the root folder:  

```bash
cd streamlit
pip install -r requirements.txt
```

---

## 🚀 Running Streamlit App

```bash
streamlit run app.py
```

---

## 📸 How to Use

- Select “Image Analysis” to analyze static images (JPG/PNG).  
- Select “Video Analysis” to upload a video (MP4/AVI).  
- Select “Overall Performance” to view aggregated statistics.  

**Note**: In the *Overall Performance* tab, the “Process Media” button is hidden and the historical timeline plot is disabled for cleaner KPI reporting.

---

## 🧩 Visuals

- **Pie chart** → emotion distribution  
- **Radar** → emotion coverage  
- **Sentiment gauge** → global sentiment  
- **Rolling line** → temporal share  
- **Heatmap** → emotion transitions

---

## 📝 Notes

- ResNet model weights are loaded from the trained checkpoint.  
- Supports half-precision on CUDA if available.  
- Requires `fer2013.csv` in the root data directory to re-train.

---

## 🔗 Related

- Main training repo: [../README.md](../README.md)  
- Kaggle dataset: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

# Happy Streamliting! 🎈

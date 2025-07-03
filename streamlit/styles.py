# C:\Users\alvar\Documents\GitHub\emotion_recognition\streamlit\styles.py

def build_theme(palette: dict[str, str]) -> str:
    # Note: 'palette' here is expected to contain only string values, as CSS variables
    # cannot directly use lists. 'chart_colors' will be accessed directly from cfg.palette
    # in viz.py.
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --primary: {palette['accent']};
        --background: {palette['background']};
        --card: {palette['card']};
        --text: {palette['text']};
        --secondary: {palette['secondary']};
        --border: {palette['border']};
        --success: {palette['success']};
        --warning: {palette['warning']};
        --error: {palette['error']};
    }}

    html, body, [class*="View"] {{
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        scroll-behavior: smooth; /* Smooth scrolling for better UX */
    }}

    .stApp {{
        background: var(--background); /* Simpler, more minimalist background */
        background-attachment: fixed;
    }}

    .st-emotion-cache-1y4p8pa {{ /* Main content padding for the Streamlit container */
        padding: 2rem 3rem; /* Increased side padding for a more spacious feel */
    }}
    
    /* Specific target for streamlit_elements dashboard.Grid container to allow scrolling */
    .st-emotion-cache-nahz7x {{ /* This class name is specific to the rendered Streamlit HTML for dashboard.Grid */
        overflow-y: auto; /* Enable vertical scrolling if content overflows */
        height: 100%; /* Ensure it takes full height of its parent for proper scrolling */
    }}

    .card {{ /* General styling for mui.Paper elements */
        background: var(--card);
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06); /* Softer, slightly more pronounced shadow */
        padding: 1.5rem;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }}

    .card:hover {{
        box-shadow: 0 8px 25px rgba(0,0,0,0.12); /* More pronounced hover shadow */
        transform: translateY(-4px);
    }}

    .metric-card {{ /* Styling for the custom metric containers */
        background: var(--card);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04); /* Subtle shadow for metric cards */
        border: 1px solid var(--border);
        transition: all 0.2s ease-in-out;
    }}
    .metric-card:hover {{
        transform: scale(1.02); /* Slight zoom on hover */
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }}

    .metric-title {{
        color: var(--text);
        opacity: 0.6; /* Slightly lighter for a subtle feel */
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px; /* More prominent letter spacing */
        margin-bottom: 0.5rem;
        font-weight: 700; /* Bolder title */
    }}

    .metric-value {{
        color: var(--primary);
        font-size: 2.2rem; /* Slightly larger value */
        font-weight: 800; /* Extra bold for impact */
        line-height: 1.2;
    }}

    .stButton>button {{
        background: var(--primary);
        color: #fff !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease; /* Smooth transition */
        width: 100%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1); /* Softer, modern shadow */
        margin-bottom: 0.75rem; /* Spacing between buttons */
    }}

    .stButton>button:hover {{
        transform: translateY(-2px); /* Subtle lift on hover */
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        background: var(--primary); /* Keep background solid on hover */
    }}
    
    /* Styling for st.radio (Analysis Mode) */
    .stRadio > label > div[data-testid="stFlexbox"] {{
        background-color: var(--secondary);
        border-radius: 8px;
        padding: 0.75rem 1rem; /* More padding */
        margin-bottom: 0.6rem;
        border: 1px solid var(--border);
        transition: all 0.2s ease;
    }}
    .stRadio > label > div[data-testid="stFlexbox"]:hover {{
        background-color: #EBF2F3; /* Lighter hover for radio buttons */
        border-color: var(--primary); /* Accent border on hover */
    }}
    .stRadio > label > div[data-testid="stFlexbox"] > div:first-child {{
        border-color: var(--primary); /* Accent color for radio dot border */
    }}
    .stRadio > label > div[data-testid="stFlexbox"] > div:first-child > div {{
        background-color: var(--primary); /* Accent color for radio dot fill */
    }}
    .stRadio > label > div[data-testid="stFlexbox"] p {{ /* Text inside radio option */
        color: var(--text);
        font-weight: 500;
    }}


    .stSidebar {{
        background: var(--card);
        border-right: 1px solid var(--border);
        box-shadow: 2px 0 10px rgba(0,0,0,0.02); /* Subtle shadow for sidebar */
    }}

    /* Global Header Styling (for titles within mui.Paper) */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text);
        font-weight: 700;
        margin-bottom: 1rem;
        line-height: 1.2;
    }}
    h1 {{ font-size: 2.5rem; }}
    h2 {{ font-size: 2rem; }}
    h3 {{ font-size: 1.75rem; }}
    h4 {{ font-size: 1.5rem; }}

    /* Streamlit specific components that need styling adjustment */
    .st-emotion-cache-lck164 {{ /* st.expander header */
        background-color: var(--secondary);
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-bottom: 0.5rem;
        padding: 0.5rem 1rem;
    }}
    .st-emotion-cache-lck164:hover {{
        background-color: #E2E8F0;
    }}

    .st-emotion-cache-r42sel {{ /* st.expander content area */
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin-top: -0.5rem; /* Reduce gap */
    }}

    /* Streamlit Slider */
    .stSlider > div > div > div[data-testid="stSliderThumb"] {{
        background-color: var(--primary);
        border: 3px solid var(--primary);
    }}
    .stSlider > div > div > div[data-testid="stSliderTrack"] {{
        background-color: var(--secondary);
    }}
    .stSlider > div > div > div[data-testid="stSlider"] > div[role="slider"] {{
        background-color: var(--primary);
    }}

    /* Streamlit File Uploader */
    .stFileUploader > div > button {{
        background-color: var(--secondary);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stFileUploader > div > button:hover {{
        background-color: #E2E8F0;
        border-color: var(--primary);
    }}
    .stFileUploader > div > p {{
        color: var(--text);
        opacity: 0.8;
    }}

    /* Streamlit Status Box */
    .st-emotion-cache-vdpy7o {{ /* Targeting the status box background */
        background-color: var(--secondary) !important;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }}
    .st-emotion-cache-vdpy7o p {{
        color: var(--text) !important;
    }}
    .st-emotion-cache-vdpy7o .stSpinner div {{
        border-top-color: var(--primary) !important;
        border-left-color: var(--primary) !important;
    }}
    
    /* Streamlit Info/Warning/Error messages */
    div[data-testid="stAlert"] {{
        border-radius: 8px;
        padding: 1rem 1.5rem;
    }}
    div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] p {{
        font-size: 1rem;
        font-weight: 500;
    }}
    div[data-testid="stAlert"].st-emotion-cache-v06p1g {{ /* Info alert */
        background-color: rgba(0, 123, 255, 0.1); /* light primary */
        border-left-color: var(--primary) !important;
        color: var(--primary);
    }}
    div[data-testid="stAlert"].st-emotion-cache-v06p1g p {{
        color: var(--primary) !important;
    }}

    .streamlit-container {{
        max-width: 100vw;
        overflow-x: hidden;
    }}

    .metric-card {{
        background: var(--card);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }}
    .metric-title {{
        font-weight: 600;
        opacity: 0.7;
        margin-bottom: 0.5rem;
    }}
    .metric-value {{
        font-size: 1.4rem;
        font-weight: bold;
    }}

    </style>
    """
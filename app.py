import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AerialVision · RESISC45",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #090e1a;
    --bg2:       #0f1829;
    --bg3:       #162035;
    --accent:    #00e5ff;
    --accent2:   #7c3aed;
    --accent3:   #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --border:    rgba(0,229,255,0.15);
    --glow:      0 0 30px rgba(0,229,255,0.2);
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: var(--font-head) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 40%, #0a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(0,229,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 20% 80%, rgba(124,58,237,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-grid {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-head) !important;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #fff;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-title span { color: var(--accent); }
.hero-subtitle {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-desc {
    font-size: 0.88rem;
    color: #94a3b8;
    max-width: 560px;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 8px;
    margin-bottom: 1rem;
}
.badge-purple {
    background: rgba(124,58,237,0.1);
    border-color: rgba(124,58,237,0.3);
    color: #a78bfa;
}
.badge-amber {
    background: rgba(245,158,11,0.1);
    border-color: rgba(245,158,11,0.3);
    color: var(--accent3);
}

/* ── Cards ── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: var(--font-head);
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, var(--bg2), var(--bg3));
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--glow);
    animation: fadeSlide 0.5s ease-out;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.prediction-label {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    text-transform: capitalize;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.prediction-label span { color: var(--accent); }
.confidence-large {
    font-family: var(--font-mono);
    font-size: 3.5rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin: 0.5rem 0;
}
.confidence-label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    margin: 0.6rem 0;
}
.conf-bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    margin-bottom: 4px;
}
.conf-bar-name { color: var(--text); text-transform: capitalize; }
.conf-bar-pct  { color: var(--accent); font-weight: 700; }
.conf-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Stat chip ── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem; }
.stat-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    flex: 1;
    min-width: 80px;
}
.stat-chip-val {
    font-family: var(--font-head);
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
}
.stat-chip-lbl {
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 2px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,229,255,0.25) !important;
    border-radius: 12px !important;
    background: rgba(0,229,255,0.02) !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,229,255,0.5) !important;
}

/* ── Sidebar ── */
.sidebar-section {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.8rem;
}
.sidebar-title {
    font-family: var(--font-head);
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.7rem;
}
.class-pill {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    font-size: 0.65rem;
    padding: 2px 9px;
    margin: 2px 3px 2px 0;
    color: #94a3b8;
    text-transform: capitalize;
}

/* ── Divider ── */
.scan-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.5rem 0;
    opacity: 0.3;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stSelectbox > div > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_meta(model_path, meta_path, idx_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load
    model = keras_load(model_path, compile=False)
    with open(meta_path)  as f: meta      = json.load(f)
    with open(idx_path)   as f: idx2cls   = json.load(f)
    return model, meta, idx2cls


def preprocess(img: Image.Image, size=(128, 128)):
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def confidence_color(pct: float) -> str:
    if pct >= 0.75: return "#00e5ff"
    if pct >= 0.50: return "#22c55e"
    if pct >= 0.30: return "#f59e0b"
    return "#ef4444"


# ── All 45 RESISC classes ─────────────────────────────────────────────────────
ALL_CLASSES = [
    "airplane","airport","baseball_diamond","basketball_court","beach",
    "bridge","chaparral","church","circular_farmland","cloud",
    "commercial_area","dense_residential","desert","forest","freeway",
    "golf_course","ground_track_field","harbor","industrial_area","intersection",
    "island","lake","meadow","medium_residential","mobile_home_park",
    "mountain","overpass","palace","parking_lot","railway",
    "railway_station","rectangular_farmland","river","roundabout","runway",
    "sea_ice","ship","snowberg","sparse_residential","stadium",
    "storage_tank","tennis_court","terrace","thermal_power_station","wetland"
]

CLASS_ICONS = {
    "airplane":"✈️","airport":"🛬","baseball_diamond":"⚾","basketball_court":"🏀",
    "beach":"🏖️","bridge":"🌉","chaparral":"🌿","church":"⛪","circular_farmland":"🌾",
    "cloud":"☁️","commercial_area":"🏬","dense_residential":"🏘️","desert":"🏜️",
    "forest":"🌲","freeway":"🛣️","golf_course":"⛳","ground_track_field":"🏃",
    "harbor":"⚓","industrial_area":"🏭","intersection":"🚦","island":"🏝️",
    "lake":"🌊","meadow":"🌻","medium_residential":"🏠","mobile_home_park":"🏕️",
    "mountain":"⛰️","overpass":"🌁","palace":"🏯","parking_lot":"🅿️",
    "railway":"🚂","railway_station":"🚉","rectangular_farmland":"🌾","river":"🏞️",
    "roundabout":"🔄","runway":"🛫","sea_ice":"🧊","ship":"🚢","snowberg":"❄️",
    "sparse_residential":"🏡","stadium":"🏟️","storage_tank":"🛢️","tennis_court":"🎾",
    "terrace":"🌄","thermal_power_station":"⚡","wetland":"🦆"
}


# ── Global paths (computed once, before sidebar) ─────────────────────────────
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()

BASE_CNN_DIR = os.path.join(_here, "base CNN -71 % accuracy")
FINETUNE_DIR = os.path.join(BASE_CNN_DIR, "fine tuned")
META_PATH    = os.path.join(BASE_CNN_DIR, "model_metadata.json")
IDX_PATH     = os.path.join(BASE_CNN_DIR, "idx_to_class.json")
BASE_MODEL   = os.path.join(BASE_CNN_DIR, "cnn_resisc45.keras")
FT_MODEL     = os.path.join(FINETUNE_DIR, "cnn_resisc45_finetuned_best.keras")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem;'>🛰️</div>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800;
                    color:#fff; letter-spacing:-0.5px;'>AerialVision</div>
        <div style='font-size:0.6rem; letter-spacing:3px; color:#64748b;
                    text-transform:uppercase; margin-top:2px;'>RESISC-45 · CNN Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hardcoded folder structure (relative to app.py location) ──────────────
    # AERIAL SURFACE DETECTION USING CNN/   ← app.py lives here
    #   ├── app.py
    #   ├── requirements.txt
    #   └── base CNN -71 % accuracy/
    #         ├── cnn_resisc45.keras
    #         ├── idx_to_class.json
    #         ├── model_metadata.json
    #         ├── class_indices.json
    #         ├── training_history.json
    #         └── fine tuned/
    #               ├── cnn_resisc45_finetuned_best.keras
    #               ├── cnn_resisc45_finetuned.keras
    #               ├── finetune_log.csv
    #               └── training_history_finetuned.json

    # app.py directory → always correct regardless of where you run streamlit from
    # Paths come from globals defined above the sidebar block
    meta_path = META_PATH
    idx_path  = IDX_PATH

    # ── Debug expander ────────────────────────────────────────────────────────
    with st.expander("📂 Debug paths", expanded=True):
        # List actual contents of root dir so we can see exact folder names
        try:
            root_contents = os.listdir(_here)
        except Exception as e:
            root_contents = [f"ERROR: {e}"]
        try:
            base_contents = os.listdir(BASE_CNN_DIR) if os.path.exists(BASE_CNN_DIR) else ["(dir not found)"]
        except Exception as e:
            base_contents = [f"ERROR: {e}"]
        try:
            ft_contents = os.listdir(FINETUNE_DIR) if os.path.exists(FINETUNE_DIR) else ["(dir not found)"]
        except Exception as e:
            ft_contents = [f"ERROR: {e}"]

        st.text("── Root dir: " + _here)
        st.text("   Contents: " + str(root_contents))
        st.text("")
        st.text("── Base CNN dir: " + BASE_CNN_DIR)
        st.text("   Contents: " + str(base_contents))
        st.text("")
        st.text("── Fine tuned dir: " + FINETUNE_DIR)
        st.text("   Contents: " + str(ft_contents))
        st.text("")
        st.text("File checks:")
        st.text(("✔" if os.path.exists(BASE_MODEL) else "✘") + "  " + BASE_MODEL)
        st.text(("✔" if os.path.exists(FT_MODEL)   else "✘") + "  " + FT_MODEL)
        st.text(("✔" if os.path.exists(META_PATH)  else "✘") + "  " + META_PATH)
        st.text(("✔" if os.path.exists(IDX_PATH)   else "✘") + "  " + IDX_PATH)

    # ── Model selector ──
    st.markdown('<div class="sidebar-title">🔧 Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select model",
        ["Fine-tuned CNN (best)", "Base CNN"],
        label_visibility="collapsed"
    )

    if model_choice == "Fine-tuned CNN (best)":
        model_path  = FT_MODEL
        model_label = "Fine-tuned · best checkpoint"
    else:
        model_path  = BASE_MODEL
        model_label = "Base CNN · 71% accuracy"

    # ── Status indicator ──
    all_exist = {
        "model" : os.path.exists(model_path),
        "meta"  : os.path.exists(meta_path),
        "labels": os.path.exists(idx_path),
    }

    # ── Load model ──
    model_ok = all(all_exist.values())

    if model_ok:
        with st.spinner("Loading model…"):
            model, meta, idx2cls = load_model_and_meta(model_path, meta_path, idx_path)
        st.markdown(
            f'<div style="font-size:0.75rem; color:#22c55e; margin-top:6px;">'
            f'<span class="status-dot"></span>{model_label}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-size:0.75rem; color:#ef4444; margin-top:6px;">'
            '⚠️ Model files not found</div>',
            unsafe_allow_html=True
        )
        # Show which files are missing
        for label, exists in all_exist.items():
            color = "#22c55e" if exists else "#ef4444"
            icon  = "✔" if exists else "✘"
            st.markdown(
                f'<div style="font-size:0.65rem; color:{color}; font-family:monospace;">'
                f'  {icon} {label}</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

    # ── Settings ──
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K predictions", 3, 10, 5)
    show_all = st.checkbox("Show all 45 class scores", False)

    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

    # ── About ──
    st.markdown('<div class="sidebar-title">📦 Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div style="font-size:0.75rem; color:#94a3b8; line-height:1.7;">
            <b style="color:#e2e8f0;">NWPU-RESISC45</b><br>
            45 scene classes · 700 images/class<br>
            31,500 total · 256×256 px<br>
            Resized to <b style="color:#00e5ff;">128×128</b> for inference
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🗂️ All 45 Classes</div>', unsafe_allow_html=True)
    pills = "".join(
        f'<span class="class-pill">{CLASS_ICONS.get(c,"🔷")} {c.replace("_"," ")}</span>'
        for c in ALL_CLASSES
    )
    st.markdown(f'<div style="line-height:2;">{pills}</div>', unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-subtitle">// Deep Learning · Satellite Imagery</div>
    <div class="hero-title">Aerial <span>Scene</span><br>Recognition</div>
    <div style="margin-top:0.8rem;">
        <span class="badge">CNN</span>
        <span class="badge badge-purple">45 Classes</span>
        <span class="badge badge-amber">NWPU-RESISC45</span>
    </div>
    <div class="hero-desc">
        Upload any aerial or satellite image to classify it into one of 45 scene categories
        using a custom-trained Convolutional Neural Network with mixed-precision inference.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="card-title">// INPUT IMAGE</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop aerial image here",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True,
                 caption=f"📁 {uploaded.name}  ·  {img.size[0]}×{img.size[1]} px")

        # ── Image stats ──
        arr_raw = np.array(img.convert("RGB"))
        st.markdown("""<div class="scan-line"></div>""", unsafe_allow_html=True)
        st.markdown('<div class="card-title">// IMAGE STATS</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val">{img.size[0]}px</div>
                <div class="stat-chip-lbl">Width</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val">{img.size[1]}px</div>
                <div class="stat-chip-lbl">Height</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val">{img.mode}</div>
                <div class="stat-chip-lbl">Mode</div>
            </div>""", unsafe_allow_html=True)

        mean_rgb = arr_raw.reshape(-1, 3).mean(0)
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val" style="color:#ef4444;">{mean_rgb[0]:.0f}</div>
                <div class="stat-chip-lbl">Avg R</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val" style="color:#22c55e;">{mean_rgb[1]:.0f}</div>
                <div class="stat-chip-lbl">Avg G</div>
            </div>""", unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val" style="color:#60a5fa;">{mean_rgb[2]:.0f}</div>
                <div class="stat-chip-lbl">Avg B</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="border:1px dashed rgba(0,229,255,0.15); border-radius:12px;
                    height:280px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center; color:#334155;">
            <div style="font-size:3rem; margin-bottom:1rem;">🛰️</div>
            <div style="font-size:0.8rem; letter-spacing:2px; text-transform:uppercase;">
                No image uploaded
            </div>
            <div style="font-size:0.7rem; margin-top:0.5rem; color:#1e293b;">
                JPG · PNG · WEBP · TIF
            </div>
        </div>
        """, unsafe_allow_html=True)


with right:
    st.markdown('<div class="card-title">// PREDICTION OUTPUT</div>', unsafe_allow_html=True)

    if uploaded and model_ok:
        tensor = preprocess(img)

        with st.spinner("🔍 Running inference…"):
            t0   = time.perf_counter()
            pred = model.predict(tensor, verbose=0)[0]
            elapsed = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(pred)[::-1]
        best_idx    = top_indices[0]
        best_cls    = idx2cls[str(best_idx)]
        best_conf   = float(pred[best_idx])
        icon        = CLASS_ICONS.get(best_cls, "🔷")

        # ── Primary result ──
        entropy     = float(-np.sum(pred * np.log(pred + 1e-9)))
        max_entropy = float(np.log(45))
        certainty   = round((1 - entropy / max_entropy) * 100, 1)

        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; align-items:flex-start; justify-content:space-between;">
                <div>
                    <div class="confidence-label">TOP PREDICTION</div>
                    <div class="prediction-label">
                        {icon} {best_cls.replace('_',' ').title()}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="confidence-label">CONFIDENCE</div>
                    <div class="confidence-large">{best_conf*100:.1f}<span style="font-size:1.5rem; color:#64748b;">%</span></div>
                </div>
            </div>
            <div class="stat-row">
                <div class="stat-chip">
                    <div class="stat-chip-val" style="color:#f59e0b;">{elapsed:.0f}ms</div>
                    <div class="stat-chip-lbl">Inference</div>
                </div>
                <div class="stat-chip">
                    <div class="stat-chip-val" style="color:#a78bfa;">{certainty}%</div>
                    <div class="stat-chip-lbl">Certainty</div>
                </div>
                <div class="stat-chip">
                    <div class="stat-chip-val">{entropy:.2f}</div>
                    <div class="stat-chip-lbl">Entropy</div>
                </div>
                <div class="stat-chip">
                    <div class="stat-chip-val" style="color:#22c55e;">45</div>
                    <div class="stat-chip-lbl">Classes</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

        # ── Top-K bars ──
        st.markdown(
            f'<div class="card-title">// TOP {top_k} PREDICTIONS</div>',
            unsafe_allow_html=True
        )

        for rank, idx in enumerate(top_indices[:top_k]):
            cls_name = idx2cls[str(idx)]
            conf     = float(pred[idx])
            color    = confidence_color(conf)
            ico      = CLASS_ICONS.get(cls_name, "🔷")
            rank_tag = "🥇" if rank == 0 else ("🥈" if rank == 1 else ("🥉" if rank == 2 else f"#{rank+1}"))

            st.markdown(f"""
            <div class="conf-bar-wrap">
                <div class="conf-bar-header">
                    <span class="conf-bar-name">{rank_tag} {ico} {cls_name.replace('_',' ')}</span>
                    <span class="conf-bar-pct" style="color:{color};">{conf*100:.2f}%</span>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill"
                         style="width:{conf*100:.1f}%; background:linear-gradient(90deg,{color}aa,{color});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Model info strip ──
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">// MODEL INFO</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        ft_acc = meta.get("finetune_test_accuracy", meta.get("test_accuracy", 0))
        with m1:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val" style="color:#00e5ff;">{ft_acc*100:.1f}%</div>
                <div class="stat-chip-lbl">Test Acc</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            total_ep = meta.get("total_epochs", meta.get("epochs_trained", "—"))
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val">{total_ep}</div>
                <div class="stat-chip-lbl">Epochs</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-val">128px</div>
                <div class="stat-chip-lbl">Input</div>
            </div>""", unsafe_allow_html=True)

        # ── All 45 scores (optional) ──
        if show_all:
            st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">// ALL 45 CLASS SCORES</div>', unsafe_allow_html=True)
            all_sorted = [(idx2cls[str(i)], float(pred[i])) for i in np.argsort(pred)[::-1]]
            for cls_name, conf in all_sorted:
                color = confidence_color(conf)
                ico   = CLASS_ICONS.get(cls_name, "🔷")
                st.markdown(f"""
                <div class="conf-bar-wrap">
                    <div class="conf-bar-header">
                        <span class="conf-bar-name">{ico} {cls_name.replace('_',' ')}</span>
                        <span class="conf-bar-pct" style="color:{color};">{conf*100:.2f}%</span>
                    </div>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill"
                             style="width:{conf*100:.1f}%; background:linear-gradient(90deg,{color}66,{color});"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif uploaded and not model_ok:
        st.markdown(f"""
        <div class="card" style="border-color:rgba(239,68,68,0.3); padding:2rem;">
            <div style="text-align:center; margin-bottom:1.2rem;">
                <div style="font-size:2rem;">⚠️</div>
                <div style="font-family:Syne,sans-serif; font-size:1rem; font-weight:700;
                            color:#ef4444; margin-top:0.5rem;">Model Files Not Found</div>
            </div>
            <div style="font-size:0.75rem; color:#64748b; line-height:1.8; margin-bottom:1rem;">
                Make sure your project folder exactly matches this structure
                and that <code style="color:#f59e0b;">app.py</code> is in the root:
            </div>
            <pre style="font-size:0.68rem; color:#94a3b8; background:rgba(0,0,0,0.3);
                        border-radius:8px; padding:1rem; line-height:1.9;
                        border:1px solid rgba(255,255,255,0.06); white-space:pre-wrap;">AERIAL SURFACE DETECTION USING CNN/
├── app.py                       ← run from here
├── requirements.txt
└── base CNN -71 % accuracy/
    ├── cnn_resisc45.keras
    ├── idx_to_class.json
    ├── model_metadata.json
    └── fine tuned/
        └── cnn_resisc45_finetuned_best.keras</pre>
            <div style="font-size:0.7rem; color:#475569; margin-top:1rem; line-height:1.8;">
                💡 Run streamlit from the project root:<br>
                <code style="color:#f59e0b; font-size:0.65rem;">
                cd "AERIAL SURFACE DETECTION USING CNN"<br>
                streamlit run app.py</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px dashed rgba(0,229,255,0.1); border-radius:12px;
                    height:420px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center; color:#1e293b;">
            <div style="font-size:3.5rem; margin-bottom:1rem; opacity:0.4;">📊</div>
            <div style="font-size:0.8rem; letter-spacing:2px; text-transform:uppercase;
                        color:#334155;">Awaiting image input</div>
            <div style="font-size:0.7rem; margin-top:0.5rem; color:#1e293b;">
                Upload an image on the left to begin
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="scan-line"></div>
<div style="text-align:center; font-size:0.65rem; color:#1e293b; letter-spacing:2px;
            text-transform:uppercase; padding-bottom:1rem;">
    AerialVision · NWPU-RESISC45 · Custom CNN · Built with Streamlit
</div>
""", unsafe_allow_html=True)
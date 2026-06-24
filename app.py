import streamlit as st
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path
from PIL import Image
import io
import tempfile
import pandas as pd

# ─── PAGE CONFIG ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    [data-testid="stAppViewContainer"],
    .main,
    .block-container,
    [class*="css-"],
    [data-testid="stForm"] {
        background-color: #0b1220 !important;
        color: #e2e8f0 !important;
    }

    input,
    textarea,
    select,
    button,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[role="button"],
    [data-testid="stNumberInput"] input,
    [data-testid="stFileUploader"] {
        background-color: #111827 !important;
        color: #e2e8f0 !important;
        border-color: #1f2937 !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1;
    }

    .main { background-color: #0b1220; }
    .block-container { padding: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #38bdf8 !important;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .css-1n0tads,
    [data-testid="stSidebar"] .css-1q1w79c,
    [data-testid="stSidebar"] .css-1jsoq2c {
        color: #ffffff !important;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #111827 0%, #111827 100%);
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
        line-height: 1.2;
    }
    .app-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .header-icon {
        font-size: 3rem;
        line-height: 1;
    }

    /* Metric cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #38bdf8; }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #38bdf8;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    /* Section titles */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1f2937;
    }

    /* Detection result cards */
    .detection-card {
        background: #111827;
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.4rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .detection-card:hover { border-color: rgba(56, 189, 248, 0.4); }
    .detection-name {
        font-weight: 500;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .detection-conf {
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .no-detection {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #374151;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #38bdf8;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #0ea5e9 !important;
        color: #ffffff !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #38bdf8;
    }

    /* Info / warning badges */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-ok { background: #0f766e; color: #a7f3d0; }
    .badge-warn { background: #78350f; color: #fdba74; }
    .badge-crit { background: #7f1d1d; color: #fecaca; }

    /* Model comparison table */
    .model-row {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.35rem 0;
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        align-items: center;
        gap: 1rem;
    }
    .model-row.best { border-color: #0f766e; }
    .model-name { color: #e2e8f0; font-weight: 500; font-size: 0.9rem; }
    .model-stat {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        text-align: center;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0b1220; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────
CLASS_NAMES = [
    'elbow positive',
    'fingers positive',
    'forearm fracture',
    'humerus fracture',
    'humerus',
    'shoulder fracture',
    'wrist positive',
]

CLASS_COLORS = {
    'elbow positive':    '#38bdf8',
    'fingers positive':  '#22c55e',
    'forearm fracture':  '#f59e0b',
    'humerus fracture':  '#ef4444',
    'humerus':           '#a855f7',
    'shoulder fracture': '#fb7185',
    'wrist positive':    '#0ea5e9',
}

MODEL_HISTORY = [
    {"name": "v2 — YOLOv8s (50 ep)",          "mAP50": 0.280, "precision": 0.310, "recall": 0.290},
    {"name": "v3 — YOLOv8s + fixes (100 ep)",  "mAP50": 0.302, "precision": 0.373, "recall": 0.286},
]

CLASS_AP50_V3 = {
    'elbow positive':    0.200,
    'fingers positive':  0.311,
    'forearm fracture':  0.516,
    'humerus fracture':  0.000,
    'humerus':           0.510,
    'shoulder fracture': 0.240,
    'wrist positive':    0.037,
}

TRAIN_COUNTS = {
    'elbow positive':    87,
    'fingers positive':  198,
    'forearm fracture':  312,
    'humerus fracture':  203,
    'humerus':           287,
    'shoulder fracture': 156,
    'wrist positive':    74,
}

# ─── SIDEBAR ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦴 Navigare")
    page = st.radio(
        "",
        ["🔍 Detecție Fracturi", "📊 Metrici Antrenare"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("## ⚙️ Setări Model")

    conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                            help="Pragul minim de confidență pentru detecții")
    iou_thresh = st.slider("IOU threshold", 0.1, 0.9, 0.45, 0.05,
                           help="Pragul pentru Non-Maximum Suppression")

    st.markdown("---")
    st.markdown("## 📂 Model")
    model_path = st.text_input(
        "Cale fișier .pt",
        placeholder="/path/to/best.pt",
        help="Calea către modelul antrenat YOLOv8"
    )

    model_loaded = False
    model_obj = None

    if model_path and os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            model_obj = YOLO(model_path)
            model_loaded = True
            st.success("✅ Model încărcat!")
        except Exception as e:
            st.error(f"❌ Eroare: {e}")
    elif model_path:
        st.warning("⚠️ Fișierul nu a fost găsit")

    st.markdown("---")
    st.caption("YOLOv8 Bone Fracture Detection\nProiect CV · 7 clase")

# ─── HEADER ──────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-icon">🦴</div>
    <div>
        <h1>Bone Fracture Detection</h1>
        <p>Sistem de detecție a fracturilor osoase bazat pe YOLOv8 · 7 tipuri de fracturi</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: DETECȚIE
# ═══════════════════════════════════════════════════════════════════════
if page == "🔍 Detecție Fracturi":

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">📤 Încarcă Imagine Radiografie</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Trage sau selectează o imagine",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Imaginea originală", use_container_width=True)

            w, h = img.size
            st.markdown(f"""
            <div style="display:flex;gap:0.5rem;margin-top:0.5rem;flex-wrap:wrap;">
                <span class="badge badge-ok">{w}×{h}px</span>
                <span class="badge badge-ok">{uploaded.name}</span>
                <span class="badge badge-ok">{uploaded.size // 1024} KB</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#161b22;border:2px dashed #30363d;border-radius:10px;
                        padding:3rem;text-align:center;color:#8b949e;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">🩻</div>
                <div style="font-weight:500;color:#f0f6fc;margin-bottom:0.3rem;">Nicio imagine selectată</div>
                <div style="font-size:0.85rem;">Suportă JPG, PNG, BMP, WEBP</div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-title">🎯 Rezultat Detecție</div>', unsafe_allow_html=True)

        if uploaded and model_loaded and model_obj:
            with st.spinner("Analizez imaginea..."):
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    tmp_path = tmp.name

                results = model_obj(tmp_path, conf=conf_thresh, iou=iou_thresh)[0]
                annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
                os.unlink(tmp_path)

            st.image(annotated, caption="Detecții YOLOv8", use_container_width=True)

            boxes = results.boxes
            if boxes and len(boxes) > 0:
                detections = []
                for b in boxes:
                    cls_id = int(b.cls)
                    conf = float(b.conf)
                    name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
                    detections.append((name, conf))

                detections.sort(key=lambda x: -x[1])

                st.markdown(f"""
                <div style="margin:1rem 0 0.5rem 0;">
                    <span style="color:#3fb950;font-weight:600;">{len(detections)} detecții găsite</span>
                </div>
                """, unsafe_allow_html=True)

                for name, conf in detections:
                    color = CLASS_COLORS.get(name, '#58a6ff')
                    bar_w = int(conf * 100)
                    st.markdown(f"""
                    <div class="detection-card" style="border-color:{color}33;">
                        <div>
                            <div class="detection-name">{name}</div>
                            <div style="background:#21262d;border-radius:4px;height:4px;width:140px;margin-top:6px;">
                                <div style="background:{color};width:{bar_w}%;height:4px;border-radius:4px;"></div>
                            </div>
                        </div>
                        <span class="detection-conf" style="color:{color};">{conf:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Download result
                result_img = Image.fromarray(annotated)
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Descarcă imagine cu detecții",
                    buf.getvalue(),
                    file_name=f"fracture_detected_{uploaded.name}",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.markdown("""
                <div class="no-detection">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
                    <div style="color:#f0f6fc;font-weight:500;">Nicio fractură detectată</div>
                    <div style="margin-top:0.3rem;">Încearcă să scazi confidence threshold-ul din sidebar</div>
                </div>
                """, unsafe_allow_html=True)

        elif uploaded and not model_loaded:
            st.markdown("""
            <div class="no-detection">
                <div style="font-size:2rem;margin-bottom:0.5rem;">⚙️</div>
                <div style="color:#f0f6fc;font-weight:500;">Model neîncărcat</div>
                <div style="margin-top:0.3rem;">Introdu calea către fișierul <code>.pt</code> în sidebar pentru a rula inferența</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-title">📋 Clase detectabile</div>', unsafe_allow_html=True)
            for name in CLASS_NAMES:
                color = CLASS_COLORS[name]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;
                            border-bottom:1px solid #21262d;">
                    <div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                    <span style="color:#f0f6fc;font-size:0.9rem;">{name}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-detection" style="height:200px;display:flex;flex-direction:column;
                        justify-content:center;align-items:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">⬅️</div>
                <div style="color:#f0f6fc;font-weight:500;">Încarcă o imagine pentru a începe</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: METRICI ANTRENARE
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: METRICI ANTRENARE
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Metrici Antrenare":
    
    # Load metrics from results.csv
    csv_path = "runs/results .csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        st.markdown('<div class="section-title">📈 Metrici Finale — Best Model</div>', unsafe_allow_html=True)
        
        # Get last row (best metrics)
        last_row = df.iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        metrics_data = [
            (c1, f"{last_row['metrics/mAP50(B)']:.3f}", "mAP@0.5"),
            (c2, f"{last_row['metrics/precision(B)']:.3f}", "Precision"),
            (c3, f"{last_row['metrics/recall(B)']:.3f}", "Recall"),
            (c4, "7", "Clase"),
        ]
        for col, val, label in metrics_data:
            with col:
                st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Training progress chart
        st.markdown('<div class="section-title">📉 Evoluția Metricilor per Epocă</div>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#0b1220')
        
        # Loss chart
        ax = axes[0, 0]
        ax.set_facecolor('#161b22')
        ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#38bdf8', linewidth=2)
        ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#f59e0b', linewidth=2)
        ax.set_xlabel('Epoch', color='#8b949e')
        ax.set_ylabel('Loss', color='#8b949e')
        ax.set_title('Box Loss Evolution', color='#e2e8f0', fontweight='bold')
        ax.legend(loc='best', facecolor='#21262d', edgecolor='#30363d', labelcolor='#f0f6fc')
        ax.grid(True, alpha=0.1, color='#30363d')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        
        # Precision & Recall
        ax = axes[0, 1]
        ax.set_facecolor('#161b22')
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='#22c55e', linewidth=2, marker='o', markersize=4)
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='#f59e0b', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', color='#8b949e')
        ax.set_ylabel('Score', color='#8b949e')
        ax.set_title('Precision & Recall', color='#e2e8f0', fontweight='bold')
        ax.legend(loc='best', facecolor='#21262d', edgecolor='#30363d', labelcolor='#f0f6fc')
        ax.grid(True, alpha=0.1, color='#30363d')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        
        # mAP scores
        ax = axes[1, 0]
        ax.set_facecolor('#161b22')
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='#a855f7', linewidth=2, marker='D', markersize=4)
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5-0.95', color='#0ea5e9', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Epoch', color='#8b949e')
        ax.set_ylabel('mAP', color='#8b949e')
        ax.set_title('Mean Average Precision', color='#e2e8f0', fontweight='bold')
        ax.legend(loc='best', facecolor='#21262d', edgecolor='#30363d', labelcolor='#f0f6fc')
        ax.grid(True, alpha=0.1, color='#30363d')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        
        # Learning rate
        ax = axes[1, 1]
        ax.set_facecolor('#161b22')
        ax.plot(df['epoch'], df['lr/pg0'], label='Learning Rate (pg0)', color='#fb7185', linewidth=2)
        ax.set_xlabel('Epoch', color='#8b949e')
        ax.set_ylabel('LR', color='#8b949e')
        ax.set_title('Learning Rate Schedule', color='#e2e8f0', fontweight='bold')
        ax.legend(loc='best', facecolor='#21262d', edgecolor='#30363d', labelcolor='#f0f6fc')
        ax.grid(True, alpha=0.1, color='#30363d')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Detailed metrics table
        st.markdown('<div class="section-title">📋 Detalii Metrice pe Epocă</div>', unsafe_allow_html=True)
        
        display_cols = ['epoch', 'train/box_loss', 'val/box_loss', 'metrics/precision(B)', 
                       'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        display_df = df[display_cols].copy()
        display_df.columns = ['Epoch', 'Train Box Loss', 'Val Box Loss', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5-0.95']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        st.markdown('<div class="section-title">📊 Statistici Antrenare</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Epoci Antrenate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            best_map_idx = df['metrics/mAP50(B)'].idxmax()
            best_map_epoch = df.loc[best_map_idx, 'epoch']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(best_map_epoch)}</div>
                <div class="metric-label">Epoca cu Cel Mai Bun mAP</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_time = df['time'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_time/60:.1f}m</div>
                <div class="metric-label">Timp Total Antrenare</div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error(f"❌ Nu s-a găsit fișierul {csv_path}")
        st.info("Asigură-te că antrenamentul a fost efectuat și fișierul results.csv există în folderul `runs/`")
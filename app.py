"""
🌊 Disaster Damage Analysis — Streamlit Dashboard
====================================================
Main application that orchestrates the full AI pipeline:
Upload → Preprocess → Classify → Segment → Detect → Assess → Report
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time

# ── Page Configuration (must be first Streamlit call) ──────
st.set_page_config(
    page_title="🌊 Disaster Damage Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

 # ── Custom CSS for Enterprise Disaster Dashboard ─────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
  :root{--bg:#0B1120;--sidebar:#111827;--card:#1F2937;--accent:#3B82F6;--success:#22C55E;--warn:#F59E0B;--danger:#EF4444;--text:#FFFFFF}
  html,body,.stApp{background:var(--bg);color:var(--text);font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial}
  [data-testid="stSidebar"]{background:linear-gradient(180deg,var(--sidebar),#0f1724);padding:18px}
  .sidebar-logo{display:flex;align-items:center;gap:12px;margin-bottom:10px}
  .sidebar-logo img{width:48px;height:48px;border-radius:8px}
  .project-title{font-size:18px;font-weight:800}
  .muted{color:rgba(255,255,255,0.65);font-size:13px}
  .card{background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,0.03);box-shadow:0 8px 30px rgba(2,6,23,0.6)}
  .dashboard-header{display:flex;justify-content:space-between;align-items:center;padding:18px;margin-bottom:18px;border-radius:12px}
  .dash-title{font-size:22px;font-weight:800}
  .dash-sub{color:rgba(255,255,255,0.75);font-size:14px}
  .status-pill{padding:8px 12px;border-radius:999px;font-weight:700;font-size:13px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.03)}
  .metrics-row{display:flex;gap:12px;margin-bottom:16px}
  .metric{flex:1;padding:18px;border-radius:12px;background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));box-shadow:0 8px 30px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.04)}
  .metric .icon{font-size:26px;margin-bottom:8px}
  .metric .value{font-size:28px;font-weight:800}
  .metric .label{color:rgba(255,255,255,0.72);font-size:13px}
  .upload-preview{border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.03)}
  .stTabs [data-baseweb="tab-list"]{gap:8px}
  .stTabs [data-baseweb="tab"]{background:transparent;border-radius:10px;padding:8px 14px}
  footer{visibility:hidden}
    /* KPI / Dashboard styles */
    .kpi-row{display:flex;gap:14px;margin:18px 0}
    .kpi{flex:1;padding:18px;border-radius:14px;background:linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));box-shadow:0 12px 30px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.04);min-height:84px}
    .kpi .label{font-size:13px;color:rgba(255,255,255,0.85);margin-bottom:6px}
    .kpi .value{font-size:24px;font-weight:800}
    .kpi .sub{font-size:12px;color:rgba(255,255,255,0.65)}
    .badge{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:700;font-size:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.04)}
    .sidebar-nav{margin-bottom:12px}
    .nav-item{padding:10px 12px;border-radius:8px;margin-bottom:6px;color:rgba(255,255,255,0.85)}
    .nav-item:hover{background:linear-gradient(90deg, rgba(59,130,246,0.12), rgba(59,130,246,0.06));cursor:pointer}
    /* KPI color variations */
    .kpi-blue{background:linear-gradient(135deg,#0ea5e9 0%, #6366f1 100%);color:rgba(255,255,255,0.98)}
    .kpi-green{background:linear-gradient(135deg,#10b981 0%, #06b6d4 100%);color:rgba(255,255,255,0.98)}
    .kpi-purple{background:linear-gradient(135deg,#7c3aed 0%, #a78bfa 100%);color:rgba(255,255,255,0.98)}
    .kpi-orange{background:linear-gradient(135deg,#f97316 0%, #fb923c 100%);color:rgba(255,255,255,0.98)}
    /* Card headers */
    .card-header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.02);margin:-14px -14px 12px -14px}
    .card-title{font-weight:700;color:rgba(255,255,255,0.92)}
    .small-badge{background:rgba(0,0,0,0.45);padding:6px 8px;border-radius:8px;font-size:12px;border:1px solid rgba(255,255,255,0.03)}
    .image-card img{border-radius:8px;border:1px solid rgba(255,255,255,0.03)}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "uploaded_image": None,
        "preprocessed": None,
        "classification_result": None,
        "segmentation_result": None,
        "detection_result": None,
        "assessment_report": None,
        "pipeline_complete": False,
        "models_loaded": False,
        "classifier": None,
        "segmenter": None,
        "detector": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ── Model Loading (Cached) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load and cache the ResNet50 flood classifier."""
    from models.classifier import FloodClassifier
    return FloodClassifier(weights_path=None)

@st.cache_resource(show_spinner=False)
def load_segmenter():
    """Load and cache the SegFormer model."""
    from models.segmenter import FloodSegmenter
    return FloodSegmenter()

@st.cache_resource(show_spinner=False)
def load_detector():
    """Load and cache the YOLOv8 model."""
    from models.detector import ObjectDetector
    return ObjectDetector()


# ── Helper Functions ──────────────────────────────────────
def reset_pipeline():
    """Clear all pipeline results."""
    keys = [
        "preprocessed", "classification_result", "segmentation_result",
        "detection_result", "assessment_report", "pipeline_complete",
    ]
    for k in keys:
        st.session_state[k] = None
    st.session_state["pipeline_complete"] = False


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    # Sidebar header with logo
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/Emoji/objects/waving-hand.png", width=48)
    with col2:
        st.markdown("<div class='project-title'>🌊 Disaster Damage Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Emergency Response Dashboard</div>", unsafe_allow_html=True)

    # Navigation
    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    nav = st.radio("Navigation", ["Dashboard", "Upload Image", "Analysis History", "Reports", "Map View", "Alerts"], index=0, label_visibility='collapsed')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Upload card
    st.markdown("**Upload Image**")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "tiff", "bmp"], key='uploader', label_visibility='collapsed')
    if uploaded_file is not None:
        st.session_state["uploaded_image"] = uploaded_file
        st.image(uploaded_file, width='stretch', caption='Preview', output_format='PNG')
    else:
        st.markdown("<div class='card' style='text-align:center; padding:18px;'>No image selected</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Action buttons
    run_analysis = st.button("🚀 Run Analysis", type='primary', help='Run the full AI pipeline', disabled=st.session_state['uploaded_image'] is None)
    if st.button("🔄 Reset"):
        reset_pipeline()
        st.rerun()

    st.markdown("---")

    # Settings
    st.markdown("**Settings**")
    confidence_threshold = st.slider("Classification Confidence", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    yolo_confidence = st.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

    st.markdown("**Location (Optional)**")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat = st.number_input("Latitude", value=20.5937, format="%.4f", key="lat_input")
    with col_lon:
        lon = st.number_input("Longitude", value=78.9629, format="%.4f", key="lon_input")
    use_location = st.checkbox("Use GPS coordinates", value=False)

    # Status
    try:
        import torch
        gpu_status = 'GPU' if torch.cuda.is_available() else 'CPU'
    except Exception:
        gpu_status = 'Unknown'
    st.markdown(f"**Compute:** {gpu_status}  ")
    models_loaded = st.session_state.get('models_loaded', False)
    st.markdown(f"**Models Loaded:** {'Yes' if models_loaded else 'No'}")


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class='dashboard-header card'>
    <div style='display:flex;flex-direction:column'>
        <div class='dash-title'>🌊 Disaster Damage Assessment</div>
        <div class='dash-sub'>AI-driven flood detection, segmentation, and damage assessment</div>
    </div>
    <div style='text-align:right'>
        <div class='muted'>""" + time.strftime('%Y-%m-%d %H:%M') + """</div>
        <div style='height:8px'></div>
        <div class='status-pill'>System: Online &nbsp;•&nbsp; Models: """ + ("Loaded" if st.session_state.get('models_loaded') else "Not Loaded") + """</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ─────────────────────────────────────────
# KPI row (large cards)
col_a, col_b, col_c, col_d = st.columns(4)
# safe accessors
classification = st.session_state.get('classification_result') or {}
segmentation = st.session_state.get('segmentation_result') or {}
assessment = st.session_state.get('assessment_report') or {}

with col_a:
    flood_status = 'FLOOD' if classification.get('is_flood') else ('NO' if classification.get('is_flood') is False else 'NO DATA')
    st.markdown(f"<div class='kpi'><div class='label'>Flood Detected</div><div class='value'>{flood_status}</div><div class='sub'>Flood presence confirmed</div></div>", unsafe_allow_html=True)
with col_b:
    conf = classification.get('confidence')
    conf_text = f"{conf:.1%}" if isinstance(conf, (int,float)) else "—"
    st.markdown(f"<div class='kpi'><div class='label'>Confidence Score</div><div class='value'>{conf_text}</div><div class='sub'>Model confidence</div></div>", unsafe_allow_html=True)
with col_c:
    fa = segmentation.get('flood_area_ratio')
    fa_text = f"{fa:.1%}" if isinstance(fa, (int,float)) else "—"
    st.markdown(f"<div class='kpi'><div class='label'>Flood Area</div><div class='value'>{fa_text}</div><div class='sub'>Total area affected</div></div>", unsafe_allow_html=True)
with col_d:
    sev = assessment.get('severity_score')
    sev_text = f"{sev}" if sev is not None else "—"
    st.markdown(f"<div class='kpi'><div class='label'>Severity Level</div><div class='value'>{sev_text}</div><div class='sub'>Damage severity</div></div>", unsafe_allow_html=True)


# ── Main Pipeline Execution ───────────────────────────────
if run_analysis and st.session_state["uploaded_image"] is not None:
    reset_pipeline()

    # Update config thresholds
    import config as cfg
    cfg.CLASSIFIER_CONFIDENCE_THRESHOLD = confidence_threshold
    cfg.YOLO_CONFIDENCE_THRESHOLD = yolo_confidence

    uploaded = st.session_state["uploaded_image"]
    location = {"lat": lat, "lon": lon} if use_location else None

    progress = st.progress(0, text="Initializing pipeline...")

    # ── Step 1: Load Models ────────────────────────────
    progress.progress(5, text="🔧 Loading AI models...")

    with st.spinner("Loading ResNet50 classifier..."):
        classifier = load_classifier()
    with st.spinner("Loading SegFormer segmenter..."):
        segmenter = load_segmenter()
    with st.spinner("Loading YOLOv8 detector..."):
        detector = load_detector()

    progress.progress(15, text="✅ Models loaded!")
    time.sleep(0.3)

    # ── Step 2: Preprocess ─────────────────────────────
    progress.progress(20, text="🔧 Preprocessing image...")

    from preprocessing.image_processor import ImageProcessor
    processor = ImageProcessor()
    raw_image = processor.load_image(uploaded)
    preprocessed = processor.preprocess(raw_image)
    st.session_state["preprocessed"] = preprocessed

    # Try to extract GPS
    if not location:
        uploaded.seek(0)
        exif_gps = processor.extract_exif_gps(uploaded)
        if exif_gps:
            location = exif_gps
            st.toast(f"📍 GPS extracted: {exif_gps['lat']:.4f}, {exif_gps['lon']:.4f}")

    progress.progress(30, text="✅ Image preprocessed!")
    time.sleep(0.2)

    # ── Step 3: Classification ─────────────────────────
    progress.progress(35, text="🧠 Classifying image (ResNet50)...")

    classifier.threshold = confidence_threshold
    classification = classifier.predict(
        preprocessed["tensor"]
    )
    # Since ResNet is pure ImageNet, it won't reliably set is_flood.
    # We set it to True here to let SegFormer run, and SegFormer will be the source of truth.
    classification["is_flood"] = True 
    st.session_state["classification_result"] = classification

    progress.progress(45, text=f"✅ Classification: {classification['label'].upper()}")
    time.sleep(0.3)

    # ── Step 4: Segmentation (only if flood detected) ──
    if classification["is_flood"]:
        progress.progress(50, text="🎭 Running flood segmentation (SegFormer)...")

        segmentation = segmenter.segment(preprocessed["pil"])
        st.session_state["segmentation_result"] = segmentation

        progress.progress(65, text=f"✅ Segmentation complete! Flood area: {segmentation['flood_area_ratio']:.1%}")
        time.sleep(0.2)

        # ── Step 5: Object Detection ───────────────────
        progress.progress(70, text="🔍 Detecting objects (YOLOv8)...")

        detection = detector.detect(
            preprocessed["display"],
            flood_mask=segmentation["flood_mask"]
        )
        st.session_state["detection_result"] = detection

        progress.progress(85, text=f"✅ Detected {detection['summary']['total_objects']} objects!")
        time.sleep(0.2)

        # ── Step 6: Damage Assessment ──────────────────
        progress.progress(90, text="📊 Computing damage assessment...")

        from engine.damage_assessment import DamageAssessmentEngine
        engine = DamageAssessmentEngine()
        assessment = engine.assess(classification, segmentation, detection, location)
        st.session_state["assessment_report"] = assessment

        progress.progress(100, text="✅ Analysis complete!")
        st.session_state["pipeline_complete"] = True
    else:
        # No flood — skip segmentation + detection
        progress.progress(100, text="✅ No flood detected — analysis complete!")
        st.session_state["pipeline_complete"] = True

    time.sleep(0.5)
    st.rerun()


# ── Display Results ───────────────────────────────────────
if st.session_state["uploaded_image"] is None:
    # Landing state
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; opacity: 0.7;">
        <h2 style="font-size: 3rem; margin-bottom: 10px;">📸</h2>
        <h3>Upload an image to begin analysis</h3>
        <p style="color: rgba(255,255,255,0.5); max-width: 500px; margin: 10px auto;">
            Upload a flood or disaster image using the sidebar.
            The AI pipeline will automatically classify, segment,
            detect objects, and assess damage.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state["pipeline_complete"]:
    classification = st.session_state["classification_result"]
    segmentation = st.session_state["segmentation_result"]
    detection = st.session_state["detection_result"]
    assessment = st.session_state["assessment_report"]
    preprocessed = st.session_state["preprocessed"]

    # ── Tabs ───────────────────────────────────────────
    if classification and not classification["is_flood"]:
        # No flood — simple result
        tab1, = st.tabs(["📸 Classification Result"])

        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(preprocessed["display"], caption="Uploaded Image", width='stretch')
            with col2:
                st.markdown("### ✅ No Flood Detected")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">No Flood</div>
                    <div class="label">Classification Result</div>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Confidence", f"{classification['confidence']:.1%}")
                if "probabilities" in classification:
                    st.metric("Flood Probability", f"{classification['probabilities'].get('flood', 0):.1%}")
                if "imagenet_class" in classification and classification["imagenet_class"]:
                    st.metric("Detected Class (ResNet)", classification["imagenet_class"].title())

                st.info("No flood was detected in this image. The pipeline stops here.")
    else:
        # Full flood analysis
        # Preview row: Uploaded Image | Flood Segmentation | Object Detection
        try:
            col_u, col_s, col_d = st.columns([1, 1, 1])
            with col_u:
                st.markdown("<div class='card'><div class='card-header'><div class='card-title'>Uploaded Image</div><div class='small-badge'>Image Loaded</div></div>", unsafe_allow_html=True)
                try:
                    img_name = st.session_state['uploaded_image'].name
                except Exception:
                    img_name = ''
                    st.image(preprocessed['display'], width='stretch', caption=img_name)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_s:
                st.markdown("<div class='card'><div class='card-header'><div class='card-title'>Flood Segmentation</div><div class='small-badge'>Overlay</div></div>", unsafe_allow_html=True)
                from utils.visualization import create_flood_overlay
                if segmentation:
                    overlay_img = create_flood_overlay(preprocessed['display'], segmentation['flood_mask'])
                    seg_pct = segmentation.get('flood_area_ratio')
                    badge_text = f"{seg_pct:.1%}" if isinstance(seg_pct, (int,float)) else ""
                    st.image(overlay_img, width='stretch', caption=("Segmentation Confidence: " + badge_text) if badge_text else None)
                else:
                    st.markdown("<div class='image-card card' style='padding:18px;text-align:center;'>No segmentation available</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_d:
                st.markdown("<div class='card'><div class='card-header'><div class='card-title'>Object Detection</div><div class='small-badge'>Detection</div></div>", unsafe_allow_html=True)
                if detection:
                    ann = detection.get('annotated_image')
                    total = detection.get('summary', {}).get('total_objects', 0)
                    st.image(ann, width='stretch', caption=f"{total} objects detected")
                else:
                    st.markdown("<div class='image-card card' style='padding:18px;text-align:center;'>No detections</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            # Keep UI robust: if preview fails, continue to tabs
            pass
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📸 Classification",
            "💧 Segmentation",
            "🔍 Detection",
            "📊 Assessment",
            "🗺️ Map",
            "📄 Report",
        ])

        # ── Tab 1: Classification ──────────────────────
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(preprocessed["display"], caption="Uploaded Image", width='stretch')
            with col2:
                st.markdown("### 🌊 Flood Detected!")

                # Confidence gauge
                from utils.visualization import generate_severity_gauge
                if assessment:
                    gauge_bytes = generate_severity_gauge(
                        assessment["severity_score"], assessment["damage_category"]
                    )
                    if gauge_bytes:
                        st.image(gauge_bytes, width='stretch')

                st.metric("Classification", "🌊 FLOOD DETECTED")
                st.metric("Confidence", f"{classification['confidence']:.1%}")

                if "imagenet_class" in classification and classification["imagenet_class"]:
                    st.metric("Detected Class (ResNet)", classification["imagenet_class"].title())

        # ── Tab 2: Segmentation ────────────────────────
        with tab2:
            if segmentation:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### Original Image")
                    st.image(preprocessed["display"], width='stretch')

                with col2:
                    st.markdown("#### Flood Mask Overlay")
                    from utils.visualization import create_flood_overlay
                    overlay = create_flood_overlay(
                        preprocessed["display"], segmentation["flood_mask"]
                    )
                    st.image(overlay, width='stretch')

                # Metrics
                st.divider()
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">{segmentation['flood_area_ratio']:.1%}</div>
                        <div class="label">Flood Area Coverage</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol2:
                    total_px = segmentation['flood_mask'].shape[0] * segmentation['flood_mask'].shape[1]
                    flood_px = np.count_nonzero(segmentation['flood_mask'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">{flood_px:,}</div>
                        <div class="label">Flood Pixels</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">{segmentation['num_labels']}</div>
                        <div class="label">Segmentation Classes</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 3: Object Detection ────────────────────
        with tab3:
            if detection:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### Detected Objects")
                    # Convert BGR annotated image to RGB for display
                    annotated = detection["annotated_image"]
                    if annotated is not None:
                        st.image(annotated, width='stretch')

                with col2:
                    st.markdown("#### Detection Summary")
                    summary = detection["summary"]

                    st.metric("Total Objects", summary["total_objects"])
                    st.metric("In Flood Zone", summary["objects_in_flood"])

                    if summary["by_class"]:
                        st.markdown("##### Objects by Class")
                        from utils.visualization import generate_objects_chart
                        chart_bytes = generate_objects_chart(summary["by_class"])
                        if chart_bytes:
                            st.image(chart_bytes, width='stretch')

                # Detection table
                if detection["detections"]:
                    st.divider()
                    st.markdown("#### Detection Details")
                    import pandas as pd
                    det_data = []
                    for i, d in enumerate(detection["detections"]):
                        det_data.append({
                            "#": i + 1,
                            "Class": d["class"].title(),
                            "Confidence": f"{d['confidence']:.0%}",
                            "In Flood Zone": "⚠️ Yes" if d["in_flood_zone"] else "✅ No",
                            "BBox": f"({d['bbox'][0]}, {d['bbox'][1]}) → ({d['bbox'][2]}, {d['bbox'][3]})",
                        })
                    st.dataframe(pd.DataFrame(det_data), width='stretch', hide_index=True)

        # ── Tab 4: Assessment ──────────────────────────
        with tab4:
            if assessment:
                # Severity badge
                category = assessment["damage_category"]
                severity = assessment["severity_score"]
                emoji = assessment["damage_category_emoji"]
                css_class = f"severity-{category.lower()}"

                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <span class="severity-badge {css_class}">
                        {emoji} {category} — Score: {severity}/10
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.divider()

                # Key metrics row
                mcols = st.columns(4)
                metrics = [
                    ("Flood Coverage", f"{assessment['flood_area_percentage']:.1f}%"),
                    ("Objects Detected", str(assessment["total_objects_detected"])),
                    ("In Flood Zone", str(assessment["objects_in_flood_zone"])),
                    ("Severity Score", f"{severity}/10"),
                ]
                for mcol, (label, value) in zip(mcols, metrics):
                    with mcol:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="value">{value}</div>
                            <div class="label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.divider()

                # Score breakdown
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### Score Breakdown")
                    from utils.visualization import generate_score_breakdown_chart
                    chart = generate_score_breakdown_chart(assessment["score_breakdown"])
                    if chart:
                        st.image(chart, width='stretch')

                with col2:
                    st.markdown("#### Breakdown Details")
                    bd = assessment["score_breakdown"]
                    import pandas as pd
                    bd_data = [
                        {
                            "Component": "Flood Area",
                            "Raw": f"{bd['flood_area']['raw']}%",
                            "Score": f"{bd['flood_area']['score']}/10",
                            "Weight": f"{bd['flood_area']['weight']:.0%}",
                            "Contribution": bd['flood_area']['contribution'],
                        },
                        {
                            "Component": "Structures",
                            "Raw": str(bd['affected_structures']['count']),
                            "Score": f"{bd['affected_structures']['score']}/10",
                            "Weight": f"{bd['affected_structures']['weight']:.0%}",
                            "Contribution": bd['affected_structures']['contribution'],
                        },
                        {
                            "Component": "Object Types",
                            "Raw": "—",
                            "Score": f"{bd['object_types']['score']}/10",
                            "Weight": f"{bd['object_types']['weight']:.0%}",
                            "Contribution": bd['object_types']['contribution'],
                        },
                    ]
                    st.dataframe(pd.DataFrame(bd_data), width='stretch', hide_index=True)

                # Recommendations
                st.divider()
                st.markdown("#### 📋 Recommendations")
                recs = assessment.get("recommendations", [])
                for rec in recs:
                    priority = rec["priority"].lower()
                    st.markdown(f"""
                    <div class="rec-card rec-{priority}">
                        <strong>[{rec['priority']}]</strong> {rec['action']}
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 5: Map ─────────────────────────────────
        with tab5:
            location = {"lat": lat, "lon": lon} if use_location else None

            if location:
                from reporting.map_generator import MapGenerator
                from streamlit_folium import st_folium

                map_gen = MapGenerator()
                folium_map = map_gen.generate_map(
                    location=location,
                    flood_mask=segmentation["flood_mask"] if segmentation else None,
                    detections=detection["detections"] if detection else [],
                    assessment_report=assessment,
                )
                st_folium(folium_map, width='stretch', height=500)
            else:
                st.info(
                    "📍 Enable GPS coordinates in the sidebar to view the interactive map. "
                    "You can enter the latitude/longitude of the disaster location."
                )

        # ── Tab 6: PDF Report ──────────────────────────
        with tab6:
            st.markdown("#### 📄 Generate PDF Report")
            st.markdown("Download a comprehensive PDF report of the analysis results.")

            if st.button("📥 Generate PDF Report", type="primary", width='stretch'):
                with st.spinner("Generating PDF report..."):
                    from reporting.pdf_report import PDFReportGenerator
                    from utils.visualization import create_flood_overlay

                    pdf_gen = PDFReportGenerator()

                    # Prepare images
                    seg_overlay = None
                    det_image = None

                    if segmentation:
                        seg_overlay = create_flood_overlay(
                            preprocessed["display"], segmentation["flood_mask"]
                        )

                    if detection:
                        det_image = detection["annotated_image"]

                    pdf_bytes = pdf_gen.generate(
                        assessment_report=assessment,
                        original_image=preprocessed["display"],
                        segmentation_overlay=seg_overlay,
                        detection_image=det_image,
                        flood_mask=segmentation["flood_mask"] if segmentation else None,
                    )

                    st.download_button(
                        label="⬇️ Download Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"disaster_damage_report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )

                st.success("✅ PDF report generated successfully!")

elif st.session_state["uploaded_image"] is not None and not st.session_state["pipeline_complete"]:
    # Image uploaded but not analyzed yet
    pil_image = Image.open(st.session_state["uploaded_image"])
    st.session_state["uploaded_image"].seek(0)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_image, caption="Uploaded Image — Ready for Analysis", width='stretch')
    with col2:
        st.markdown("""
        ### 🔄 Ready to Analyze

        Your image has been uploaded. Click **🚀 Run Full Analysis** in the sidebar to start the AI pipeline.

        **Pipeline stages:**
        1. 🔧 OpenCV Preprocessing
        2. 🧠 ResNet50 Flood Classification
        3. 🎭 SegFormer Flood Segmentation
        4. 🔍 YOLOv8 Object Detection
        5. 📊 Damage Assessment
        6. 📄 Report Generation
        """)


# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.4; font-size: 0.8rem; padding: 10px 0;">
    🌊 Disaster Damage Analysis System — Powered by ResNet50 · SegFormer · YOLOv8<br>
    Built with Streamlit · PyTorch · HuggingFace Transformers · Ultralytics
</div>
""", unsafe_allow_html=True)

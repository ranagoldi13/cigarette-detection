import os
import cv2
import tempfile
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smoking Detection in Restricted Areas",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Dark Glassmorphism CSS
st.markdown(
    """
    <style>
    /* Global Styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Custom Metric Badges */
    .stMetric {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    
    /* Alert Banners */
    .alert-danger-box {
        background-color: rgba(225, 29, 72, 0.2);
        border: 1px solid #f43f5e;
        color: #fecdd3;
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .alert-safe-box {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #a7f3d0;
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* Sidebar styling */
    .css-1d3b8ab, [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PALETTE (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_GREEN = (50, 205, 50)     # Safe Person (Green)
C_RED = (30, 30, 220)       # Smoking Person (Red)
C_ORANGE = (0, 165, 255)    # Cigarette (Orange)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_YELLOW = (0, 215, 255)
C_DARK_BG = (15, 15, 15)
C_ACCENT = (0, 120, 255)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO Neural Networks...")
def load_models(person_model_path="yolo11s.pt", cigarette_model_path="best.pt"):
    """Load both Person detection model and custom Cigarette detection model."""
    person_model = YOLO(person_model_path)
    
    if not os.path.exists(cigarette_model_path):
        st.error(f"Custom model weight `{cigarette_model_path}` not found in root directory!")
        st.stop()
        
    cig_model = YOLO(cigarette_model_path)
    return person_model, cig_model

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def containment_ratio(small_box, large_box):
    """Compute what fraction of the small_box (cigarette) lies inside large_box (person)."""
    xA = max(small_box[0], large_box[0])
    yA = max(small_box[1], large_box[1])
    xB = min(small_box[2], large_box[2])
    yB = min(small_box[3], large_box[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
        
    small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1]) + 1e-6
    return inter_area / small_area


def box_iou(boxA, boxB):
    """Compute standard Intersection over Union (IoU)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.45):
    """Blend a filled transparent rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud(frame, total_persons, smoking_count, safe_count, frame_w, frame_h, show_hud=True, show_banner=True):
    """Draw live statistics panel on top right and violation alert banner at bottom."""
    if show_hud:
        px1 = frame_w - 230
        py1 = 15
        px2 = frame_w - 10
        py2 = py1 + 130

        alpha_rect(frame, px1, py1, px2, py2, C_DARK_BG, alpha=0.75)
        cv2.rectangle(frame, (px1, py1), (px2, py2), C_ACCENT, 1)

        cv2.putText(frame, "LIVE SURVEILLANCE", (px1 + 12, py1 + 24), FONT, 0.50, C_YELLOW, 1, cv2.LINE_AA)
        cv2.line(frame, (px1 + 8, py1 + 30), (px2 - 8, py1 + 30), C_ACCENT, 1)

        cv2.putText(frame, f"Total Persons : {total_persons}", (px1 + 12, py1 + 54), FONT_SM, 0.45, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"SMOKING       : {smoking_count}", (px1 + 12, py1 + 78), FONT_SM, 0.45, C_RED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"SAFE          : {safe_count}", (px1 + 12, py1 + 102), FONT_SM, 0.45, C_GREEN, 1, cv2.LINE_AA)

    if show_banner:
        banner_h = max(35, int(frame_h * 0.07))
        by1 = frame_h - banner_h
        cv2.rectangle(frame, (0, by1), (frame_w, frame_h), C_BLACK, -1)

        if smoking_count > 0:
            alert_txt = "🚨 RESTRICTED AREA VIOLATION: SMOKER DETECTED"
            scale = max(0.6, frame_w / 1200.0)
            thick = 2
            (tw, th), _ = cv2.getTextSize(alert_txt, FONT, scale, thick)
            tx = (frame_w - tw) // 2
            ty = by1 + (banner_h + th) // 2
            cv2.putText(frame, alert_txt, (tx, ty), FONT, scale, C_RED, thick, cv2.LINE_AA)
        else:
            safe_txt = "ZONE CLEAR — NO VIOLATIONS DETECTED"
            scale = max(0.5, frame_w / 1400.0)
            thick = 1
            (tw, th), _ = cv2.getTextSize(safe_txt, FONT_SM, scale, thick)
            tx = (frame_w - tw) // 2
            ty = by1 + (banner_h + th) // 2
            cv2.putText(frame, safe_txt, (tx, ty), FONT_SM, scale, C_GREEN, thick, cv2.LINE_AA)


def process_image(frame, person_model, cig_model, person_conf, cig_conf, iou_thresh, overlap_thresh, show_hud=True, show_banner=True):
    """Run dual-model detection pipeline on a single image frame."""
    frame_draw = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Predict Persons
    person_results = person_model.predict(
        frame, classes=[0], conf=person_conf, iou=iou_thresh, verbose=False
    )
    
    # Predict Cigarettes
    cig_results = cig_model.predict(
        frame, conf=cig_conf, iou=iou_thresh, verbose=False
    )

    persons = []
    if person_results and person_results[0].boxes is not None:
        boxes = person_results[0].boxes
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for idx, (xyxy, conf) in enumerate(zip(xyxys, confs)):
            persons.append({"box": xyxy, "conf": conf, "id": idx + 1})

    cigarettes = []
    if cig_results and cig_results[0].boxes is not None:
        boxes = cig_results[0].boxes
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for xyxy, conf in zip(xyxys, confs):
            cigarettes.append({"box": xyxy, "conf": conf})

    # Association Check
    smoking_person_ids = set()
    for cig in cigarettes:
        for p in persons:
            c_ratio = containment_ratio(cig["box"], p["box"])
            i_ratio = box_iou(cig["box"], p["box"])
            if c_ratio >= overlap_thresh or i_ratio > 0.05:
                smoking_person_ids.add(p["id"])

    # Draw Cigarette Boxes
    for cig in cigarettes:
        x1, y1, x2, y2 = [int(v) for v in cig["box"]]
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), C_ORANGE, 2)
        label = f"Cigarette {cig['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.45, 1)
        ly1 = max(0, y1 - th - 4)
        cv2.rectangle(frame_draw, (x1, ly1), (x1 + tw + 6, y1), C_ORANGE, -1)
        cv2.putText(frame_draw, label, (x1 + 3, y1 - 3), FONT_SM, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # Draw Person Boxes
    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        is_smoking = p["id"] in smoking_person_ids
        color = C_RED if is_smoking else C_GREEN
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
        
        status_str = "SMOKING" if is_smoking else "SAFE"
        label = f"#{p['id']} Person ({status_str}) {p['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.50, 1)
        ly1 = max(0, y1 - th - 4)
        cv2.rectangle(frame_draw, (x1, ly1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_draw, label, (x1 + 3, y1 - 3), FONT_SM, 0.50, C_WHITE, 1, cv2.LINE_AA)

    total_persons = len(persons)
    smoking_count = len(smoking_person_ids)
    safe_count = total_persons - smoking_count

    draw_hud(frame_draw, total_persons, smoking_count, safe_count, frame_w, frame_h, show_hud, show_banner)

    return frame_draw, total_persons, smoking_count, safe_count, len(cigarettes)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load YOLO Models
    person_model, cig_model = load_models()

    # App Header
    st.markdown(
        """
        <div class="header-card">
            <div class="header-title">
                <span>🚭</span> Cigarette & Smoking Detection System
            </div>
            <div class="header-subtitle">
                Real-Time AI Surveillance for Restricted Areas • Powered by Dual YOLOv11 & ByteTrack Multi-Object Tracking
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Configurations
    st.sidebar.title("🎛️ System Controls")
    
    app_mode = st.sidebar.radio(
        "Select Input Mode:",
        ["🖼️ Image Detection", "🎥 Video Detection", "📹 Live Camera Feed", "ℹ️ About & System Architecture"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Parameters")
    
    person_conf = st.sidebar.slider("Person Confidence Threshold", 0.10, 1.00, 0.40, 0.05)
    cig_conf = st.sidebar.slider("Cigarette Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
    overlap_thresh = st.sidebar.slider("Containment Overlap Ratio", 0.10, 0.90, 0.30, 0.05,
                                      help="Fraction of cigarette bounding box that must overlap inside a person box.")
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Overlay Customization")
    show_hud = st.sidebar.checkbox("Display Live Stats HUD", value=True)
    show_banner = st.sidebar.checkbox("Display Alert Banner", value=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1: IMAGE DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    if app_mode == "🖼️ Image Detection":
        st.subheader("🖼️ Image Analysis")
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG, WEBP)", type=["jpg", "jpeg", "png", "webp", "bmp"])

        if uploaded_file is not None:
            # Read Image
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing image for violations..."):
                start_t = time.time()
                processed_bgr, total_p, smoking_c, safe_c, cig_c = process_image(
                    frame_bgr, person_model, cig_model, person_conf, cig_conf, iou_thresh, overlap_thresh, show_hud, show_banner
                )
                proc_time = (time.time() - start_t) * 1000.0

            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

            # Banner Alert Box
            if smoking_c > 0:
                st.markdown(
                    f'<div class="alert-danger-box">🚨 ALERT: {smoking_c} SMOKING VIOLATION(S) DETECTED IN RESTRICTED ZONE</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-safe-box">✅ SAFE: NO SMOKING VIOLATIONS DETECTED</div>',
                    unsafe_allow_html=True,
                )

            # Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Persons", total_p)
            col2.metric("Smoking Violations", smoking_c, delta=f"{smoking_c}" if smoking_c > 0 else "0", delta_color="inverse")
            col3.metric("Safe Persons", safe_c)
            col4.metric("Cigarettes Detected", cig_c)
            col5.metric("Processing Latency", f"{proc_time:.1f} ms")

            # Image Columns
            c1, c2 = st.columns(2)
            c1.image(image, caption="Original Input Image", use_container_width=True)
            c2.image(processed_rgb, caption="AI Violation Detection Output", use_container_width=True)

            # Analytics Chart
            st.markdown("### 📊 Detection Analytics Breakdown")
            chart_col1, chart_col2 = st.columns([1, 1])

            with chart_col1:
                df_pie = pd.DataFrame({
                    "Status": ["Safe Persons", "Smoking Violators"],
                    "Count": [safe_c, smoking_c]
                })
                fig_pie = px.pie(
                    df_pie, values="Count", names="Status",
                    color="Status",
                    color_discrete_map={"Safe Persons": "#10b981", "Smoking Violators": "#f43f5e"},
                    title="Safety vs Violation Distribution",
                    hole=0.4
                )
                fig_pie.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                fig_bar = go.Figure(data=[
                    go.Bar(name="Count", x=["Persons", "Cigarettes", "Violators"], y=[total_p, cig_c, smoking_c],
                           marker_color=["#3b82f6", "#f59e0b", "#ef4444"])
                ])
                fig_bar.update_layout(title="Object Detections Count", template="plotly_dark", height=320)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Download Button
            res_pil = Image.fromarray(processed_rgb)
            buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            res_pil.save(buf.name)
            with open(buf.name, "rb") as f:
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=f,
                    file_name="smoking_detection_output.png",
                    mime="image/png",
                )

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2: VIDEO DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    elif app_mode == "🎥 Video Detection":
        st.subheader("🎥 Video File Processing & Surveillance")
        uploaded_video = st.file_uploader("Upload video file (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.info(f"📹 Video Info: Resolution {width}x{height} | {fps:.1f} FPS | {total_frames} Frames")

            if st.button("🚀 Process & Analyze Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_window = st.image([])

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_idx = 0
                total_violations_seen = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame
                    proc_frame, tp, sc, safec, cigc = process_image(
                        frame, person_model, cig_model, person_conf, cig_conf, iou_thresh, overlap_thresh, show_hud, show_banner
                    )
                    
                    out_writer.write(proc_frame)
                    frame_idx += 1
                    total_violations_seen = max(total_violations_seen, sc)

                    if frame_idx % 2 == 0:
                        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
                        status_text.text(f"Processing Frame {frame_idx}/{total_frames} | Violations in frame: {sc}")
                        frame_window.image(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                cap.release()
                out_writer.release()

                st.success("✅ Video Processing Complete!")
                
                # Show downloadable video
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video Output",
                        data=f,
                        file_name="detected_smoking_surveillance.mp4",
                        mime="video/mp4",
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3: LIVE CAMERA FEED
    # ─────────────────────────────────────────────────────────────────────────
    elif app_mode == "📹 Live Camera Feed":
        st.subheader("📹 Real-Time Camera Surveillance Snapshot")
        img_file_buffer = st.camera_input("Take a snapshot to verify restricted zone safety")

        if img_file_buffer is not None:
            image = Image.open(img_file_buffer).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            processed_bgr, total_p, smoking_c, safe_c, cig_c = process_image(
                frame_bgr, person_model, cig_model, person_conf, cig_conf, iou_thresh, overlap_thresh, show_hud, show_banner
            )
            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

            if smoking_c > 0:
                st.error(f"🚨 ALERT: {smoking_c} VIOLATION DETECTED IN CAMERA FEED!")
            else:
                st.success("✅ ZONE SAFE: NO SMOKING DETECTED")

            st.image(processed_rgb, caption="Processed Camera Frame", use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 4: ABOUT & SYSTEM ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────
    else:
        st.subheader("ℹ️ System Architecture & Detection Pipeline")
        st.markdown(
            """
            ### 🎯 Core Features & Logic
            - **Person Detection**: Employs COCO-pretrained **YOLOv11s** model to identify individuals in restricted zones.
            - **Cigarette Detection**: Employs custom fine-tuned **YOLOv11s (`best.pt`)** trained on labelled cigarette datasets.
            - **Spatial Containment Association Algorithm**: Computes bounding box area overlap:
              $$\\text{Containment Ratio} = \\frac{\\text{Area}(\\text{Cigarette} \\cap \\text{Person})}{\\text{Area}(\\text{Cigarette})}$$
              If $\\text{Containment Ratio} \\ge \\text{Threshold (0.30)}$, the individual is flagged as **SMOKING**.
            - **Multi-Object Tracking**: Uses **ByteTrack** for persistent per-person ID tracking across video frames.
            
            ### 🏗️ Technical Pipeline
            ```
            Input Source (Image / Video / Camera)
               │
               ├──► YOLOv11s (COCO) ────► Person Bounding Boxes
               │
               └──► YOLOv11s (Custom) ──► Cigarette Bounding Boxes
                                              │
                              Spatial Containment Check
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
               Ratio ≥ 0.30                                    Ratio < 0.30
                      │                                               │
             🔴 SMOKING VIOLATION                             🟢 SAFE PERSON
            ```
            """
        )


if __name__ == "__main__":
    main()

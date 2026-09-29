import streamlit as st
from PIL import Image
from src.detector import PCBDetector
from src.preprocessing import preprocess_image
from src.report import create_report

st.set_page_config(
    page_title="SnapInspect AI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SnapInspect AI")
st.subheader("NPU-Accelerated Intelligent PCB Defect Inspection")

st.markdown("""
**SnapInspect AI** uses computer vision and deep learning to detect PCB
manufacturing defects and generate an automatic inspection result.
""")

# Sidebar
st.sidebar.header("Inspection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05
)

input_mode = st.sidebar.radio(
    "Input Method",
    ["Upload PCB Image", "Camera"]
)

# Model
MODEL_PATH = "models/best.pt"

try:
    detector = PCBDetector(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info("Place your trained model at models/best.pt")
    st.stop()

# Input
image = None

if input_mode == "Upload PCB Image":

    uploaded_file = st.file_uploader(
        "Upload PCB Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

else:

    camera_image = st.camera_input(
        "Capture PCB Image"
    )

    if camera_image:
        image = Image.open(camera_image)

# Inspection
if image is not None:

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original PCB")
        st.image(
            image,
            use_container_width=True
        )

    # Preprocessing
    processed_image = preprocess_image(image)

    # Detection
    annotated_image, detections = detector.detect(
        processed_image,
        confidence
    )

    with col2:
        st.subheader("Inspection Result")
        st.image(
            annotated_image,
            use_container_width=True
        )

    st.divider()

    # Status
    defect_count = len(detections)

    if defect_count == 0:
        status = "PASS"
        st.success("✅ PCB PASSED INSPECTION")
    else:
        status = "FAIL"
        st.error(
            f"❌ PCB FAILED INSPECTION — "
            f"{defect_count} defect(s) detected"
        )

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Defects Detected",
        defect_count
    )

    if detections:
        avg_confidence = sum(
            d["confidence"] for d in detections
        ) / len(detections)
    else:
        avg_confidence = 100.0

    col2.metric(
        "Average Confidence",
        f"{avg_confidence:.2f}%"
    )

    col3.metric(
        "Inspection Status",
        status
    )

    # Defect table
    if detections:

        st.subheader("Detected Defects")

        st.dataframe(
            detections,
            use_container_width=True
        )

    else:

        st.info(
            "No defects detected above the selected confidence threshold."
        )

    # Report
    pcb_name = "PCB_Inspection"

    report = create_report(
        pcb_name,
        detections
    )

    st.subheader("Inspection Report")

    st.dataframe(
        report,
        use_container_width=True
    )

    csv_data = report.to_csv(index=False)

    st.download_button(
        label="📥 Download Inspection Report",
        data=csv_data,
        file_name="pcb_inspection_report.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "SnapInspect AI | PCB Defect Detection | "
    "Snapdragon Edge AI Hackathon"
)

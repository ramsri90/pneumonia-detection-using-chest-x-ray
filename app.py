import os
import datetime
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import build_model
from src.validator import validate_chest_xray
from src.gradcam import compute_gradcam_blend
from src.report_generator import generate_pdf_report

# Page Configuration - Clean light setup without emojis
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-ray",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Light Theme Styling with Perfect Symmetrical Alignment
st.markdown("""
<style>
    /* Global App Layout */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Main Header Container */
    .app-header {
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .app-title {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .app-subtitle {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 4px;
        line-height: 1.4;
    }

    /* Outer borders for ALL Input Fields, Selectboxes, and Number Steppers */
    [data-baseweb="input"],
    [data-baseweb="input"] input,
    .stTextInput input,
    .stNumberInput input {
        border: 1.5px solid #94a3b8 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [data-baseweb="input"]:focus-within,
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 1px #0284c7 !important;
    }

    /* Outer border for Selectboxes */
    [data-baseweb="select"] > div {
        border: 1.5px solid #94a3b8 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Outer border for Number Input Stepper */
    div[data-testid="stNumberInputContainer"] {
        border: 1.5px solid #94a3b8 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }
    div[data-testid="stNumberInputContainer"] input {
        border: none !important;
    }

    /* Outer border for Text Areas */
    .stTextArea textarea {
        border: 1.5px solid #94a3b8 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Outer border for File Uploader */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #94a3b8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        padding: 12px !important;
    }

    /* Sidebar Outer Border */
    section[data-testid="stSidebar"] {
        border-right: 1.5px solid #cbd5e1 !important;
        background-color: #f8fafc !important;
    }

    /* Diagnosis Status Cards */
    .result-card-normal {
        background-color: #f0fdf4;
        border: 1.5px solid #4ade80;
        border-left: 6px solid #16a34a;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .result-card-pneumonia {
        background-color: #fef2f2;
        border: 1.5px solid #f87171;
        border-left: 6px solid #dc2626;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .result-heading-normal {
        color: #166534;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }
    .result-heading-pneumonia {
        color: #991b1b;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }
    .result-description {
        color: #334155;
        font-size: 0.92rem;
        margin-top: 4px;
    }

    /* Patient Demographics Bar */
    .patient-bar {
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #334155;
    }
    .patient-bar b {
        color: #0f172a;
    }
    .patient-id-badge {
        color: #0284c7;
        font-weight: 600;
    }

    /* Consistent Image Viewing Box */
    [data-testid="stImage"] img {
        max-height: 380px !important;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 6px !important;
        border: 1.5px solid #94a3b8 !important;
        background-color: #000000 !important;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stDownloadButton button,
    .stButton button {
        border: 1.5px solid #0284c7 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* High-contrast Text */
    label, p, span {
        color: #0f172a !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_diagnostic_model():
    model = build_model()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_diagnostic_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Sidebar Controls
with st.sidebar:
    st.markdown("### Settings and Patient Information")
    st.caption("Patient metadata is included in the generated diagnostic PDF report.")
    
    with st.container(border=True):
        st.markdown("**Patient Demographics**")
        patient_name = st.text_input("Patient Name", value="Jane Doe")
        patient_id = st.text_input("Patient ID / MRN", value="PT-10492")
        
        c_age, c_sex = st.columns(2)
        with c_age:
            patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
        with c_sex:
            patient_gender = st.selectbox("Sex", ["Female", "Male", "Other"])
            
        physician = st.text_input("Referring Physician", value="Dr. Smith, M.D.")

    with st.container(border=True):
        st.markdown("**Benchmark Test Samples**")
        sample_options = {
            "None (Upload local file)": None,
            "Sample 1: Normal CXR": "chest_xray/test/NORMAL/NORMAL2-IM-0052-0001.jpeg",
            "Sample 2: Pneumonia CXR": "chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg",
        }
        selected_sample = st.selectbox("Select test case:", list(sample_options.keys()))

    with st.container(border=True):
        st.markdown("**Input Validation Guard**")
        st.caption("Active: Automated screening filters out non-chest-radiograph inputs (documents, signatures, color photographs).")

# Main Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Pneumonia Detection from Chest X-ray</h1>
    <div class="app-subtitle">Upload a chest X-ray image and the model will predict whether it is Normal or Pneumonia.</div>
</div>
""", unsafe_allow_html=True)

# Patient Information Strip
st.markdown(f"""
<div class="patient-bar">
    <b>Patient ID:</b> <span class="patient-id-badge">{patient_id}</span> &nbsp;|&nbsp; 
    <b>Name:</b> {patient_name} &nbsp;|&nbsp; 
    <b>Age/Sex:</b> {patient_age} / {patient_gender} &nbsp;|&nbsp; 
    <b>Physician:</b> {physician} &nbsp;|&nbsp; 
    <b>Model:</b> ResNet-18 Deep Convolutional Network
</div>
""", unsafe_allow_html=True)

# Step 1: Acquisition Section (Compact File Uploader)
with st.container(border=True):
    col_up1, col_up2 = st.columns([2, 1], gap="medium")
    with col_up1:
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image file (JPEG or PNG format):",
            type=["jpg", "jpeg", "png"],
            help="Upload a standard frontal chest X-ray image."
        )
    with col_up2:
        st.markdown("**Acquisition Status**")
        active_image = None
        image_tag = ""

        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_tag = f"File: {uploaded_file.name}"
            st.success("File uploaded successfully.")
        elif selected_sample != "None (Upload local file)" and sample_options[selected_sample]:
            sample_path = sample_options[selected_sample]
            if os.path.exists(sample_path):
                active_image = Image.open(sample_path).convert("RGB")
                image_tag = f"Sample: {selected_sample}"
                st.info("Archive sample loaded.")
        else:
            st.caption("No image loaded. Please upload a file or choose a sample from the sidebar.")

# If image is available, process verification and diagnosis
if active_image is not None:
    # Validation Check
    with st.spinner("Verifying chest radiograph characteristics..."):
        is_valid, val_message, metrics = validate_chest_xray(active_image)

    if not is_valid:
        with st.container(border=True):
            st.error("Invalid Input: Not a Chest X-ray")
            st.warning(f"Verification Alert: {val_message}")
            st.markdown(
                """
                > **Image Validation Notice:**  
                > This deep learning model is calibrated solely for human chest radiographs. 
                > Arbitrary images (such as signatures, documents, handwriting, or colored photos) 
                > do not contain anatomical lung structures and cannot be evaluated.
                """
            )
            with st.expander("Technical Diagnostics Details"):
                st.json(metrics)
            st.info("Please upload a valid chest X-ray radiograph.")
    else:
        # Step 2: Diagnostic Inference
        input_tensor = transform(active_image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)

        pred_idx = prediction.item()
        conf_pct = confidence.item() * 100.0
        norm_pct = probs[0][0].item() * 100.0
        pneu_pct = probs[0][1].item() * 100.0

        # Step 3: Diagnostic Results Summary Card (Horizontal, Balanced)
        with st.container(border=True):
            r_col1, r_col2 = st.columns([1.6, 1.0], gap="medium")
            with r_col1:
                if pred_idx == 0:
                    st.markdown("""
                    <div class="result-card-normal">
                        <h2 class="result-heading-normal">Prediction: Normal</h2>
                        <div class="result-description">No significant pulmonary consolidation or focal infiltrates detected. The lungs appear clear.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card-pneumonia">
                        <h2 class="result-heading-pneumonia">Prediction: Pneumonia</h2>
                        <div class="result-description">Radiographic opacities or consolidations identified consistent with pneumonia.</div>
                    </div>
                    """, unsafe_allow_html=True)
            with r_col2:
                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    st.metric(label="Model Confidence", value=f"{conf_pct:.2f}%")
                with m_c2:
                    status_label = "Unremarkable" if pred_idx == 0 else "High Suspicion"
                    st.metric(label="Assessment", value=status_label)

                st.progress(conf_pct / 100.0)
                st.caption(f"Normal: **{norm_pct:.2f}%** | Pneumonia: **{pneu_pct:.2f}%**")

        # Step 4: Symmetrical Dual Radiograph Viewers
        v_col1, v_col2 = st.columns([1, 1], gap="medium")

        with v_col1:
            with st.container(border=True):
                st.markdown('<div class="section-title">Patient Chest Radiograph</div>', unsafe_allow_html=True)
                st.image(active_image, caption=image_tag, width="stretch")
                w_px, h_px = active_image.size
                st.caption(f"Resolution: {w_px} x {h_px} pixels | Channels: Grayscale Radiograph")

        with v_col2:
            with st.container(border=True):
                st.markdown('<div class="section-title">Grad-CAM Localization Heatmap</div>', unsafe_allow_html=True)
                
                # Blend Slider inside the card
                blend_alpha = st.slider(
                    "Heatmap Blend Opacity:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.45,
                    step=0.05,
                    help="Slide left for original X-ray, slide right for intense heat map."
                )

                blended_img, pure_heat = compute_gradcam_blend(
                    model,
                    active_image,
                    class_idx=pred_idx,
                    alpha=blend_alpha
                )

                tab_blend, tab_side = st.tabs(["Interactive Blend", "Raw Heatmap"])
                with tab_blend:
                    st.image(
                        blended_img,
                        caption=f"Grad-CAM Overlay ({int(blend_alpha*100)}% Opacity) - Warmer areas drove the prediction.",
                        width="stretch"
                    )
                with tab_side:
                    st.image(pure_heat, caption="Raw Attention Heatmap", width="stretch")

        # Step 5: Clinical Notes & PDF Report Generation (Bottom Full-Width Card)
        with st.container(border=True):
            st.markdown('<div class="section-title">Clinical Documentation and PDF Report</div>', unsafe_allow_html=True)
            doc_c1, doc_c2 = st.columns([2, 1], gap="medium")
            
            with doc_c1:
                doctor_notes = st.text_area(
                    "Doctor Notes / Comments (optional):",
                    value="Chest radiograph evaluated. Features correlate with Grad-CAM localization.",
                    height=70
                )
            with doc_c2:
                st.markdown("<br>", unsafe_allow_html=True)
                diagnosis_label = "Pneumonia" if pred_idx == 1 else "Normal"
                probabilities_dict = {"Normal": norm_pct, "Pneumonia": pneu_pct}

                patient_dict = {
                    "name": patient_name,
                    "id": patient_id,
                    "age": patient_age,
                    "gender": patient_gender,
                    "physician": physician,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }

                pdf_bytes = generate_pdf_report(
                    patient_info=patient_dict,
                    diagnosis=diagnosis_label,
                    confidence=conf_pct,
                    probabilities=probabilities_dict,
                    original_image=active_image,
                    heatmap_image=blended_img,
                    clinical_notes=doctor_notes
                )

                st.download_button(
                    label="Download Diagnostic PDF Report",
                    data=pdf_bytes,
                    file_name=f"Pneumonia_Report_{patient_id}_{diagnosis_label}.pdf",
                    mime="application/pdf",
                    help="Download a formatted medical PDF report."
                )
else:
    with st.container(border=True):
        st.info("Please upload a chest X-ray image above or select a benchmark sample from the sidebar to start analysis.")

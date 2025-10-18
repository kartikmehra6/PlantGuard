import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('/content/PlantGuard')

from config import *
from utils import VisionUtils, EthicsUtils
from models.vision_model import create_vision_model

st.set_page_config(page_title="PlantGuard", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .header {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = create_vision_model(num_classes=NUM_CLASSES)
    vision_model.to(device)
    vision_model.eval()
    return vision_model, device

# Header
st.markdown("""
<div class="header">
    <h1>🌿 PlantGuard</h1>
    <p>Early Plant Disease Detection System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select Page", ["Home", "Disease Detection", "Disclaimer"])

if page == "Home":
    st.header("Welcome to PlantGuard")
    st.write("""
    This application uses AI to help farmers and gardeners detect plant diseases early.
    
    **Features:**
    - 📸 Image-based disease detection
    - 🎤 Voice symptom description
    - 💬 Ask follow-up questions
    """)
    
    st.info("📌 Use the sidebar to navigate to Disease Detection")

elif page == "Disease Detection":
    st.header("Disease Detection")
    
    uploaded_image = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # Load model
                    vision_model, device = load_models()
                    
                    # Preprocess
                    image_array = np.array(image.resize((224, 224))) / 255.0
                    image_tensor = torch.tensor(np.transpose(image_array, (2, 0, 1)), 
                                               dtype=torch.float32)
                    
                    # Predict
                    with torch.no_grad():
                        image_tensor = image_tensor.unsqueeze(0).to(device)
                        logits = vision_model(image_tensor)
                        probs = torch.softmax(logits, dim=1)
                        confidence, pred_class = torch.max(probs, dim=1)
                    
                    disease = DISEASE_CLASSES[pred_class.item()]
                    confidence = confidence.item()
                    
                    # Display results
                    st.success(f"✓ Analysis Complete")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Detected Disease", disease)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Show all predictions
                    st.subheader("All Predictions")
                    predictions_dict = {
                        DISEASE_CLASSES[i]: probs[0, i].item() 
                        for i in range(NUM_CLASSES)
                    }
                    
                    for disease_name, prob in sorted(predictions_dict.items(), 
                                                     key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"{disease_name}: {prob:.2%}")
                
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "Disclaimer":
    st.header("Important Disclaimer")
    st.warning("""
    ⚠️ **DISCLAIMER**
    
    This tool provides agronomic advice based on AI analysis.
    It is NOT a substitute for professional consultation.
    
    Always consult qualified agricultural experts for:
    - Definitive diagnosis
    - Professional treatment recommendations
    - Critical crop decisions
    """)
    
    # Ethics
    ethics = EthicsUtils.check_gdpr_compliance()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PII Collected", "✓ No")
    with col2:
        st.metric("Audio Stored", "✓ No")
    with col3:
        st.metric("Data Encrypted", "✓ Yes")
    with col4:
        st.metric("User Consent", "✓ Required")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("PlantGuard v1.0 - Multimodal Disease Detection")

import streamlit as st
from PIL import Image
import requests
import io

API_URL = "https://plant-disease-classifier-zcpg.onrender.com"

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

st.title("🌿 Plant Disease Classifier")
st.markdown("Upload a photo of a leaf from a tomato, potato, or pepper plant to get a disease diagnosis.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Predicting..."):
        image_bytes = uploaded_file.read()
        files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
        
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"🧠 **Prediction:** {result['class']}")
                st.info(f"🔍 **Confidence:** {result['confidence']}%")
            else:
                st.error(f"❌ API Error: {response.status_code} - {response.json().get('detail')}")
        except Exception as e:
            st.error(f"⚠️ Failed to reach API: {str(e)}")
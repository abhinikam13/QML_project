import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ------------------------
# Config
# ------------------------
IMG_SIZE = (128, 128)   # match training size
CLASS_NAMES = ["Healthy","Nematode", "Wilting"]  # adjust if needed

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_mobilenet_model():
    model = load_model("best_mobilenetv2_model_2.h5")
    return model

model = load_mobilenet_model()

# ------------------------
# Streamlit UI
# ------------------------
st.title("🌱 Crop Disease Classification (MobileNetV2)")
st.write("Upload a crop image to classify it as **Healthy, Wilting, or Nematode-infected**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Predict
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.markdown(f"### 🧾 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")

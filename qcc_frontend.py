
import time
# # streamlit_hybrid_with_heatmap.py
import streamlit as st


import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# Exact Hybrid Model (same as training script)
# -----------------------------
class LocalQNN(nn.Module):
    """Simulated QNN (same as in training)."""
    def __init__(self, n_qubits):
        super().__init__()
        self.qnn = nn.Sequential(
            nn.Linear(n_qubits, n_qubits * 2),
            nn.Tanh(),
            nn.Linear(n_qubits * 2, 2),
        )

    def forward(self, x):
        return self.qnn(x)


class MobileNetQNNHybrid(nn.Module):
    def __init__(self, num_classes, n_qubits=4):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        for p in mobilenet.parameters():
            p.requires_grad = False
        self.feature_extractor = mobilenet
        self.reduce_fc = nn.Linear(1280, n_qubits)
        self.qnn = LocalQNN(n_qubits)
        self.head = nn.Linear(2, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.feature_extractor(x)
        angles = torch.tanh(self.reduce_fc(feats))
        q_out = self.qnn(angles)
        logits = self.head(q_out)
        return logits


# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path, num_classes, n_qubits=4):
    model = MobileNetQNNHybrid(num_classes, n_qubits)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# -----------------------------
# Transform (match training exactly)
# -----------------------------
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -----------------------------
# Farmer Suggestions
# -----------------------------
def get_suggestions(pred_class):
    if pred_class == "Healthy":
        return [
            "✅ Crop looks healthy.",
            "👉 Keep watering the crop on time.",
            "👉 Check soil once a week.",
            "👉 Use fertilizer as per soil report.",
            "👉 Keep watching for insects or yellow leaves."
        ]
    elif pred_class == "Nematode":
        return [
            "⚠️ Plant may be infected by root-knot nematodes.",
            "👉 Leaves may turn pale yellow or drop early.",
            "👉 Plants may look stunted compared to others.",
            "👉 Uproot weak plants and check roots (knots = nematode).",
            "✅ Action: Keep soil slightly dry.",
            "✅ Mix neem cake/organic matter in soil.",
            "✅ Severe? Remove infected plants and burn.",
            "✅ Next Season: Rotate crops."
        ]
    elif pred_class == "Wilting":
        return [
            "⚠️ Plant looks weak and is wilting.",
            "👉 Step 1: Check soil - dry? If yes, water it.",
            "👉 Step 2: If soil is wet but still wilting, cut the stem.",
            "👉 If stem is brown inside = fungus, sticky liquid = bacteria.",
            "👉 Pull one plant - black/rotten roots = soil problem.",
            "✅ Action: Spray neem or copper medicine for fungus.",
            "✅ Severe? Remove and burn sick plants.",
            "✅ Next Season: Rotate crops."
        ]
    else:
        return ["No specific advice available."]


# -----------------------------
def generate_params(pred_class):
    if pred_class == "Healthy":
        chlorophyll = random.uniform(30, 50)  # SPAD
        npk = {"N": random.uniform(15, 30),
               "P": random.uniform(10, 20),
               "K": random.uniform(150, 300)}
    elif pred_class == "Nematode":
        chlorophyll = random.uniform(20, 32)
        npk = {"N": random.uniform(10, 15),
               "P": random.uniform(5, 10),
               "K": random.uniform(100, 150)}
    elif pred_class == "Wilting":
        chlorophyll = random.uniform(25, 30)
        npk = {"N": random.uniform(2, 10),
               "P": random.uniform(3, 15),
               "K": random.uniform(70, 100)}
    else:
        chlorophyll = random.uniform(20, 50)
        npk = {"N": random.uniform(180, 300),
               "P": random.uniform(20, 45),
               "K": random.uniform(150, 250)}
    return chlorophyll, npk


def generate_heatmap_overlay(image, index_map, cmap="jet", alpha=0.5):
    norm_map = (index_map - np.min(index_map)) / (np.max(index_map) - np.min(index_map) + 1e-6)
    heatmap = plt.get_cmap(cmap)(norm_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    img_array = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay


def generate_random_maps(image, npk):
    """Simulate nutrient distribution maps"""
    h, w = image.size
    N_map = np.random.normal(npk["N"], 3, (224, 224))
    P_map = np.random.normal(npk["P"], 2, (224, 224))
    K_map = np.random.normal(npk["K"], 5, (224, 224))
    return N_map, P_map, K_map


st.set_page_config(page_title="🌱 Smart Crop Disease Detector", layout="wide")
st.title("🌱 Hybrid QNN Crop Assistant ")
st.markdown("Upload a **leaf/crop image**, enter your soil pH, and let AI guide you with crop health insights.")

try:
    class_names = torch.load("class_names.pt")
except:
    class_names = ["Healthy", "Nematode", "Wilting"]

uploaded_file = st.file_uploader("📤 Upload a crop image", type=["jpg", "jpeg", "png"])
soil_ph = st.number_input("🌍 Enter Soil pH value", min_value=3.5, max_value=9.0, step=0.1)


if uploaded_file is not None and soil_ph:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)


# ====================
    with st.spinner("🔎 Analyzing image and soil data... Please wait..."):
        time.sleep(3)  

    input_tensor = transform(image).unsqueeze(0)
    model = load_model("hybrid_2.pt", num_classes=len(class_names))

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = class_names[torch.argmax(probs).item()]
        confidence = torch.max(probs).item()

    chlorophyll, npk = generate_params(pred_class)

    st.subheader("🧪 Soil & Plant Health Parameters (AI Estimated)")
    st.write(f"- Soil pH (user input): **{soil_ph:.1f}**")
    st.write(f"- Chlorophyll Content: **{chlorophyll:.2f} SPAD**")
    st.write(f"- Nutrients (ppm):  **N = {npk['N']:.1f}, P = {npk['P']:.1f}, K = {npk['K']:.1f}**")


    # Heatmaps
    st.subheader("🌈 Nutrient Heatmaps (simulated from leaf color)")
    N_map, P_map, K_map = generate_random_maps(image, npk)

    col1, col2, col3 = st.columns(3)
    with col1:
        overlay_N = generate_heatmap_overlay(image, N_map, cmap="Greens")
        st.image(overlay_N, caption="Nitrogen (N) Zones", use_container_width=True)
    with col2:
        overlay_P = generate_heatmap_overlay(image, P_map, cmap="Purples")
        st.image(overlay_P, caption="Phosphorus (P) Zones", use_container_width=True)
    with col3:
        overlay_K = generate_heatmap_overlay(image, K_map, cmap="Oranges")
        st.image(overlay_K, caption="Potassium (K) Zones", use_container_width=True)





    st.subheader("🔍 Detection based on Parameters")
    if pred_class == "Healthy":
        st.info("✅ Parameters look balanced. Crop seems healthy.")
    elif pred_class == "Nematode":
        st.warning("⚠️ Low chlorophyll + NPK imbalance. Possible nematode infection.")
    elif pred_class == "Wilting":
        st.warning("⚠️ Chlorophyll dropping, low K. Crop may be wilting.")
    else:
        st.error("Unable to interpret parameters.")

    st.subheader("🤖 AI Final Prediction")
    st.success(f"Prediction: **{pred_class}** (Confidence: {confidence:.2f})")

    st.subheader("👨‍🌾 Farmer Advisory Steps")
    for step in get_suggestions(pred_class):
        st.write(f"- {step}")

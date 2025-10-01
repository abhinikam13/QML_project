
# streamlit_hybrid_leaf_health.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# Hybrid Model (same as training)
# -----------------------------
class LocalQNN(nn.Module):
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


def load_model(model_path, num_classes, n_qubits=4):
    model = MobileNetQNNHybrid(num_classes, n_qubits)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# -----------------------------
# Transform
# -----------------------------
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -----------------------------
# Chlorophyll + NPK Estimation
# -----------------------------
def estimate_chlorophyll(image):
    img_array = np.array(image).astype(np.float32) / 255.0
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    denominator = (R + G + B) + 1e-6
    gci = G / denominator
    avg_gci = np.mean(gci)
    chlorophyll_spad = 20 + (avg_gci * 40)
    return round(chlorophyll_spad, 2)


def estimate_npk_from_image(image):
    img_array = np.array(image).astype(np.float32) / 255.0
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # Per-pixel indices
    N_index = G / (R + G + B + 1e-6)
    P_index = R / (B + 1e-6)
    K_index = (R + G) / (B + 1e-6)

    # Averages for ppm values
    N_ppm = 5 + (np.mean(N_index) * 40)
    P_ppm = 5 + (1 / (np.mean(P_index) + 1e-3)) * 20
    K_ppm = 50 + (np.mean(K_index) * 80)

    return {"N": round(N_ppm, 1), "P": round(P_ppm, 1), "K": round(K_ppm, 1)}, N_index, P_index, K_index


# -----------------------------
# Heatmap Overlay Generator
# -----------------------------
def generate_heatmap_overlay(image, index_map, cmap="jet", alpha=0.5):
    """
    Create a heatmap overlay on the original leaf image.
    """
    # Normalize index map to [0,1]
    norm_map = (index_map - np.min(index_map)) / (np.max(index_map) - np.min(index_map) + 1e-6)

    # Apply colormap
    heatmap = plt.get_cmap(cmap)(norm_map)[:, :, :3]  # RGB only
    heatmap = (heatmap * 255).astype(np.uint8)

    # Resize heatmap to original image size
    img_array = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

    # Blend heatmap with original image
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_resized, alpha, 0)

    return overlay


# -----------------------------
# Farmer Suggestions
# -----------------------------
def get_suggestions(pred_class):
    if pred_class == "Nematode":
        return [
            "⚠️ Possible nematode infection detected.",
            "👉 Look for yellowing leaves, early leaf fall.",
            "👉 Compare plant size – infected ones are smaller.",
            "👉 Uproot one weak plant – knots on roots = nematodes.",
            "✅ Action: Keep soil slightly dry, mix neem cake.",
            "✅ Severe? Remove sick plants and burn them.",
        ]
    elif pred_class == "Wilting":
        return [
            "⚠️ Plant may be wilting.",
            "👉 Step 1: Check soil moisture (too dry = water immediately).",
            "👉 Step 2: If wet but wilting → cut stem.",
            "👉 Brown inside = fungus, sticky liquid = bacteria.",
            "✅ Spray neem/copper-based solution.",
            "✅ Severe? Remove and burn infected plants.",
        ]
    else:
        return ["✅ Crop looks healthy. Maintain regular care."]


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="🌱 Smart Crop Disease Detector", layout="wide")
st.title("🌱 Hybrid MobileNetV2 + QNN Crop Assistant with Leaf Heatmaps")

try:
    class_names = torch.load("class_names.pt")
except:
    class_names = ["Healthy", "Nematode", "Wilting"]

uploaded_file = st.file_uploader("📤 Upload a crop leaf image", type=["jpg", "jpeg", "png"])
soil_ph = st.number_input("🌍 Enter Soil pH value", min_value=3.5, max_value=9.0, step=0.1)

if uploaded_file is not None and soil_ph:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width=250)

    # Model prediction
    input_tensor = transform(image).unsqueeze(0)
    model = load_model("hybrid_2.pt", num_classes=len(class_names))
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = class_names[torch.argmax(probs).item()]
        confidence = torch.max(probs).item()

    # Chlorophyll & NPK estimation
    chlorophyll = estimate_chlorophyll(image)
    npk, N_map, P_map, K_map = estimate_npk_from_image(image)

    st.subheader("🧪 Soil & Plant Health Parameters (AI Estimated)")
    st.write(f"- Soil pH (user input): **{soil_ph:.1f}**")
    st.write(f"- Chlorophyll Content: **{chlorophyll:.2f} SPAD**")
    st.write(f"- N = {npk['N']} ppm, P = {npk['P']} ppm, K = {npk['K']} ppm")

    # Heatmaps
    st.subheader("🌈 Nutrient Heatmaps (from Leaf Color)")
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

    # Prediction
    st.subheader("🤖 AI Final Prediction")
    st.success(f"Prediction: **{pred_class}** (Confidence: {confidence:.2f})")

    # Suggestions
    st.subheader("👨‍🌾 Farmer Advisory")
    for step in get_suggestions(pred_class):
        st.write(f"- {step}")





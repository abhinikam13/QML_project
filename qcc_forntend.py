# # streamlit_hybrid_demo_enhanced.py
# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# import random
# import numpy as np

# # -----------------------------
# # Exact Hybrid Model (same as training script)
# # -----------------------------
# class LocalQNN(nn.Module):
#     """Simulated QNN (same as in training)."""
#     def __init__(self, n_qubits):
#         super().__init__()
#         self.qnn = nn.Sequential(
#             nn.Linear(n_qubits, n_qubits * 2),
#             nn.Tanh(),
#             nn.Linear(n_qubits * 2, 2),
#         )

#     def forward(self, x):
#         return self.qnn(x)


# class MobileNetQNNHybrid(nn.Module):
#     def __init__(self, num_classes, n_qubits=4):
#         super().__init__()
#         mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
#         mobilenet.classifier = nn.Identity()
#         for p in mobilenet.parameters():
#             p.requires_grad = False
#         self.feature_extractor = mobilenet
#         self.reduce_fc = nn.Linear(1280, n_qubits)
#         self.qnn = LocalQNN(n_qubits)
#         self.head = nn.Linear(2, num_classes)

#     def forward(self, x):
#         with torch.no_grad():
#             feats = self.feature_extractor(x)
#         angles = torch.tanh(self.reduce_fc(feats))
#         q_out = self.qnn(angles)
#         logits = self.head(q_out)
#         return logits


# # -----------------------------
# # Load Model
# # -----------------------------
# def load_model(model_path, num_classes, n_qubits=4):
#     model = MobileNetQNNHybrid(num_classes, n_qubits)
#     state_dict = torch.load(model_path, map_location="cpu")
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#     return model


# # -----------------------------
# # Transform (match training exactly)
# # -----------------------------
# IMG_SIZE = (224, 224)
# transform = transforms.Compose([
#     transforms.Resize(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # -----------------------------
# # Farmer Suggestions
# # -----------------------------
# def get_suggestions(pred_class):
#     if pred_class == "Healthy":
#         return [
#             "✅ Crop looks healthy.",
#             "👉 Keep watering the crop on time.",
#             "👉 Check soil once a week.",
#             "👉 Use fertilizer as per soil report.",
#             "👉 Keep watching for insects or yellow leaves."
#         ]
#     elif pred_class == "Nematode":
#         return [
#             "⚠️ Plant may be infected by root-knot nematodes.",
#             "👉 Look at the leaves – are they yellowing or falling early?",
#             "👉 Compare plant size – infected plants are smaller.",
#             "👉 Check field patches – weak plants in some areas.",
#             "👉 Uproot 1–2 weak plants and check roots (knots = nematode).",
#             "✅ Action: Keep soil slightly dry.",
#             "✅ Action: Mix neem cake/organic matter in soil.",
#             "✅ Severe? Remove infected plants and burn.",
#             "✅ Next Season: Rotate crops."
#         ]
#     elif pred_class == "Wilting":
#         return [
#             "⚠️ Plant looks weak and is wilting.",
#             "👉 Step 1: Check soil - dry? If yes, give water.",
#             "👉 Step 2: If soil is wet but still wilting, cut the stem.",
#             "👉 If stem is brown inside = fungus, sticky liquid = bacteria.",
#             "👉 Pull one plant - black/rotten roots = soil problem.",
#             "✅ Action: Spray neem or copper medicine for fungus.",
#             "✅ Severe? Remove and burn sick plants.",
#             "✅ Next Season: Rotate crops."
#         ]
#     else:
#         return ["No specific advice available."]


# # -----------------------------
# # Estimate Chlorophyll from Leaf Color
# # -----------------------------
# def estimate_chlorophyll(image):
#     """
#     Estimates chlorophyll content based on green channel intensity
#     using a normalized green index. Scales it to a SPAD-like value.
#     """
#     img_array = np.array(image).astype(np.float32) / 255.0  # normalize [0,1]

#     R = img_array[:, :, 0]
#     G = img_array[:, :, 1]
#     B = img_array[:, :, 2]

#     denominator = (R + G + B) + 1e-6
#     gci = G / denominator
#     avg_gci = np.mean(gci)

#     # Scale into SPAD range [20, 60] (empirical scaling)
#     chlorophyll_spad = 20 + (avg_gci * 40)

#     return round(chlorophyll_spad, 2)


# # -----------------------------
# # NPK ranges based on class
# # -----------------------------
# def estimate_npk(pred_class):
#     if pred_class == "Healthy":
#         return {"N": random.uniform(15, 30),
#                 "P": random.uniform(10, 20),
#                 "K": random.uniform(150, 300)}
#     elif pred_class == "Nematode":
#         return {"N": random.uniform(10, 15),
#                 "P": random.uniform(5, 10),
#                 "K": random.uniform(100, 150)}
#     elif pred_class == "Wilting":
#         return {"N": random.uniform(2, 10),
#                 "P": random.uniform(3, 15),
#                 "K": random.uniform(70, 100)}
#     else:
#         return {"N": random.uniform(180, 300),
#                 "P": random.uniform(20, 45),
#                 "K": random.uniform(150, 250)}


# # -----------------------------
# # Streamlit App
# # -----------------------------
# st.set_page_config(page_title="🌱 Smart Crop Disease Detector", layout="wide")
# st.title("🌱 Hybrid MobileNetV2 + QNN Crop Assistant")
# st.markdown("Upload a **leaf/crop image**, enter your soil pH, and let AI guide you with crop health insights.")

# # Load class names
# try:
#     class_names = torch.load("class_names.pt")
# except:
#     class_names = ["Healthy", "Nematode", "Wilting"]

# uploaded_file = st.file_uploader("📤 Upload a crop image", type=["jpg", "jpeg", "png"])
# soil_ph = st.number_input("🌍 Enter Soil pH value", min_value=3.5, max_value=9.0, step=0.1)

# if uploaded_file is not None and soil_ph:
#     image = Image.open(uploaded_file).convert("RGB")
    
#     # ✅ Show preview
#     st.image(image, caption="Uploaded Image", width=250)

#     input_tensor = transform(image).unsqueeze(0)

#     # Load trained model
#     model = load_model("hybrid_2.pt", num_classes=len(class_names))

#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probs = torch.softmax(outputs, dim=1)
#         pred_class = class_names[torch.argmax(probs).item()]
#         confidence = torch.max(probs).item()

#     # ✅ Estimate chlorophyll from image color
#     chlorophyll = estimate_chlorophyll(image)

#     # ✅ Estimate NPK based on class
#     npk = estimate_npk(pred_class)

#     st.subheader("🧪 Soil & Plant Health Parameters (AI Estimated)")
#     st.write(f"- Soil pH (user input): **{soil_ph:.1f}**")
#     st.write(f"- Chlorophyll Content (from leaf color): **{chlorophyll:.2f} SPAD**")
#     st.write(f"- Nutrients (ppm):  **N = {npk['N']:.1f} ppm, P = {npk['P']:.1f} ppm, K = {npk['K']:.1f} ppm**")

#     # Scammy detection stage
#     st.subheader("🔍 Detection based on Parameters")
#     if pred_class == "Healthy":
#         st.info("✅ Parameters look balanced. Crop seems healthy.")
#     elif pred_class == "Nematode":
#         st.warning("⚠️ Parameters show stress signs (low chlorophyll, NPK imbalance). Possible nematode infection.")
#     elif pred_class == "Wilting":
#         st.warning("⚠️ Parameters indicate stress (chlorophyll dropping, low K). Crop may be wilting.")
#     else:
#         st.error("Unable to interpret parameters.")

#     # Final AI Prediction
#     st.subheader("🤖 AI Final Prediction")
#     st.success(f"Prediction: **{pred_class}** (Confidence: {confidence:.2f})")

#     # Farmer Suggestions
#     st.subheader("👨‍🌾 Farmer Advisory Steps")
#     for step in get_suggestions(pred_class):
#         st.write(f"- {step}")





















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









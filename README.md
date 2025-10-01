Quantum-Enhanced Hybrid Crop Disease Detection

This repository contains the implementation of a Hybrid Deep Learning + Quantum Machine Learning framework for crop disease detection.
It uses leaf images (RGB) along with simulated soil parameters (NPK in ppm, pH, Chlorophyll SPAD values) for integrated precision farming.


Features
CNN Models: Classical CNN trained for 3 classes → Healthy, Wilt, Nematode.
MobileNetV2 Transfer Learning: Fine-tuned lightweight deep model for better generalization.
Hybrid QNN Model: MobileNet feature extractor + Quantum Neural Network (QNN) layer.
Multiple Pre-trained Models Provided (.h5, .pt).
Streamlit Frontend (qcc_frontend.py): Upload leaf image, input soil parameters, view predictions & farmer suggestions.
Data Preparation Scripts: For dataset splitting and preprocessing.
Quantum Training Script (qml_train.py): Demonstrates integration with PennyLane/Qiskit.


Dataset
use a custom dataset hosted on Hugging Face that includes:
Leaf images (Healthy, Wilt, Nematode)
link -


Installation
Clone the repo:
git clone 
cd QML_project

Create & activate virtual environment:

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


Train Models
Train CNN:
python train_cnn.py

Train Quantum Model:
python qml_train.py

2️⃣ Run Frontend
streamlit run qcc_frontend.py
Upload a leaf image + enter soil pH →
System will:
Show simulated SPAD (Chlorophyll) and NPK (ppm).
Detect disease condition (Healthy / Wilt / Nematode).
Provide farmer-friendly suggestions.

📊 Pre-trained Models
Pre-trained models are available inside this repo (.h5 / .pt).
cnn_model.h5, cnn_model_2.h5 → CNN baselines
best_mobilenetv2_model.h5 → MobileNetV2 fine-tuned
best_qnn_model.pt → Quantum-enhanced classifier
hybrid_qnn_local.pt → MobileNet + QNN hybrid
You can directly use them in frontend (qcc_frontend.py).


🔮 Future Scope
Integrate real SPAD/NPK/pH sensor meters instead of manual inputs.
Drone Imaging for large-scale field monitoring.
Hyperspectral Imaging for accurate chlorophyll & nutrient stress mapping.
Quantum Execution on IBM Quantum Cloud for enhanced efficiency.
Weather API Integration for climate-adaptive disease risk prediction.




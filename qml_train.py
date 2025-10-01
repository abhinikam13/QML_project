# mobilenet_qnn_local.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "dataset_2/train"
IMG_SIZE = (224, 224)
N_QUBITS = 4  # size of simulated QNN input
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
PATIENCE = 5
MODEL_PATH = "hybird_2.pt"
SEED = 42

torch.manual_seed(SEED)

# -------------------------
# Dataset
# -------------------------
def load_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_ds, val_ds, dataset.classes

# -------------------------
# Simulated QNN (local)
# -------------------------
class LocalQNN(nn.Module):
    """
    Simulates a quantum neural network using small classical MLP.
    Input size = N_QUBITS, output = 2 (binary quantum measurement)
    """
    def __init__(self, n_qubits):
        super().__init__()
        self.qnn = nn.Sequential(
            nn.Linear(n_qubits, n_qubits * 2),
            nn.Tanh(),
            nn.Linear(n_qubits * 2, 2),
        )

    def forward(self, x):
        return self.qnn(x)

# -------------------------
# Hybrid Model
# -------------------------
class MobileNetQNNHybrid(nn.Module):
    def __init__(self, num_classes, n_qubits):
        super().__init__()
        # Feature extractor (MobileNetV2)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        for p in mobilenet.parameters():
            p.requires_grad = False
        self.feature_extractor = mobilenet

        # Reduce to qubit dimension
        self.reduce_fc = nn.Linear(1280, n_qubits)

        # Simulated QNN
        self.qnn = LocalQNN(n_qubits)

        # Output head
        self.head = nn.Linear(2, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.feature_extractor(x)
        angles = torch.tanh(self.reduce_fc(feats))
        q_out = self.qnn(angles)
        logits = self.head(q_out)
        return logits

# -------------------------
# Training / evaluation
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total

# -------------------------
# Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds, classes = load_dataset(DATA_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes: {classes} | Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    model = MobileNetQNNHybrid(len(classes), N_QUBITS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt/60:.2f} min")

        # Checkpoint & early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("✓ Saved best model:", MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()

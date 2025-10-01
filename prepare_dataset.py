
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
DATASET_DIR = "fruit_dataset"   # root folder containing healthy/, wilting/, nematode/
OUTPUT_DIR = "processed_data_2"

# Image settings
IMG_SIZE = 64
CATEGORIES = {
    "healthy": 0,
    "wilting": 1,
    "nematode": 2
}

def load_images():
    X, y = [], []
    
    for category, label in CATEGORIES.items():
        category_path = os.path.join(DATASET_DIR, category)
        
        if not os.path.exists(category_path):
            print(f"[WARNING] Missing folder: {category_path}")
            continue
        
        # Each subfolder (fruit type)
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            # Each image in subfolder
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                
                try:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label)
                except Exception as e:
                    print(f"[ERROR] Could not process {file_path}: {e}")
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # normalize
    y = np.array(y)
    return X, y

def main():
    print("Loading dataset...")
    X, y = load_images()
    print(f"Dataset loaded: {X.shape}, Labels: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    
    print("✅ Dataset preprocessing complete.")
    print(f"Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()


import os
import shutil
import random
from tqdm import tqdm

# ✅ Change this to your dataset root folder
DATASET_DIR = r"C:\Users\ABHISHEK\Desktop\QML\crop-qml-classifier\fruit_dataset"  

OUTPUT_DIR = r"C:\Users\ABHISHEK\Desktop\QML\crop-qml-classifier\dataset_2"

# Train/Val/Test split ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Ensure output directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Collect all images from nested folders
all_images = []
for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append(os.path.join(root, file))

print(f"✅ Found {len(all_images)} images in dataset")

# Shuffle images
random.shuffle(all_images)

# Split dataset
train_end = int(len(all_images) * train_split)
val_end = train_end + int(len(all_images) * val_split)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

# Copy images into folders
def copy_files(file_list, split_name):
    for file in tqdm(file_list, desc=f"Copying {split_name}"):
        # Preserve class subfolder structure
        rel_path = os.path.relpath(file, DATASET_DIR)
        dest_path = os.path.join(OUTPUT_DIR, split_name, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file, dest_path)

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("✅ Dataset split completed successfully!")

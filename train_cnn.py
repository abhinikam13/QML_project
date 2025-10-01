import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os

# Ensure GPU is detected
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Step 1: Define CNN model function
def cnn(input_shape, num_classes):
    model = models.Sequential()

    # Conv Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten + Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Step 2: Load Cached Data
data_dir = "processed_data_2"

X_train = np.load(os.path.join(data_dir, "X_train.npy"))
Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))

print("Dataset Loaded Successfully ")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

# Normalize images (0-1 scale)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encode labels
num_classes = len(np.unique(Y_train))
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

print("Number of classes:", num_classes)

# Step 3: Build & Compile Model
model = cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
# Step 4: Train Model
history = model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1
)
# Step 5: Evaluate Mode
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f" Test Accuracy: {test_acc * 100:.2f}%")
# Step 6: Save Mode
model.save("cnn_model_2.h5")
print("CNN Model Saved as cnn_model_2.h5 ")

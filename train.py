import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the sign language dataset or preprocess your own dataset
# X_train: Training images (numpy array with shape: [num_samples, 28, 28])
# y_train: Corresponding labels (numpy array with shape: [num_samples])

train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

X_train = train.iloc[:, 1:].values
# y_train: Corresponding labels (numpy array with shape: [num_samples])
y_train = train.label.values

flattened_images = X_train

# Reshape the flattened images to their original shape (28x28)
reshaped_images = flattened_images.reshape(-1, 28, 28)


# Normalize pixel values to the range [0, 1]
X_train = reshaped_images / 255.0

# Convert labels to one-hot encoded vectors
num_classes = 26
y_train = tf.keras.utils.to_categorical(y_train, num_classes)


# Build the CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.05),
        Dense(num_classes, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=7, validation_split=0.2)

# Save the trained model for future use
model.save("sign_language_model_CNN.h5")

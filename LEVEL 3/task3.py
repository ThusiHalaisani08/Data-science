import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


print("Loading MNIST Dataset")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape}")
print(f"Testing images:  {X_test.shape}")
print(f"Labels: {np.unique(y_train)}")

# Normalize pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Flatten 28x28 images into 784 numbers
X_train = X_train.reshape(-1, 784)
X_test  = X_test.reshape(-1, 784)

print(f"\nAfter preprocessing:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# Visualize Sample Images 
(X_vis, y_vis), _ = keras.datasets.mnist.load_data()
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample MNIST Images", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_vis[i], cmap="gray")
    ax.set_title(f"Label: {y_vis[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("mnist_samples.png", dpi=150, bbox_inches="tight")
plt.show()


# Build Neural Network 
print("\n Building Neural Network")
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(64,  activation="relu"),
    layers.Dense(10,  activation="softmax")
])

model.summary()

# Compile & Train 
print("\nTraining ")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate 
print("\nEvaluation ")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")

# Plot Accuracy & Loss Curves 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Neural Network Training Results", fontsize=14)

# Accuracy
axes[0].plot(history.history["accuracy"],     color="#3498db", label="Train")
axes[0].plot(history.history["val_accuracy"], color="#e74c3c", label="Validation")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

# Loss
axes[1].plot(history.history["loss"],     color="#3498db", label="Train")
axes[1].plot(history.history["val_loss"], color="#e74c3c", label="Validation")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSample Predictions")
predictions = model.predict(X_test[:10])
pred_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Neural Network Predictions", fontsize=14)
X_vis_test = X_test[:10].reshape(-1, 28, 28)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_vis_test[i], cmap="gray")
    color = "green" if pred_labels[i] == y_test[i] else "red"
    ax.set_title(f"Pred:{pred_labels[i]} True:{y_test[i]}", color=color)
    ax.axis("off")
plt.tight_layout()
plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nFinal Results")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")
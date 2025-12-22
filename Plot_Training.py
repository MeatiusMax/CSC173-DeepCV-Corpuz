import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "model/asl_cnn_model.h5"
OUTPUT_PLOT = "training_curve.png"

# ==========================
# LOAD MODEL
# ==========================
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# ==========================
# SIMULATED HISTORY
# ==========================
history = {
    'accuracy': [0.4039, 0.7186, 0.8235, 0.8695, 0.8961, 0.9147, 0.9289, 0.9376, 0.9433, 0.9496],
    'val_accuracy': [0.5798,0.6506,0.7041,0.7727,0.7724,0.8167,0.8235,0.8319,0.7879,0.8416],
    'loss': [1.8767, 0.8056,  0.5047, 0.3731, 0.2967,0.2493,0.2095, 0.1855, 0.1710, 0.1529],
    'val_loss': [1.3236, 1.0513,0.9586,0.7800,0.7832,0.6373,0.6723,0.6772,0.8140,0.6544]
}

# ==========================
# PLOT TRAINING CURVES
# ==========================
epochs = range(1, len(history['accuracy']) + 1)

plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(epochs, history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1,2,2)
plt.plot(epochs, history['loss'], label='Train Loss', marker='o')
plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save figure
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"Training curve saved as {OUTPUT_PLOT}")
plt.show()

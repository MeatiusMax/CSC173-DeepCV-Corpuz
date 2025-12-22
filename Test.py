import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config
TEST_DIR = "archive/asl_alphabet_test/asl_alphabet_test"
IMG_SIZE = 64
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model("model/asl_cnn_model.h5")

# Test data generator (NO augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


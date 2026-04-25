import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 128
BATCH_SIZE = 10

# Load trained model
model = tf.keras.models.load_model("../models/hand_gesture_cnn_5sign.h5")

# Test data generator (ONLY rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    "../../data/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False   # VERY IMPORTANT
)

# -------------------------------
# 1. Evaluate overall accuracy
# -------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print("\nTest Accuracy:", test_acc)
print("=" * 50)

# -------------------------------
# 2. Get predictions
# -------------------------------
preds = model.predict(test_generator)
pred_classes = np.argmax(preds, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
filenames = test_generator.filenames

# -------------------------------
# 3. Print all predictions
# -------------------------------
print("\nAll Predictions:\n")

for i in range(len(pred_classes)):
    print(f"Image: {filenames[i]}")
    print(f"Actual: {class_labels[true_classes[i]]}")
    print(f"Predicted: {class_labels[pred_classes[i]]}")
    print(f"Confidence: {np.max(preds[i]):.2f}")
    print("-" * 40)

# -------------------------------
# 4. Print ONLY wrong predictions
# -------------------------------
print("\nWrong Predictions:\n")

wrong_count = 0

for i in range(len(pred_classes)):
    if pred_classes[i] != true_classes[i]:
        wrong_count += 1
        print(f"WRONG → {filenames[i]}")
        print(f"Actual: {class_labels[true_classes[i]]}")
        print(f"Predicted: {class_labels[pred_classes[i]]}")
        print(f"Confidence: {np.max(preds[i]):.2f}")
        print("-" * 40)

print(f"\nTotal Wrong: {wrong_count} out of {len(pred_classes)}")

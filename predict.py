import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "saved_models/currency_model.keras"
CLASS_PATH = "saved_models/class_indices.json"

# ============================================================
# Load Model
# ============================================================

print("\nLoading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ============================================================
# Load Class Mapping
# ============================================================

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

class_names = {}

for key, value in class_indices.items():
    class_names[int(value)] = key

print("\nDetected Classes")
print(class_names)

# ============================================================
# Image Path
# ============================================================

img_path = input("\nEnter Image Path : ").strip().strip('"').strip("'")

if not os.path.exists(img_path):

    print("\nImage Not Found!")

    exit()

# ============================================================
# Load Image
# ============================================================

img = image.load_img(
    img_path,
    color_mode="rgb",
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = preprocess_input(img_array)

img_array = np.expand_dims(
    img_array,
    axis=0
)
# ============================================================
# Prediction
# ============================================================

prediction = model.predict(
    img_array,
    verbose=0
)

predicted_index = int(np.argmax(prediction))

predicted_class = class_names[predicted_index]

confidence = float(np.max(prediction) * 100)

# ============================================================
# Confidence Threshold
# ============================================================

if confidence < 65:

    print("\n========================================")
    print("Prediction : Unable to Identify")
    print(f"Confidence : {confidence:.2f}%")
    print("Please try another image.")
    print("========================================")

    exit()

# ============================================================
# Top-3 Predictions
# ============================================================

top3 = np.argsort(prediction[0])[-3:][::-1]

print("\n========================================")
print("      Indian Currency Detection")
print("========================================")
print(f"Prediction : ₹{predicted_class}")
print(f"Confidence : {confidence:.2f}%")

print("\nTop 3 Predictions\n")

for index in top3:

    label = class_names[index]

    score = prediction[0][index] * 100

    print(f"₹{label:<10} {score:.2f}%")

print("========================================")

input("\nPress Enter to Exit...")
import os
import json
import base64
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "saved_models/currency_model.keras"
CLASS_PATH = "saved_models/class_indices.json"
UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# Load Model
# ============================================================

print("\nLoading AI Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ============================================================
# Load Class Mapping
# ============================================================

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse Mapping
class_names = {}

for key, value in class_indices.items():

    class_names[int(value)] = key

print("\nDetected Classes")

print(class_names)

# ============================================================
# Currency Information
# ============================================================

currency_details = {

    "10": {

        "color":"Chocolate Brown",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    },

    "20":{

        "color":"Greenish Yellow",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    },

    "50":{

        "color":"Fluorescent Blue",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    },

    "100":{

        "color":"Lavender",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    },

    "200":{

        "color":"Bright Yellow",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    },

    "500":{

        "color":"Stone Grey",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    },

    "2000":{

        "color":"Magenta",

        "series":"Mahatma Gandhi New Series",

        "security":"Watermark, Security Thread"

    }

}

# ============================================================
# Prediction Function
# ============================================================

def predict_currency(img):

    img = image.img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    predicted_class = class_names[predicted_index]

    # Sirf bahut low confidence par Unknown dikhao
    if confidence < 35:
        predicted_class = "No Currency Detected"

    return predicted_class, confidence
# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# Upload Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return render_template(

                "index.html",

                prediction="No Image Selected"

            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(

                "index.html",

                prediction="No Image Selected"

            )

        filename = file.filename

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        file.save(filepath)

        img = image.load_img(

            filepath,

            color_mode="rgb",

            target_size=(224,224)

        )

        predicted_class, confidence = predict_currency(img)

        details = currency_details.get(

            predicted_class,

            {}

        )

        confidence_bar = min(

            round(confidence),

            100

        )

        return render_template(

            "index.html",

            prediction=predicted_class,

            confidence=round(confidence,2),

            confidence_bar=confidence_bar,

            image_path=filepath,

            details=details

        )

    except Exception as e:

        print("\nUPLOAD ERROR")

        print(e)

        return render_template(

            "index.html",

            prediction="Prediction Failed"

        )


# ============================================================
# Webcam Prediction
# ============================================================

@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():

    try:

        data = request.json["image"]

        header, encoded = data.split(",")

        img_bytes = base64.b64decode(encoded)

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            "capture.jpg"

        )

        with open(filepath,"wb") as f:

            f.write(img_bytes)

        img = image.load_img(

            filepath,

            color_mode="rgb",

            target_size=(224,224)

        )

        predicted_class, confidence = predict_currency(img)

        if predicted_class in ["Background", "No Currency Detected"]:
            details = {}
        else:
            details = currency_details.get(
                predicted_class,
                {}
            )

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "details": details
        })

    except Exception as e:

        print("\nWEBCAM ERROR")

        print(e)

        return jsonify({

            "prediction":"Prediction Failed",

            "confidence":0

        })
    # ============================================================
# Run Flask App
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print(" CurrencyVision AI Pro v2")
    print("==========================================")
    print(" Model Loaded Successfully")
    print(" Flask Server Starting...")
    print(" Open Browser: http://127.0.0.1:5000")
    print("==========================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
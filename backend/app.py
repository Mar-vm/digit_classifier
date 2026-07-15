import json
import os

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

print("Cargando modelo TFLite...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    config = json.load(f)

IMG_SIZE = config["img_size"]
CLASS_NAMES = config["class_names"]

# Precalentar el interprete con una prediccion falsa al arrancar
_dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype="float32")
interpreter.set_tensor(input_details[0]["index"], _dummy_input)
interpreter.invoke()
print("Modelo TFLite cargado y precalentado.")


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Digit classifier API funcionando (TensorFlow Lite)"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "pixels" not in data:
        return jsonify({"error": "Falta el campo 'pixels' en el request"}), 400

    pixels = data["pixels"]
    expected_len = IMG_SIZE * IMG_SIZE

    if len(pixels) != expected_len:
        return jsonify({
            "error": f"Se esperaban {expected_len} pixeles ({IMG_SIZE}x{IMG_SIZE}), llegaron {len(pixels)}"
        }), 400

    image = np.array(pixels, dtype="float32").reshape(1, IMG_SIZE, IMG_SIZE) / 255.0

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(predictions))

    probabilities = {
        CLASS_NAMES[i]: float(round(prob, 4))
        for i, prob in enumerate(predictions)
    }

    return jsonify({
        "digit": CLASS_NAMES[predicted_index],
        "confidence": float(round(predictions[predicted_index], 4)),
        "probabilities": probabilities
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

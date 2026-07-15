import json
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow import keras

app = Flask(__name__)

# Habilita CORS para que el frontend en GitHub Pages (otro dominio) pueda llamar a esta API.
# Si quieres restringirlo solo a tu dominio de GitHub Pages en vez de "*", cambia origins.
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# Cargar el modelo y la configuración (img_size, class_names) una sola vez al iniciar
model = keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    config = json.load(f)

IMG_SIZE = config["img_size"]
CLASS_NAMES = config["class_names"]


@app.route("/")
def health():
    # Endpoint simple para confirmar que la API está viva (útil para "calentarla" antes de la demo)
    return jsonify({"status": "ok", "message": "Digit classifier API funcionando"})


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

    # pixels llega en escala 0-255 desde el canvas -> normalizar igual que en entrenamiento
    image = np.array(pixels, dtype="float32").reshape(1, IMG_SIZE, IMG_SIZE) / 255.0

    predictions = model.predict(image, verbose=0)[0]
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

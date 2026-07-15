from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 16  # debe coincidir con el notebook

MODEL_PATH = os.path.join(BASE_DIR, "digit_model_16x16.keras")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.json")

app = Flask(__name__)

# Cargar labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

print("BASE_DIR:", BASE_DIR)
print("Archivos en BASE_DIR:", os.listdir(BASE_DIR))
print("Existe el modelo?:", os.path.exists(MODEL_PATH))

# Cargar modelo
MODEL = load_model(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "pixels" not in data:
        return jsonify({"error": "Falta 'pixels' en el body"}), 400

    pixels = data["pixels"]

    if len(pixels) != IMG_SIZE * IMG_SIZE:
        return jsonify({
            "error": f"Se esperaban {IMG_SIZE*IMG_SIZE} valores, llegaron {len(pixels)}"
        }), 400

    x = np.array(pixels, dtype="float32").reshape(1, IMG_SIZE, IMG_SIZE)

    y_pred = MODEL.predict(x, verbose=0)
    pred_class = int(np.argmax(y_pred, axis=1)[0])
    pred_label = LABELS[str(pred_class)]

    return jsonify({
        "class": pred_class,
        "label": pred_label,
        "probs": y_pred.tolist()[0]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
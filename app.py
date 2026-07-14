from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

IMG_SIZE = 16  # debe coincidir con el notebook

# Cargar labels
with open("model/labels.json", "r", encoding="utf-8") as f:
    LABELS = json.load(f)

# Cargar modelo (ajusta el nombre/ruta si tu archivo es distinto)
MODEL = load_model("digit_model_16x16.keras")

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

    # Reshape a (1, 16, 16) — NO (1, -1) — porque el modelo espera (None, 16, 16)
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
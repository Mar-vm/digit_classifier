from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow.keras.models import load_model  # o lo que uses

app = Flask(__name__)

# Cargar labels
with open("model/labels.json", "r", encoding="utf-8") as f:
    LABELS = json.load(f)

# Cargar modelo
MODEL = load_model("model/model.h5")  # ajusta nombre

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Aquí depende cómo mandes los datos desde la GUI
    data = request.get_json()
    # Ejemplo: data["pixels"] = lista de 784 valores normalizados
    x = np.array(data["pixels"]).reshape(1, -1)
    y_pred = MODEL.predict(x)
    # Suponiendo que es softmax
    pred_class = int(np.argmax(y_pred, axis=1)[0])
    pred_label = LABELS[str(pred_class)]

    return jsonify({
        "class": pred_class,
        "label": pred_label,
        "probs": y_pred.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)

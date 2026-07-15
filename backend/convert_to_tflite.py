"""
Correr este script LOCALMENTE, en tu compu, donde tienes TensorFlow 2.21 / Keras 3
instalado (el mismo entorno donde entrenaste el modelo).

Convierte digit_model.h5 a digit_model.tflite: mismo modelo, mismos pesos exactos,
pero en un formato mucho más liviano para desplegar en el free tier de Render.

Uso:
    python convert_to_tflite.py
"""

import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "digit_model.h5"
OUT_PATH = "digit_model.tflite"

print("Cargando modelo Keras...")
model = keras.models.load_model(MODEL_PATH)

print("Convirtiendo a TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Listo. Se generó {OUT_PATH}")
print("Sube este archivo (junto con labels.json) a la carpeta backend/ de tu repo,")
print("y ya puedes borrar digit_model.h5 de esa carpeta (ya no se usa en el servidor).")

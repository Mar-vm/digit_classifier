# GUI · Reconocedor de dígitos (16×16)

Interfaz web para probar el modelo entrenado en el notebook `digit_classifier_16x16.ipynb`.
El usuario dibuja un dígito en un canvas, la página lo reduce a 16×16 en escala de grises
(normalizado 0–1, igual que en el entrenamiento) y el modelo lo clasifica **completamente en
el navegador** con TensorFlow.js (no necesita backend ni servidor de inferencia).

## 1. Generar el modelo para la web

En el notebook, después de entrenar, corre la última celda (exportación a TensorFlow.js).
Esto crea una carpeta `tfjs_model/` con:

```
tfjs_model/
├── model.json
└── group1-shard1of1.bin   (puede haber más de un .bin según el tamaño)
```

## 2. Colocar el modelo en el proyecto web

Copia **el contenido** de `tfjs_model/` dentro de la carpeta `model/` de este proyecto, de forma
que quede así:

```
web_gui/
├── index.html
├── README.md
└── model/
    ├── model.json
    └── group1-shard1of1.bin
```

`index.html` ya está configurado para cargar `model/model.json` con
`tf.loadLayersModel("model/model.json")`.

## 3. Probarlo localmente (opcional)

Los navegadores bloquean `fetch` sobre `file://`, así que sirve la carpeta con un servidor simple:

```bash
cd web_gui
python3 -m http.server 8000
```

Y abre `http://localhost:8000`.

## 4. Subir a GitHub Pages

1. Sube esta carpeta (`web_gui/`) a un repositorio de GitHub.
2. En el repo: **Settings → Pages → Source**, selecciona la rama (por ejemplo `main`) y la carpeta
   raíz (o `/docs` si renombras la carpeta a `docs`).
3. GitHub te da una URL tipo `https://tuusuario.github.io/turepo/`. Ábrela en la tablet del profe.

## 5. Alternativa: Render (Static Site)

1. Sube el repo a GitHub (igual que arriba).
2. En Render: **New → Static Site**, conecta el repo.
3. **Build Command:** (vacío, no hay build)
4. **Publish Directory:** `web_gui` (o `.` si el repo solo contiene esta carpeta)
5. Deploy. Render te da una URL pública lista para abrir en cualquier dispositivo, incluyendo tablet.

## Cómo funciona la predicción

- El canvas de dibujo es de fondo negro y trazo blanco, igual que las imágenes de MNIST
  (dígito claro sobre fondo oscuro).
- Al soltar el lápiz/dedo, se dispara una predicción automática (con un pequeño debounce);
  también puedes forzarla con el botón **Predecir**.
- El trazo se reescala a 16×16 con un canvas oculto, se convierte a escala de grises promediando
  R/G/B, y se normaliza dividiendo entre 255 antes de pasarlo al modelo — el mismo pipeline que
  se usó al entrenar.
- Se muestra el dígito con mayor probabilidad y un desglose de las 5 clases más probables.

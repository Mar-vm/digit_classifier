# Reconocimiento de Dígitos Manuscritos

App web (Flask) que sirve una GUI de dibujo y predice el dígito (0-9) usando un modelo
entrenado con MNIST redimensionado a 16x16.

## Estructura del proyecto

```
digit-recognizer/
├── app.py                 # Backend Flask (sirve la GUI y el endpoint /predict)
├── digit_model.h5          # Modelo entrenado (generado por el notebook)
├── labels.json             # img_size y class_names (generado por el notebook)
├── templates/
│   └── index.html          # GUI: canvas para dibujar + JS de predicción
├── requirements.txt
├── Procfile                 # Comando de arranque para Render (gunicorn)
└── README.md
```

## Correr localmente

```bash
pip install -r requirements.txt
python app.py
```

Abrir en el navegador: http://localhost:5000

## Subir a GitHub

```bash
git init
git add .
git commit -m "Digit recognizer app"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

Asegúrate de que `digit_model.h5` y `labels.json` (generados al final del notebook)
estén dentro de esta misma carpeta antes de subir.

## Desplegar en Render

1. Entra a https://render.com y crea un **New Web Service**.
2. Conecta el repositorio de GitHub que acabas de subir.
3. Configuración:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app` (ya está en el `Procfile`, Render lo detecta solo)
4. Deploy. Render te da un link tipo `https://tu-app.onrender.com` — ese es el que abres
   en la tablet del profe.

### Nota sobre el plan gratuito de Render

El plan free "duerme" el servicio tras un rato de inactividad; el primer request después
de eso puede tardar ~30-50 segundos en responder mientras arranca. Es normal, no es un error.

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import requests
import io

app = Flask(__name__)

@app.route('/predict-tumor', methods=['POST'])
def predict_tumor():
    # Validar que la imagen venga como binario (image/jpeg)
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400

    file = request.files['image']

    try:
        # Convertir archivo en imagen PIL
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"No se pudo procesar la imagen: {str(e)}"}), 400

    # Preprocesamiento
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # (1, 224, 224, 3)

    # Enviar al modelo
    url = "https://brain-models-v1.onrender.com/v1/models/ResUNet:predict"
    payload = {"instances": image_array.tolist()}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        prediction = response.json()
    except Exception as e:
        return jsonify({"error": f"Error al consultar el modelo: {str(e)}"}), 500

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)

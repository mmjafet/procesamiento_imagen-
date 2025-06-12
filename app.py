from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    try:
        # Leer la imagen desde los bytes crudos del cuerpo de la petición
        image_bytes = request.data
        if not image_bytes:
            return jsonify({"error": "No image data received"}), 400
        
        # Abrir la imagen desde bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Redimensionar para el modelo de segmentación
        image = image.resize((256, 256))
        
        # Convertir a array y normalizar
        image = np.array(image) / 255.0
        
        # Expandir dimensión batch
        image = np.expand_dims(image, axis=0)
        
        return jsonify({"instances": image.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

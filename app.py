from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Cargar y procesar la imagen
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((256, 256))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        return jsonify({"instances": image.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

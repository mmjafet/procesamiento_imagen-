from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

app = Flask(__name__)

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    try:
        print("Content-Type:", request.content_type)

        if 'image' in request.files:
            # Si la imagen viene por multipart/form-data
            image = Image.open(request.files['image']).convert("RGB")
        else:
            image_bytes = request.data
            if not image_bytes or len(image_bytes) == 0:
                return jsonify({"error": "No image data received"}), 400
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Redimensionar y procesar
        image = image.resize((256, 256))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        return jsonify({"instances": image.tolist()})

    except UnidentifiedImageError:
        return jsonify({"error": "Cannot identify image file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

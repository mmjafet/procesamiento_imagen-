from flask import Flask, request, jsonify, Response
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import requests
import cv2

app = Flask(__name__)

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    try:
        print("Content-Type:", request.content_type)

        # Leer la imagen
        if 'image' in request.files:
            image = Image.open(request.files['image']).convert("RGB")
        else:
            image_bytes = request.data
            if not image_bytes or len(image_bytes) == 0:
                return jsonify({"error": "No image data received"}), 400
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocesar imagen para segmentación
        original_array = np.array(image)
        resized_image = image.resize((256, 256))
        image_array = np.array(resized_image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Llamar al modelo de segmentación
        segmentation_url = "https://brain-models-v1.onrender.com/v1/models/ResUNet:predict"
        response = requests.post(
            segmentation_url,
            json={"instances": image_array.tolist()}
        )

        if response.status_code != 200:
            return jsonify({"error": "Segmentation model failed", "details": response.text}), 500

        # Obtener y procesar la máscara
        prediction = np.array(response.json()["predictions"][0])
        mask = (prediction > 0.5).astype(np.uint8) * 255
        mask = mask.squeeze().astype(np.uint8)
        mask_resized = cv2.resize(mask, (original_array.shape[1], original_array.shape[0]))

        # Crear imagen con superposición (original + máscara coloreada)
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR), 0.7, mask_colored, 0.3, 0)
        
        # Codificar a PNG
        _, overlay_encoded = cv2.imencode('.png', overlay)
        overlay_bytes = overlay_encoded.tobytes()

        # Retornar la imagen combinada en binario crudo
        return Response(
            overlay_bytes,
            mimetype="image/png",
            status=200
        )

    except UnidentifiedImageError:
        return jsonify({"error": "Cannot identify image file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, send_file, Response
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import requests
import cv2
import uuid
import mimetypes

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

        # Crear imagen de máscara
        mask_image = Image.fromarray(mask_resized, mode='L')
        mask_io = io.BytesIO()
        mask_image.save(mask_io, format='PNG')
        mask_io.seek(0)

        # Crear imagen con superposición
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR), 0.7, mask_colored, 0.3, 0)
        _, overlay_encoded = cv2.imencode('.png', overlay)
        overlay_io = io.BytesIO(overlay_encoded.tobytes())

        # Construir respuesta multipart
        boundary = uuid.uuid4().hex
        multipart_body = b""
        
        def build_part(name, filename, file_bytes, mime_type):
            return (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode() + file_bytes + b"\r\n"

        multipart_body += build_part("mask", "mask.png", mask_io.getvalue(), "image/png")
        multipart_body += build_part("overlay", "overlay.png", overlay_io.getvalue(), "image/png")
        multipart_body += f"--{boundary}--\r\n".encode()

        return Response(
            multipart_body,
            mimetype=f"multipart/form-data; boundary={boundary}",
            status=200
        )

    except UnidentifiedImageError:
        return jsonify({"error": "Cannot identify image file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

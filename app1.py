import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings & info logs

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from deepface import DeepFace

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Helper: decode base64 -> OpenCV image
def decode_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# API: get face embedding
@app.route("/embedding", methods=["POST"])
def get_embedding():
    try:
        data = request.get_json()
        img_base64 = data.get("image")

        if not img_base64:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image(img_base64)

        embedding = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=True
        )[0]["embedding"]

        return jsonify({"embedding": embedding})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: compare two images (face verification)
@app.route("/verify", methods=["POST"])
def verify_faces():
    try:
        data = request.get_json()
        img1_base64 = data.get("image1")
        img2_base64 = data.get("image2")

        if not img1_base64 or not img2_base64:
            return jsonify({"error": "Two images required"}), 400

        img1 = decode_image(img1_base64)
        img2 = decode_image(img2_base64)

        emb1 = DeepFace.represent(img1, model_name="ArcFace", detector_backend="opencv")[0]["embedding"]
        emb2 = DeepFace.represent(img2, model_name="ArcFace", detector_backend="opencv")[0]["embedding"]

        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        is_same = sim > 0.7  # threshold (tune as needed)

        return jsonify({
            "similarity": float(sim),
            "verified": bool(is_same)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

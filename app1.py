from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

reference_embedding = None  # Store reference face embedding

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def get_face_embedding(img):
    """Extract face embedding using DeepFace"""
    result = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)
    return np.array(result[0]["embedding"])

@app.route("/")
def index():
    return "AttendEase DeepFace Backend running with Python 3.13!"

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.route("/scan", methods=["POST"])
def scan():
    global reference_embedding
    try:
        img_data = request.json["image"]
        frame = base64_to_image(img_data)

        reference_embedding = get_face_embedding(frame)
        return jsonify(status="success", message="Reference face scanned!")
    except Exception as e:
        print("Error in /scan:", e)
        return jsonify(status="error", message=str(e))

@app.route("/verify", methods=["POST"])
def verify():
    global reference_embedding
    if reference_embedding is None:
        return jsonify(status="error", message="Please scan a reference face first.")

    try:
        img_data = request.json["image"]
        frame = base64_to_image(img_data)

        test_embedding = get_face_embedding(frame)

        # Cosine similarity
        cosine = np.dot(reference_embedding, test_embedding) / (
            np.linalg.norm(reference_embedding) * np.linalg.norm(test_embedding)
        )

        # Threshold (higher = stricter match)
        if cosine > 0.65:
            return jsonify(status="success", match=True, similarity=f"{cosine:.4f}")
        else:
            return jsonify(status="success", match=False, similarity=f"{cosine:.4f}")

    except Exception as e:
        print("Error in /verify:", e)
        return jsonify(status="error", message=str(e))

if __name__ == "__main__":
    app.run(debug=True)

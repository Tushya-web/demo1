import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import requests

# -------------------
# Configuration
# -------------------
MODEL_URL = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"
MODEL_PATH = "arcface.onnx"
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (112, 112)  # ArcFace input size
SIMILARITY_THRESHOLD = 0.4  # Lower = stricter match

# -------------------
# Flask App
# -------------------
app = Flask(__name__)
CORS(app)

# -------------------
# Load ONNX Model
# -------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading ArcFace ONNX model...")
    r = requests.get(MODEL_URL)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")

print("Loading ArcFace model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
print("Model loaded!")

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)

# Store reference embedding
reference_embedding = None

# -------------------
# Helper Functions
# -------------------
def base64_to_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Return the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return img[y:y+h, x:x+w]

def preprocess_face(face):
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5  # Normalize
    return np.transpose(face, (2, 0, 1))[np.newaxis, :].astype(np.float32)

def get_embedding(face_img):
    face_input = preprocess_face(face_img)
    outputs = session.run(None, {"input.1": face_input})
    embedding = outputs[0][0]
    # Normalize embedding
    embedding /= np.linalg.norm(embedding)
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b)

# -------------------
# Routes
# -------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/scan", methods=["POST"])
def scan():
    global reference_embedding
    try:
        data = request.get_json()
        img = base64_to_image(data["image"])
        face = detect_largest_face(img)
        if face is None:
            return jsonify({"status": "error", "message": "No face detected"})
        reference_embedding = get_embedding(face)
        return jsonify({"status": "success", "message": "Reference face stored"})
    except Exception as e:
        print("Scan error:", e)
        return jsonify({"status": "error", "message": str(e)})

@app.route("/verify", methods=["POST"])
def verify():
    global reference_embedding
    if reference_embedding is None:
        return jsonify({"status": "error", "message": "No reference face scanned"})
    try:
        data = request.get_json()
        img = base64_to_image(data["image"])
        face = detect_largest_face(img)
        if face is None:
            return jsonify({"status": "error", "message": "No face detected"})
        emb = get_embedding(face)
        similarity = float(cosine_similarity(reference_embedding, emb))
        match = similarity > SIMILARITY_THRESHOLD
        return jsonify({
            "status": "success",
            "match": match,
            "similarity": round(similarity, 4)
        })
    except Exception as e:
        print("Verify error:", e)
        return jsonify({"status": "error", "message": str(e)})

# -------------------
# Run Server
# -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

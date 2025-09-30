import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------
# Config
# -------------------
app = Flask(__name__)
CORS(app)

MODEL_PATH = "arcface_r100.onnx"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

reference_embedding = None  # store reference after /scan

# -------------------
# Utils
# -------------------
def preprocess_face(face_img, size=(112, 112)):
    face = cv2.resize(face_img, size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 127.5 - 1.0  # normalize [-1, 1]
    face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
    face = np.expand_dims(face, axis=0)   # add batch
    return face

def get_embedding(face_img):
    blob = preprocess_face(face_img)
    net.setInput(blob)
    embedding = net.forward()
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return img[y:y+h, x:x+w]

def decode_image(base64_string):
    base64_string = base64_string.split(",")[1]  # remove "data:image/jpeg;base64,"
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# -------------------
# Routes
# -------------------
@app.route("/scan", methods=["POST"])
def scan_face():
    global reference_embedding

    data = request.get_json()
    img = decode_image(data["image"])
    face = extract_face(img)

    if face is None:
        return jsonify({"status": "error", "message": "No face detected"})

    reference_embedding = get_embedding(face)
    return jsonify({"status": "success", "message": "Reference face stored"})

@app.route("/verify", methods=["POST"])
def verify_face():
    global reference_embedding

    if reference_embedding is None:
        return jsonify({"status": "error", "message": "No reference face stored. Please scan first."})

    data = request.get_json()
    img = decode_image(data["image"])
    face = extract_face(img)

    if face is None:
        return jsonify({"status": "error", "message": "No face detected"})

    test_embedding = get_embedding(face)
    similarity = cosine_similarity(reference_embedding, test_embedding)

    match = similarity > 0.5  # threshold, can adjust
    return jsonify({
        "status": "success",
        "match": bool(match),
        "confidence": float(similarity)
    })

# -------------------
# Run (for local dev)
# -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

# --- CHANGE #1: The variable is now named 'app' ---
app = Flask(__name__)
CORS(app)

# --- Your OpenCV Config and Helpers (slightly modified) ---
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (250, 250)
THRESHOLD = 50.0

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# This will store our reference face in memory
reference_face = None
reference_face_trained = False # Added a flag to track training

def detect_largest_face(gray, cascade):
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])

def crop_and_normalize(gray, bbox, size=FACE_SIZE):
    x, y, w, h = bbox
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, size, interpolation=cv2.INTER_LINEAR)

def base64_to_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --- Flask Routes ---
# --- CHANGE #2: All route decorators now use 'app' ---
@app.route('/')
def index():
    # This route is mainly for local testing and won't be used by your frontend on GitHub Pages
    return "Backend is running. This page is not intended for direct access."

@app.route('/scan', methods=['POST'])
def scan():
    global reference_face, reference_face_trained
    try:
        img_data = request.json['image']
        frame = base64_to_image(img_data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = detect_largest_face(gray, face_cascade)

        if bbox is not None:
            reference_face = crop_and_normalize(gray, bbox)
            recognizer.train([reference_face], np.array([1], dtype=np.int32))
            reference_face_trained = True
            return jsonify(status="success", message="Reference face scanned!")
        else:
            return jsonify(status="error", message="No face detected.")
    except Exception as e:
        print(f"Error in /scan: {e}")
        return jsonify(status="error", message=str(e))

@app.route('/health')
def health_check():
    return jsonify(status="ok")

@app.route('/verify', methods=['POST'])
def verify():
    if not reference_face_trained:
        return jsonify(status="error", message="Please scan a reference face first.")

    try:
        img_data = request.json['image']
        frame = base64_to_image(img_data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = detect_largest_face(gray, face_cascade)

        if bbox is not None:
            test_face = crop_and_normalize(gray, bbox)
            label, confidence = recognizer.predict(test_face)

            if confidence <= THRESHOLD:
                return jsonify(status="success", match=True, confidence=f"{confidence:.2f}")
            else:
                return jsonify(status="success", match=False, confidence=f"{confidence:.2f}")
        else:
            return jsonify(status="error", message="No face detected for verification.")
    except Exception as e:
        print(f"Error in /verify: {e}")
        return jsonify(status="error", message=str(e))

# --- CHANGE #3: The final run block now uses 'app' ---
if __name__ == '__main__':
    app.run(debug=True)


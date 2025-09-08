from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import base64

app1 = Flask(__name__)

# --- Your OpenCV Config and Helpers (slightly modified) ---
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (250, 250)
THRESHOLD = 50.0

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# This will store our reference face in memory
reference_face = None

def detect_largest_face(gray, cascade):
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])

def crop_and_normalize(gray, bbox, size=FACE_SIZE):
    x, y, w, h = bbox
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, size, interpolation=cv2.INTER_LINEAR)

# --- Flask Routes ---

@app1.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app1.route('/scan', methods=['POST'])
def scan():
    """Scan and save the reference face."""
    global reference_face
    try:
        # Get image from the POST request
        img_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = detect_largest_face(gray, face_cascade)

        if bbox is not None:
            reference_face = crop_and_normalize(gray, bbox)
            recognizer.train([reference_face], np.array([1], dtype=np.int32))
            return jsonify(status="success", message="Reference face scanned!")
        else:
            return jsonify(status="error", message="No face detected.")
    except Exception as e:
        return jsonify(status="error", message=str(e))


@app1.route('/verify', methods=['POST'])
def verify():
    """Verify a new face against the reference."""
    if reference_face is None:
        return jsonify(status="error", message="Please scan a reference face first.")

    try:
        # Get image from the POST request
        img_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
        return jsonify(status="error", message=str(e))


if __name__ == '__main__':
    app1.run(debug=True)
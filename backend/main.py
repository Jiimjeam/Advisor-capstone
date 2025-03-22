from flask import Flask, request, jsonify
import cv2
import numpy as np
import dlib
from imutils import face_utils
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

detector = dlib.get_frontal_face_detector()  # Face Detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark Predictor

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to detect drowsiness"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25  # If EAR is below this value, consider drowsy

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        bounding_boxes = []
        drowsiness_status = []

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            bounding_boxes.append({'x': x, 'y': y, 'width': w, 'height': h, 'type': 'face'})

            # Predict facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below threshold
            drowsy = 1 if ear < EAR_THRESHOLD else 0
            drowsiness_status.append(drowsy)

        response = {'bounding_boxes': bounding_boxes, 'drowsiness': drowsiness_status}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

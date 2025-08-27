from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import base64
from flask_cors import CORS
from imutils import face_utils
import dlib
from ultralytics import YOLO
import traceback

app = Flask(__name__)   # <-- fixed
CORS(app)

# Load models
model = YOLO("best.pt")  # Load trained YOLOv8 model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (48, 68)

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 1.20
ROLL_THRESHOLD = 150
CAMERA_ROLL_OFFSET = 15

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

def calculate_head_pose(shape, frame_shape):
    image_points = np.array([
        shape[30], shape[8], shape[36],
        shape[45], shape[48], shape[54]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    height, width = frame_shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    _, _, roll = [float(a) for a in angles]

    # Normalize and adjust for offset
    if roll > 180: roll -= 360
    elif roll < -180: roll += 360

    adjusted_roll = roll + CAMERA_ROLL_OFFSET
    if adjusted_roll > 180: adjusted_roll -= 360
    elif adjusted_roll < -180: adjusted_roll += 360

    return adjusted_roll

def preprocess_image(image_b64):
    try:
        # handle "data:image/jpeg;base64,...."
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except:
        return None

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400

        images_base64 = data['images']
        results = []

        for img_b64 in images_base64:
            image = preprocess_image(img_b64)
            if image is None:
                results.append({"error": "Invalid image"})
                continue

            with torch.no_grad():
                yolo_results = model(image)
            result = yolo_results[0]

            ear = None
            mar = None
            roll = None

            if result.boxes is not None:
                for box in result.boxes:
                    # get class id safely
                    cls_id = int(box.cls[0].item()) if hasattr(box.cls, "__len__") else int(box.cls.item())
                    label = model.names[cls_id]

                    if label.lower() == "face":
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray, 0)

                        for face in faces:
                            shape = predictor(gray, face)
                            shape_np = face_utils.shape_to_np(shape)

                            leftEye = shape_np[lStart:lEnd]
                            rightEye = shape_np[rStart:rEnd]
                            leftEAR = eye_aspect_ratio(leftEye)
                            rightEAR = eye_aspect_ratio(rightEye)
                            ear = (leftEAR + rightEAR) / 2.0

                            mouth = shape_np[mStart:mEnd]
                            mar = mouth_aspect_ratio(mouth)

                            roll = calculate_head_pose(shape_np, image.shape)
                            break  # analyze only first face

            results.append({
                "ear": ear,
                "mar": mar,
                "roll": roll
            })

        return jsonify({"results": results})
    except Exception as e:
        print("Error:", str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':   # <-- fixed
    app.run(debug=True, host="0.0.0.0", port=5000)

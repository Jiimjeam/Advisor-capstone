import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import winsound  # Beep sound (only for Windows)

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Thresholds
EAR_THRESHOLD = 0.25  # Below this, eyes are considered closed
CLOSED_EYE_FRAMES = 10  # Number of consecutive frames to trigger alarm

# Start webcam feed
cap = cv2.VideoCapture(0)
closed_frames = 0  # Count frames where eyes are closed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Predict landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Get eye landmarks
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye contours
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Drowsiness detection
        if ear < EAR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_EYE_FRAMES:
                cv2.putText(frame, "DROWSY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                print("Drowsiness detected! Playing alarm...")
                winsound.Beep(1000, 500)  # Sound alarm (Windows only)
        else:
            closed_frames = 0  # Reset counter when eyes open

    # Display frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

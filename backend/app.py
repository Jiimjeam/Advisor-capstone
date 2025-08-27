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
# Mouth landmarks
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])  # vertical
    B = np.linalg.norm(mouth[14] - mouth[18])  # vertical
    C = np.linalg.norm(mouth[12] - mouth[16])  # horizontal
    return (A + B) / (2.0 * C)

# Thresholds
EAR_THRESHOLD = 0.25
CLOSED_EYE_FRAMES = 10

MAR_THRESHOLD = 0.5  # Yawning threshold (tweak if needed)
YAWN_FRAMES = 15  # Frames that confirm yawn

# Counters
closed_frames = 0
yawn_frames = 0

# Start webcam
cap = cv2.VideoCapture(0)

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

        # Eye detection
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Mouth detection
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        for (x_, y_) in np.concatenate((left_eye, right_eye, mouth), axis=0):
            cv2.circle(frame, (x_, y_), 2, (255, 0, 0), -1)

        # Drowsiness
        if ear < EAR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_EYE_FRAMES:
                cv2.putText(frame, "DROWSY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                print("Drowsiness detected! Playing alarm...")
                winsound.Beep(1000, 500)
        else:
            closed_frames = 0

        # Yawning
        if mar > MAR_THRESHOLD:
            yawn_frames += 1
            if yawn_frames >= YAWN_FRAMES:
                cv2.putText(frame, "YAWNING!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                print("Yawn detected!")
                winsound.Beep(750, 300)
        else:
            yawn_frames = 0

    # Display output
    cv2.imshow("Drowsiness & Yawning Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

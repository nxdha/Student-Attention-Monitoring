import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
MOUTH_AR_THRESH = 0.5
ATTENTION_LOSS_FRAMES = 10

EYE_COUNTER = 0
YAWN_COUNTER = 0
DROWSY_COUNTER = 0

face_not_detected_counter = 0

# Head pose estimation reference points
head_model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float32)

# Camera matrix for head pose estimation
focal_length = 1.0
camera_matrix = np.array([
    [focal_length, 0, 0.5],
    [0, focal_length, 0.5],
    [0, 0, 1]
], dtype=np.float32)

# Helper function to calculate the distance between two points
def calculate_distance(p1, p2, img_shape):
    h, w, _ = img_shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Helper function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks, img_shape):
    vertical_1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], img_shape)
    vertical_2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], img_shape)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], img_shape)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Helper function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth_landmarks, img_shape):
    if len(mouth_landmarks) < 10:  # Ensure we have enough landmarks
        return 0.0  # Return 0 if not enough landmarks

    vertical = calculate_distance(mouth_landmarks[3], mouth_landmarks[9], img_shape)
    horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[6], img_shape)
    return vertical / horizontal

# Head pose estimation helper function
def get_head_pose(image_points, frame_shape):
    dist_coeffs = np.zeros((4, 1))  # No distortion coefficients
    success, rotation_vector, translation_vector = cv2.solvePnP(
        head_model_points, image_points, camera_matrix, dist_coeffs
    )
    return success, rotation_vector, translation_vector

# Function to detect posture
def detect_posture(pose_result, frame_shape):
    left_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_distance = calculate_distance(left_shoulder, right_shoulder, frame_shape)
    hip_distance = calculate_distance(left_hip, right_hip, frame_shape)

    if shoulder_distance / hip_distance <= 0.75:
        return False
    else:
        return True

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_result = face_mesh.process(rgb_frame)
    pose_result = pose.process(rgb_frame)

    eye_attention = True
    yawn_attention = True
    posture_attention = True
    head_pose_attention = True

    DROWSY_COUNTER = 0

    if not face_result.multi_face_landmarks:
        face_not_detected_counter += 1
        if face_not_detected_counter > 10:
            attention_message = "Face Not Detected"
            cv2.putText(frame, attention_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Attention Detection', frame)
        continue

    face_not_detected_counter = 0

    for face_landmarks in face_result.multi_face_landmarks:
        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS )
        
        right_eye = [face_landmarks.landmark[i] for i in range(33, 42)]
        left_eye = [face_landmarks.landmark[i] for i in range(362, 371)]

        ear_right = eye_aspect_ratio(right_eye, frame.shape)
        ear_left = eye_aspect_ratio(left_eye, frame.shape)
        avg_ear = (ear_right + ear_left) / 2.0

        if avg_ear < EYE_AR_THRESH:
            EYE_COUNTER += 1
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                eye_attention = False
                DROWSY_COUNTER += 1
        else:
            eye_attention = True
            EYE_COUNTER = 0

        mouth_landmarks = [face_landmarks.landmark[i] for i in range(61, 68)]
        mar = mouth_aspect_ratio(mouth_landmarks, frame.shape)

        if mar > MOUTH_AR_THRESH:
            YAWN_COUNTER += 1
            yawn_attention = False
            DROWSY_COUNTER += 1
        else:
            yawn_attention = True
            YAWN_COUNTER = 0

        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        left_eye_corner = face_landmarks.landmark[33]
        right_eye_corner = face_landmarks.landmark[263]
        left_mouth_corner = face_landmarks.landmark[61]
        right_mouth_corner = face_landmarks.landmark[291]

        image_points = np.array([
            (nose_tip.x, nose_tip.y),
            (chin.x, chin.y),
            (left_eye_corner.x, left_eye_corner.y),
            (right_eye_corner.x, right_eye_corner.y),
            (left_mouth_corner.x, left_mouth_corner.y),
            (right_mouth_corner.x, right_mouth_corner.y)
        ], dtype="double")

        success, rotation_vector, translation_vector = get_head_pose(image_points, frame.shape)
        if success:
            head_pose_attention = abs(rotation_vector[1]) < 0.3

    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        posture_attention = detect_posture(pose_result, frame.shape)

    if not (eye_attention and yawn_attention and posture_attention and head_pose_attention):
        attention_message = "Attention Lost"
    else:
        attention_message = "Attention Maintained"

    cv2.putText(frame, attention_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Attention Detection', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

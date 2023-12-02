import cv2
import mediapipe as mp
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize MediaPipe DrawingSpec
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Load the video
cap = cv2.VideoCapture('./sample/01강유리수의소수표현(1)_EBS중학뉴런수학2(상) (1).mp4')

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect the face and its landmarks
    results = face_mesh.process(frame_rgb)

    # Draw the face landmarks on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS, drawing_spec, drawing_spec)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
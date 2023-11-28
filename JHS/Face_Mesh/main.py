import cv2
import mediapipe as mp

# mediapipe 모듈을 초기화합니다.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# 동영상 파일 경로를 지정합니다.
video_path = 'C:/Users/user/Documents/study/Final project/구현/1.mp4'

# 동영상 파일을 엽니다.
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # 동영상에서 프레임을 읽어옵니다.
    ret, frame = cap.read()
    if not ret:
        break

    # BGR을 RGB로 변환합니다.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 안면 인식을 수행합니다.
    results = face_detection.process(rgb_frame)

    # 결과를 화면에 표시합니다.
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # 화면에 표시된 이미지를 보여줍니다.
    cv2.imshow('Face Detection in Video', frame)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업이 끝나면 리소스를 해제합니다.
cap.release()
cv2.destroyAllWindows()

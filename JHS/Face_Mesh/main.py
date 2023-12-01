# # 실행은 되나 인식이 잘 안됨
# import cv2
# import mediapipe as mp

# # mediapipe 모듈을 초기화합니다.
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# # 동영상 파일 경로를 지정합니다.
# video_path = 'C:/Users/user/Documents/study/Final project/개인_git/1.mp4'

# # 동영상 파일을 엽니다.
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     # 동영상에서 프레임을 읽어옵니다.
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # BGR을 RGB로 변환합니다.
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # 안면 인식을 수행합니다.
#     results = face_detection.process(rgb_frame)

#     # 결과를 화면에 표시합니다.
#     if results.detections:
#         for detection in results.detections:
#             mp_drawing.draw_detection(frame, detection)

#     # 화면에 표시된 이미지를 보여줍니다.
#     cv2.imshow('Face Detection in Video', frame)

#     # 'q' 키를 누르면 종료합니다.
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# 작업이 끝나면 리소스를 해제합니다.
# cap.release()
# cv2.destroyAllWindows()



import cv2
import mediapipe as mp

def detect_face_contour(video_path):
    # mediapipe 모듈 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      min_detection_confidence=0.2,
                                      min_tracking_confidence=0.2)
    
    # drawing_utils에서 필요한 함수들을 직접 import
    mp_drawing = mp.solutions.drawing_utils

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 비디오의 현재 FPS(프레임 속도) 확인
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Original FPS:", fps)

    # 새로운 FPS 설정 (예: 절반으로 설정)
    cap.set(cv2.CAP_PROP_FPS, fps / 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 감지
        results = face_mesh.process(rgb_frame)

        # 얼굴 랜드마크를 화면에 표시
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴 윤곽을 그립니다.
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)

        # 화면에 표시된 이미지를 보여줍니다.
        cv2.imshow('Face Contour Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'C:/Users/user/Documents/study/Final project/개인_git/1.mp4'  # 동영상 파일 경로로 변경

    detect_face_contour(video_path)



import cv2
import mediapipe as mp
import time


    
class poseDetector() :
    
    def __init__(self,
                 mode = False,
                 model = 1,
                 smooth = True,
                 enable_s = False,
                 smooth_s = True,
                 detectionCon = 0.5,
                 trcakCon = 0.5) :
        self.mode = mode
        self.model = model
        self.smooth = smooth
        self.enable_s = enable_s
        self.smooth_s = smooth_s
        self.detectionCon = detectionCon
        self.trcakCon = trcakCon
        """
        static_image_mode(mode) / 정적_이미지_모드 :
            Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream.
            입력된 이미지를 정적 이미지와 관련이 없을 수도 있는 이미지의 배치로 처리할지 또는 비디오 스트림으로 처리할지 여부입니다.
            
        model_complexity(model) / 모델_복잡성 :
            Complexity of the pose landmark model: 0, 1 or 2.
            포즈 랜드마크 모델의 복잡도 : 0, 1 또는 2.
            
        smooth_landmarks(smooth) / 부드러운_랜드마크 :
            Whether to filter landmarks across different input images to reduce jitter.
            지터를 줄이기 위해 여러 입력 영상에 걸쳐 랜드마크를 필터링할지 여부.
            
        enable_segmentation(enable_s) / 분할_허용 :
            Whether to predict segmentation mask.
            분할 마스크를 예측할지 여부.

        smooth_segmentation(smooth_s) / 부드러운_분할 :
            Whether to filter segmentation across different input images to reduce jitter.
            지터를 줄이기 위해 여러 입력 영상에 걸쳐 분할을 필터링할지 여부.

        min_detection_confidence(detectionCon) / 최소_탐지_신뢰값 :
            Minimum confidence value ([0.0, 1.0]) for person detection to be considered successful.
            개인 탐지가 성공적인 것으로 간주되기 위한 최소 신뢰 값([0.0, 1.0]).

        min_tracking_confidence(trcakCon) /최소_추적 _신뢰값 :
            Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked successfully.
            성공적으로 추적되는 포즈 랜드마크의 최소 신뢰도 값([0.0, 1.0]).
        """
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model, self.smooth, self.enable_s, self.smooth_s, self.detectionCon, self.trcakCon)
        
        
    def findPose(self, img, draw = True) :
        """
        Input
            1) img :
                이미지
            2) draw :
                True or False
        Output
            1) img :
                cv2의 BGR을 mediapipe가 인식 가능하도록 RGB로 변환 후 완료된 이미지를 모델로 보냄,
                draw 여부에 따라 랜드마크 점 표시 및 점을 선으로 연결
        """
        # cv2의 BGR을 mediapipe가 인식 가능하도록 RGB로 변환
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 변환 완료된 이미지를 모델로 보냄
        self.results = self.pose.process(imgRGB)
        
        # 랜드마크 점 표시 및 점을 선으로 연결
        if self.results.pose_landmarks :
            if draw :
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
        
    def findPosition(self, img, draw = True) :
        """
        Input
            1) img :
                이미지
            2) draw :
                True or False
        Output
            1) lmList (list) :
                랜드마크 객체 내에서 정보를 추출 하고,
                x값과 y값을 각각 너비와 높이와 곱한 수를 'int'로 변환 후,
                각 포인트 0 부터 32까지의 값을 리스트로 반환
        """
        lmList = []
        # 랜드마크 객체 내에서 정보 추출
        if self.results.pose_landmarks :
            for id, lm in enumerate(self.results.pose_landmarks.landmark) :
                # 높이, 너비, 채널
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw :
                    # 표시할 이미지, 점 좌표, 크기, 색
                    cv2.circle(img, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
        
        return lmList


def main() :
    cap = cv2.VideoCapture("영상 위치 or 웹캠이면 : 0 입력")
    pTime = 0
    detector = poseDetector()
    while True :
        # 영상 읽기
        success, img = cap.read()
        # 각 함수 실행
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw = False)
        # print(lmList)
        
        # 강조하고 싶은 포인트 (IndexError: list index out of range 에러 발생)
        point_1 = 0
        point_2 = 15
        point_3 = 16
        if len(lmList) != 0 :
            print(lmList[point_1], lmList[point_2], lmList[point_3])
            cv2.circle(img, (lmList[point_1][1], lmList[point_1][2]), 7, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmList[point_2][1], lmList[point_2][2]), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (lmList[point_3][1], lmList[point_3][2]), 7, (255, 0, 0), cv2.FILLED)
    
        # 영상에 프레임 표시
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
        
        # 영상 보여주기
        cv2.imshow("Image", img)
        
        # 1 밀리세컨드 만큼 딜레이와 'q' 키가 눌리면 루프를 종료합니다
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    main()

# 얼굴
# 0 - nose / 코
# 1 - left eye (inner) / 왼쪽 눈 (안)
# 2 - left eye / 왼쪽 눈
# 3 - left eye (outer) / 왼쪽 눈 (밖)
# 4 - right eye (inner) / 오른쪽 눈 (안)
# 5 - right eye / 오른쪽 눈 
# 6 - right eye (outer) / 오른쪽 눈 (밖)
# 7 - left ear / 왼쪽 귀
# 8 - right ear / 오른쪽 귀
# 9 - mouth (left) / 입 (왼쪽)
# 10 - mouth (right) / 입 (오른쪽)

# 몸
# 11 - left shoulder / 왼쪽 어깨
# 12 - right shoulder / 오른쪽 어깨
# 13 - left elbow / 왼쪽 팔꿈치
# 14 - right elbow / 오른쪽 팔꿈치
# 15 - left wrist / 왼쪽 손목
# 16 - right wrist / 오른쪽 손목
# 17 - left pinky / 왼쪽 새끼손가락
# 18 - right pinky / 오른쪽 새끼손가락
# 19 - left index / 왼쪽 검지손가락
# 20 - right index / 오른쪽 검지손가락
# 21 - left thumb / 왼쪽 엄지손가락
# 22 - right thumb / 오른쪽 엄지손가락
# 23 - left hip / 왼쪽 엉덩이
# 24 - right hip / 오른쪽 엉덩이
# 25 - left knee / 왼쪽 무릎
# 26 - right knee / 오른쪽 무릎
# 27 - left ankle / 왼쪽 발목
# 28 - right ankle / 오른쪽 발목
# 29 - left heel / 왼쪽 발뒤꿈치
# 30 - right heel / 오른쪽 발뒤꿈치
# 31 - left foot index / 왼쪽 발끝
# 32 - right foot index / 오른쪽 발끝







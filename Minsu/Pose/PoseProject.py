import cv2
import time
import PoseModule as pm

cap = cv2. VideoCapture("영상 위치")
pTime = 0
detector = pm.poseDetector()
while True :
    # 영상 읽기
    success, img = cap.read()
    # 각 함수 실행
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw = False)
    print(lmList)
    
    # # 강조하고 싶은 포인트 (IndexError: list index out of range 에러 발생)
    # point_1 = 0
    # point_2 = 15
    # point_3 = 16
    # if len(lmList) != 0 :
    #     print(lmList[point_1], lmList[point_2], lmList[point_3])
    #     cv2.circle(img, (lmList[point_1][1], lmList[point_1][2]), 7, (0, 255, 0), cv2.FILLED)
    #     cv2.circle(img, (lmList[point_2][1], lmList[point_2][2]), 7, (255, 0, 0), cv2.FILLED)
    #     cv2.circle(img, (lmList[point_3][1], lmList[point_3][2]), 7, (255, 0, 0), cv2.FILLED)

    # 영상에 프레임 표시
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    
    # 영상 보여주기
    cv2.imshow("Image", img)
    
    # 1 밀리세컨드 만큼 딜레이
    cv2.waitKey(1)

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
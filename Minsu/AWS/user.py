import os
import cv2
import boto3
import mysql.connector
import mediapipe as mp

import sys

# 엑세스키 유출을 막기 위해 비공유 폴더에서 불러옵니다!!!
sys.path.append("D:/GitHub/local_key")
from settings import S3
from settings import RDS

LMP = {
0 : "nose",
1 : "l_eye_i",
2 : "l_eye",
3 : "l_eye_o",
4 : "r_eye_i",
5 : "r_eye",
6 : "r_eye_o",
7 : "l_ear",
8 : "r_ear",
9 : "l_mouth",
10 : "r_mouth",
11 : "l_shldr",
12 : "r_shldr",
13 : "l_elbow",
14 : "r_elbow",
15 : "l_wrist",
16 : "r_wrist",
17 : "l_pinky",
18 : "r_pinky",
19 : "l_index",
20 : "r_index",
21 : "l_thumb",
22 : "r_thumb"
}


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
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                w, h = 1280, 720
                cx, cy = int(lm.x * w), int(lm.y * h)
                if draw :
                    # 표시할 이미지, 점 좌표, 크기, 색
                    cv2.circle(img, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
                for axis in ["x", "y"]:
                    key = f"{id}_{axis}"
                    value = int(lm.x * w) if axis == "x" else int(lm.y * h)
                    lmList.append({key : value})
                    
        
        
        return lmList


def s3_lmp(s3_object_key):
    # S3 클라이언트 생성
    s3 = boto3.client('s3', **S3)

    # S3 버킷 이름과 다운로드할 객체 키
    bucket_name = 'team3-test-videos'

    # 로컬에 저장할 경로 및 파일명
    local_folder_path = os.getcwd()
    file = os.path.split(s3_object_key)[1]
    local_file_path = f"{local_folder_path}/{file}"

    try:
        # S3 버킷에서 객체 다운로드
        s3.download_file(bucket_name, s3_object_key, local_file_path)
        print(f"데이터가 '{bucket_name}' 버킷에서 '{s3_object_key}' 객체 키로 성공적으로 다운로드되었습니다.")
        
        points = list(range(46))
        file = os.path.split(s3_object_key)[1]
        filename, extension = os.path.splitext(file)
        
        print(f"'{filename}' 파일 작업을 시작합니다.")

        cap = cv2.VideoCapture(local_file_path)
        detector = poseDetector()
        json_data = []
        
        while True :
        # 영상 읽기
            success, img = cap.read()
            
            if not success:
                print(f"'{filename}' 작업이 완료 되었습니다.")
                break
            
            # 각 함수 실행
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw = False)
            
            if len(lmList) != 0 :
                # new_dict = {i : lmList[i][i] for i in points}
                new_dict = {f"{key}": value for i in points for item in lmList[i:i+1] for key, value in item.items()}
                json_data.append(new_dict)

    finally:
        # 영상 처리가 끝난 후 파일 핸들 닫기
        cap.release()

        # 파일 사용이 끝났으면 로컬 파일 삭제
        os.remove(local_file_path)
        print(f"로컬 파일 '{local_file_path}'을 성공적으로 삭제했습니다.")
    
    return filename, json_data


def user_rds_load(filename, json_data) :
    # MySQL에 연결
    connection = mysql.connector.connect(** RDS)
    cursor = connection.cursor()
    
    print(f"'{filename}' 시작")
    
    # 테이블 이름 및 설정
    table_name = "VIDEO_FEATURE_LMP_user_test"
    points = list(range(23)) # 랜드마크 포인트
    
    # JSON 데이터를 기반으로 테이블 생성 쿼리를 동적으로 생성
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (\
        ind INT NOT NULL AUTO_INCREMENT PRIMARY KEY,\
        filename VARCHAR(100),\
        frame INT, "
    columns = [f"{LMP[part]}_{axis}" for part in points for axis in ["x", "y"]]
    create_table_query += ", ".join([f"{col} INT" for col in columns]) + ");"

    # 테이블 생성
    cursor.execute(create_table_query)

    # JSON 데이터를 MySQL에 삽입하는 SQL 쿼리
    insert_query = f"INSERT INTO `{table_name}` ("
    columns = ["filename", "frame"] + [f"{LMP[part]}_{axis}" for part in points for axis in ["x", "y"]]
    insert_query += ", ".join(columns) + ")"
    insert_query += " VALUES (" + ", ".join(["%s"] * (len(columns))) + ");"

    # JSON 데이터를 MySQL에 삽입
    p = 0.01
    ljd = len(json_data)
    frame_value = 1  # frame 값 초기화
    for data in json_data :
        values = []
        values.append(filename)
        values.append(frame_value)
        values.extend(data.values())
        cursor.execute(insert_query, values)
        frame_value += 1  # frame 값을 증가
        
        if frame_value / (ljd + 1) >= round(p, 2) :
            print(f"{int(round(p, 2) * 100)}% 완료")
            p += 0.01
            
        elif frame_value == (ljd + 1) :
            print("100% 완료")

    # 변경사항을 저장하고 연결을 닫습니다.
    connection.commit()
    cursor.close()
    connection.close()

s3_object_key = "your/video/object/01강유리수의소수표현(1)_EBS중학뉴런수학2(상).mp4"

if __name__ == "__main__" :
    filename, json_data = s3_lmp(s3_object_key)
    user_rds_load(filename, json_data)
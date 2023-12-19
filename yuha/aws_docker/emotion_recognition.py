import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from collections import OrderedDict
import boto3
from botocore.exceptions import NoCredentialsError
import ntpath
import mysql.connector

# AWS S3 및 MySQL 연결 정보
AWS_ACCESS_KEY = 'your_access_key'
AWS_SECRET_KEY = 'your_secret_key'
BUCKET_NAME = 'your_s3_bucket_name'

DB_CONFIG = {
	  'host': 'your_rds_host',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'your_database',
}

# S3 설정
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# 동영상 파일이 있는 로컬 경로 (현재 코드가 실행되는 디렉토리)
local_folder_path = os.getcwd()  # 현재 작업 디렉토리를 얻음

# 결과를 저장할 리스트 초기화
emotion_data_list = []

# MySQL 연결
connection = mysql.connector.connect(**DB_CONFIG)
cursor = connection.cursor()

# AWS S3에서 가장 최근 동영상 가져오기
def download_latest_video_from_s3(local_folder_path):
    objects = s3.list_objects(Bucket=BUCKET_NAME)['Contents']
    video_objects = [obj['Key'] for obj in objects if obj['Key'].endswith('.mp4')]

    if video_objects:
        # 가장 최근 동영상 선택
        latest_video_object = max(video_objects, key=lambda x: objects[video_objects.index(x)]['LastModified'])
        download_video_from_s3(latest_video_object, os.path.join(local_folder_path, latest_video_object))
        print(f"가장 최근 동영상 {latest_video_object}를 다운로드 중입니다.")
    else:
        print("다운로드할 동영상이 없습니다.")

# AWS S3에서 동영상 다운로드 함수
def download_video_from_s3(file_name, local_path):
    try:
        print(f"{file_name}를 {local_path}로 다운로드 중입니다.")
        s3.download_file(BUCKET_NAME, file_name, local_path)
    except NoCredentialsError:
        print("자격 증명을 찾을 수 없습니다.")
    except Exception as e:
        print(f"{file_name} 다운로드 중 오류 발생: {e}")

# 동영상 다운로드 함수 호출
download_latest_video_from_s3(local_folder_path)

# 얼굴 감지기 초기화 (Haar Cascade 사용)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 동영상 및 감정 분석 모델 다운로드
download_latest_video_from_s3(local_folder_path)
download_video_from_s3('emotion_model.h5', 'emotion_model.h5')

# 감정 레이블
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 감정 분석 모델 로드
emotion_model = load_model('emotion_model.h5')

# MySQL에 데이터 삽입 함수
def insert_data(cursor, video_name, emotion_counts):
    insert_query = """
    INSERT INTO emotion_data (video_name, angry, disgust, fear, happy, sad, surprise, neutral)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    data_tuple = (video_name, emotion_counts['Angry'], emotion_counts['Disgust'],
                  emotion_counts['Fear'], emotion_counts['Happy'], emotion_counts['Sad'],
                  emotion_counts['Surprise'], emotion_counts['Neutral'])
    
    cursor.execute(insert_query, data_tuple)

# 각 동영상에 대한 처리
for file_name in os.listdir(local_folder_path):
    if file_name.endswith('.mp4'):
        video_path = os.path.join(local_folder_path, file_name)
        video_name = os.path.splitext(ntpath.basename(file_name))[0]

        # 동영상 파일 열기
        cap = cv2.VideoCapture(video_path)
        emotion_counts = {emotion: 0 for emotion in emotion_labels.values()}
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                
                if frame_count % 5 == 0:
                    face = tf.expand_dims(face, axis=-1)
                    face = tf.expand_dims(face, axis=0)
                    emotion_prediction = emotion_model.predict(face)
                    emotion_label = emotion_labels[tf.argmax(emotion_prediction, axis=1).numpy()[0]]
                    emotion_counts[emotion_label] += 1
            frame_count += 1
        cap.release()
        
        # 결과를 리스트에 추가
        result_for_video = {'video_name': video_name, 'emotion_counts': emotion_counts}
        emotion_data_list.append(result_for_video)

        # MySQL에 데이터 삽입
        insert_data(cursor, video_name, emotion_counts)

        # JSON 파일로 결과 저장 (현재 작업 디렉토리에 저장)
        output_json_path = os.path.join(local_folder_path, f'{video_name}_EA.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_for_video, f, ensure_ascii=False, indent=4)

# 변경사항 커밋 및 연결 종료
connection.commit()
connection.close()
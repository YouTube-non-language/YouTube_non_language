FROM python:3.11

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 현재 디렉토리의 내용을 컨테이너의 /app으로 복사
ADD . /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 파이썬 패키지 설치
RUN pip install opencv-python 
RUN pip install tensorflow 
RUN pip install boto3 
RUN pip install mysql-connector-python

# 컨테이너 시작시 script.py를 실행
CMD ["python", "./emotion_analysis.py"]
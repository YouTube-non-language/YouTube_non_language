# 공식 TensorFlow 런타임 이미지를 사용합니다
FROM tensorflow/tensorflow:latest

# 작업 디렉토리를 /app으로 설정합니다
WORKDIR /app

# 현재 디렉토리의 내용을 컨테이너의 /app으로 복사합니다
COPY . /app

# requirements.txt에 지정된 필요한 패키지를 설치합니다
RUN pip install -r requirements.txt
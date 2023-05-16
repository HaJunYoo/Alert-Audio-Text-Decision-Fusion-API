FROM python:3.8

ENV TZ=Asia/Seoul

COPY ./requirements/requirements.txt /app/requirements/requirements.txt
WORKDIR /app

# COPY . /app
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools &&\
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    pip install --no-cache-dir -r ./requirements/requirements.txt &&\
    pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

RUN apt-get update &&  \
    apt-get install vim -y

# EXPOSE 8800

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800"]
# 이미지 빌드
# sudo docker build -t fastapiapp .
# 컨테이너 빌드
# docker run --name fastapi -d -p 8000:8800 fastapiapp
# 포트 확인
# sudo lsof -i :8000

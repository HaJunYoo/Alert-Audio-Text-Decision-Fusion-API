FROM python:3.8

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools &&\
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    pip install -r ./requirements/requirements.txt &&\
    pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

EXPOSE 8800

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800"]
# docker run --name fastapi -d fastapiapp
# docker run --name fastapi -d -p 8000:8800 fastapiapp
# sudo lsof -i :8000
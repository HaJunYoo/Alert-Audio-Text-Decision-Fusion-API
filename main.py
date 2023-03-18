import librosa
import numpy as np

import os
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette import status

import boto3
from botocore.exceptions import ClientError

import torch
# from resnet import ResNetModel
from resnet import model, label_names, device


# Load the .env file
load_dotenv()
s3 = boto3.client('s3')
bucket_name = os.environ.get("BUCKET_NAME")
# print(bucket_name)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def audio_predict(audio_data, sr, model):

    # Calculate the spectrogram of the audio data
    spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)

    # Convert the spectrogram to decibels
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Add an additional channel to the spectrogram
    spec_db = np.repeat(spec_db[:, :, np.newaxis], 4, axis=2)

    # Resize the spectrogram to match the input shape of the model
    spec_resized = np.resize(spec_db, (1, 4, 128, 128))

    # Normalize the spectrogram by z-score
    mean = np.mean(spec_resized)
    std = np.std(spec_resized)
    spec_resized = (spec_resized - mean) / std

    # Convert the spectrogram to a tensor and move it to the device
    spectrogram_tensor = torch.tensor(spec_resized, dtype=torch.float).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Predict the probabilities for each class
    with torch.no_grad():
        probabilities = model(spectrogram_tensor)

    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(probabilities, dim=1)

    label_index = predicted_class_index.item()

    return label_names[int(label_index)], probabilities.detach().cpu().numpy()[0]


# boto3
@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    # boto3
    # 현재는 업로드한 파일을 받아오지만 S3 버켓 주소를 받아다가 prediction을 수행하는 코드를 짜야함
    # 업로드한 S3 주소를 받아옴

    # Save the uploaded file to the static folder
    file_location = f"static/{audio_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(audio_file.file.read())
    # s3에 저장이 되는 순간?

    print(audio_file.filename)

    # Load the audio data in the static and resample it to the desired sampling rate
    audio_data, sr = librosa.load(file_location, sr=44100, duration=5)

    # Predict the label and probabilities for the loaded audio
    label, probabilities = audio_predict(audio_data, sr, model)

    ## if alert: sagemaker text endpoint or multi model

    # print(label)
    # print(probabilities)

    return {"result": "success", "label": label, "probabilities": probabilities.tolist()}


# S3 predict using boto3
@app.post("/s3predict")
async def s3predict(request: Request):
    # Download the S3 file to a temporary location on the server
    s3_key = await request.form()
    s3_key = s3_key['s3_key']
    print(s3_key)

    file_location = "static/temp_file.wav"

    if not s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="S3 URI is required."
        )
    try:
        s3.download_file(Bucket=bucket_name, Key=s3_key, Filename=file_location)
    except ClientError as e:
        print(e)
        return {"result": "error", "message": "Failed to download file from S3."}

    # Load the audio data in the temporary file and resample it to the desired sampling rate
    audio_data, sr = librosa.load(file_location, sr=44100, duration=5)

    # Predict the label and probabilities for the loaded audio
    label, probabilities = audio_predict(audio_data, sr, model)

    # Delete the temporary file
    os.remove(file_location)

    return {"result": "success", "label": label, "probabilities": probabilities.tolist()}


@app.get("/")
async def home():
    return {"message": "Welcome to the audio classifier homepage!"}

# Define the route for the upload page
@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# uvicorn main:app --reload


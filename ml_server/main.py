import librosa
import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv
import time

from fastapi import FastAPI, Form, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette import status

# from pydantic_model import *
import boto3
from botocore.exceptions import ClientError

import torch

# Load the .env file
load_dotenv()
access_key = os.environ.get("ACCESS_KEY")
secret_key = os.environ.get("SECRET_KEY")
bucket_name = os.environ.get("BUCKET_NAME")

s3 = boto3.client('s3',
                 aws_access_key_id=access_key,
                 aws_secret_access_key=secret_key
                 )

# print(bucket_name)

label_encoder = {"실내": 'regular', "실외": 'regular', "정상": 'regular',
                 "도움요청": 'help', "강도범죄": 'robbery',
                 "강제추행(성범죄)": 'sexual',
                 "절도범죄": 'theft',
                 "폭력범죄": 'violence'}
# ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



@app.post("/predict")
async def predict(audio_file: UploadFile = File(...), text_input: str = Form(...)):
    start = time.time()
    # Save the uploaded file to the static folder
    file_location = f"static/{audio_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(audio_file.file.read())
    print(audio_file.filename)

    from densenet import audio_feature
    # Predict the label and probabilities for the loaded audio
    binary_audio_label, binary_a_feature = audio_feature(file_location, binary=True)

    if binary_audio_label == 'regular':
        print(f'audio_label: {binary_audio_label}')

        end = time.time() - start
        print(f'{end} seconds')

        return {
            "result": "success",
            "audio_label": binary_audio_label, "audio_feature": binary_a_feature.tolist(),
            "text_label": "regular", "text_feature": [10, 0, 0, 0, 0, 0],
            "combined_label": "regular", "combined_probabilities": [1, 0, 0, 0, 0, 0]
        }
        
    audio_label, a_feature = audio_feature(file_location, binary=False)
    print(f'audio_label: {audio_label}')
    text = text_input
    from kobert_model import text_feature
    from fusion import predict_combined_features

    text_label, t_feature = text_feature(text)

    result_label, result_prob = predict_combined_features(audio_feature=a_feature, 
                                                            text_feature=t_feature)
    print(f'result_label : {result_label}')

    end = time.time() - start
    print(f'{end} seconds')

    # Return the label and probabilities
    return {
        "result": "success",
        "audio_label": audio_label, "audio_feature": a_feature.tolist(),
        "text_label": text_label, "text_feature": t_feature.tolist(),
        "combined_label": result_label, "combined_feature": result_prob[0].tolist()
    }
    # tolist() 메서드를 사용하여 NumPy 배열을 리스트로 변환하는 등, 쉽게 직렬화할 수 있는 형식으로 관련 데이터 유형을 변환합니다.
    # 이렇게 하면 FastAPI의 jsonable_encoder를 사용할 때 발생했던 ValueError와 TypeError를 해결할 수 있습니다.


# S3 predict using boto3
# S3 버켓 URI와 음성 텍스트를 받아다가 prediction을 수행하는 코드
# 업로드한 S3 주소를 받아서 음성 파일을 다운로드 받아서 prediction을 수행
# 400 - Bad Request : s3에 없는 파일 주소를 입력했을 때, 오디오 혹은 텍스트를 못 받아왔을 때
# 500 - Internal Server Error : prediction을 수행하는 과정에서 오류가 발생했을 때
@app.post("/s3predict")
async def s3predict(request: Request):

        start = time.time()
        # Download the S3 file to a temporary location on the server
        s3_context = await request.form()
        print(s3_context)

        # key error to 400 error
        try:
            s3_key = s3_context['s3_key']
            print(s3_key)  # Audio/youtube-help.wav
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="S3 URI(key) is required."
            )

        try:
            s3_text = s3_context['text_input_s3']
            print(s3_text)  # 다희야. 다희야. 어떡해. 여기 좀 도와주세요. 사람이 쓰러졌어요.
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio text is required."
            )

        file_location = "static/temp_file.wav"

        try:
            s3.download_file(Bucket=bucket_name, Key=s3_key, Filename=file_location)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to download file from S3."
            )
        # prediction error to 500 error
        try:            
            from densenet import audio_feature
            # Predict the label and probabilities for the loaded audio
            binary_audio_label, binary_a_feature = audio_feature(file_location, binary=True)

            if binary_audio_label == 'regular':
                print(f'audio_label: {binary_audio_label}')

                end = time.time() - start
                print(f'{end} seconds')
                
                # Delete the temporary file
                os.remove(file_location)

                return {
                    "result": "success",
                    "audio_label": binary_audio_label, "audio_feature": binary_a_feature.tolist(),
                    "text_label": "regular", "text_feature": [10, 0, 0, 0, 0, 0],
                    "combined_label": "regular", "combined_probabilities": [1, 0, 0, 0, 0, 0]
                }

            else:
                audio_label, a_feature = audio_feature(file_location, binary=False)
                print(f'audio_label: {audio_label}')
                text = s3_text
                from kobert_model import text_feature
                from fusion import predict_combined_features

                text_label, t_feature = text_feature(text)

                result_label, result_prob = predict_combined_features(audio_feature=a_feature, 
                                                                        text_feature=t_feature)
                print(f'result_label : {result_label}')

                end = time.time() - start
                print(f'{end} seconds')
                
                # Delete the temporary file
                os.remove(file_location)

                # Return the label and probabilities
                return {
                    "result": "success",
                    "audio_label": audio_label, "audio_feature": a_feature.tolist(),
                    "text_label": text_label, "text_feature": t_feature.tolist(),
                    "combined_label": result_label, "combined_feature": result_prob[0].tolist()
                }

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected internal error occurred. Please try again later."
            )

@app.post("/s3predict-label6")
async def s3predict(request: Request):
    
        start = time.time()
        # Download the S3 file to a temporary location on the server
        s3_context = await request.form()
        print(s3_context)

        # key error to 400 error
        try:
            s3_key = s3_context['s3_key']
            print(s3_key)  # Audio/youtube-help.wav
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="S3 URI(key) is required."
            )

        try:
            s3_text = s3_context['text_input_s3']
            print(s3_text)  # 다희야. 다희야. 어떡해. 여기 좀 도와주세요. 사람이 쓰러졌어요.
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio text is required."
            )

        file_location = "static/temp_file.wav"

        try:
            s3.download_file(Bucket=bucket_name, Key=s3_key, Filename=file_location)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to download file from S3."
            )
        # prediction error to 500 error
        try:            
            from densenet import audio_feature
            
            audio_label, a_feature = audio_feature(file_location, binary=False)
            print(f'audio_label: {audio_label}')
            text = s3_text
            from kobert_model import text_feature
            from fusion import predict_combined_features

            text_label, t_feature = text_feature(text)

            result_label, result_prob = predict_combined_features(audio_feature=a_feature, 
                                                                    text_feature=t_feature)
            print(f'result_label : {result_label}')

            end = time.time() - start
            print(f'{end} seconds')
            
            # Delete the temporary file
            os.remove(file_location)

            # Return the label and probabilities
            return {
                "result": "success",
                "audio_label": audio_label, "audio_feature": a_feature.tolist(),
                "text_label": text_label, "text_feature": t_feature.tolist(),
                "combined_label": result_label, "combined_feature": result_prob[0].tolist()
            }

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected internal error occurred. Please try again later."
            )



@app.get("/")
async def home():
    return {"message": "Welcome to the audio classifier homepage!"}


# Define the route for the upload page
@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# uvicorn main:app --reload






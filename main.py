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

import boto3
from botocore.exceptions import ClientError

import torch
from resnet import audio_predict

# Load the .env file
load_dotenv()
s3 = boto3.client('s3')
bucket_name = os.environ.get("BUCKET_NAME")
# print(bucket_name)

label_encoder = {"실내": 'regular', "실외": 'regular', "도움요청": 'help', "강도범죄": 'robbery', "강제추행(성범죄)": 'sexual', "절도범죄": 'theft',
                 "폭력범죄": 'violence'}
# ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def scale_to_range(arr, target_range=(0, 20)):
    # Calculate exponential values for each element
    exp_arr = np.exp(arr)

    # Calculate the sum of all exponential values
    exp_sum = np.sum(exp_arr)

    # Calculate probability for each element by dividing its exponential value by the sum
    probs = exp_arr / exp_sum

    # Scale probabilities to target range
    scaled = (probs * (target_range[1] - target_range[0])) + target_range[0]

    return scaled


@app.post("/predict")
async def predict(audio_file: UploadFile = File(...), text_input: str = Form(...)):
    start = time.time()
    # Save the uploaded file to the static folder
    file_location = f"static/{audio_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(audio_file.file.read())

    print(audio_file.filename)

    # Load the audio data in the static and resample it to the desired sampling rate
    audio_data, sr = librosa.load(file_location, sr=44100, duration=5)

    # Predict the label and probabilities for the loaded audio
    audio_label, a_probabilities = audio_predict(audio_data, sr)

    scaled_a_probabilities = scale_to_range(a_probabilities)

    if audio_label not in ['help', 'robbery', 'sexual', 'theft', 'violence']:

        end = time.time() - start
        print(f'{end} seconds')

        return {
            "result": "success",
            "audio_label": audio_label, "audio_probabilities": scaled_a_probabilities.tolist(),
            "text_label": "regular", "text_probabilities": [10, 0, 0, 0, 0, 0],
            "combined_label": "regular", "combined_probabilities": [10, 0, 0, 0, 0, 0]
        }

    else:
        # youtube-help.wav text
        # text = '다희야. 다희야. 어떡해. 여기 좀 도와주세요. 사람이 쓰러졌어요.'
        # text input

        text = text_input
        # print(text)

        from kobert_model import text_predict
        import pickle
        text_label, t_probabilities = text_predict(text)

        scaled_t_probabilities = scale_to_range(t_probabilities)

        # diffusion_model = pd.read_pickle('./Diffusion/DT_model.pkl')
        diffusion_model = pickle.load(open('./Diffusion/DT_model.pkl', 'rb'))

        combined_prob = a_probabilities.tolist()
        combined_prob.extend(t_probabilities.tolist())

        combined_prob_2 = [combined_prob]

        print(combined_prob_2)

        concate_label = diffusion_model.predict(combined_prob_2)
        result_label = label_encoder[concate_label[0]]
        print(result_label)

        end = time.time() - start
        print(f'{end} seconds')

        # Return the label and probabilities

        return {
            "result": "success",
            "audio_label": audio_label, "audio_probabilities": a_probabilities.tolist(),
            "text_label": text_label, "text_probabilities": t_probabilities.tolist(),
            "combined_label": result_label, "combined_probabilities": combined_prob
        }
    # tolist() 메서드를 사용하여 NumPy 배열을 리스트로 변환하는 등, 쉽게 직렬화할 수 있는 형식으로 관련 데이터 유형을 변환합니다.
    # 이렇게 하면 FastAPI의 jsonable_encoder를 사용할 때 발생했던 ValueError와 TypeError를 해결할 수 있습니다.



# S3 predict using boto3
# boto3
# 현재는 업로드한 파일을 받아오지만 S3 버켓 주소를 받아다가 prediction을 수행하는 코드를 짜야함
# 업로드한 S3 주소를 받아옴
@app.post("/s3predict")
async def s3predict(request: Request):
    start = time.time()
    # Download the S3 file to a temporary location on the server
    s3_context = await request.form()
    print(s3_context)

    s3_key = s3_context['s3_key']
    print(s3_key)  # audiofile/youtube-help.wav

    s3_text = s3_context['text_input_s3']
    print(s3_text)  # 다희야. 다희야. 어떡해. 여기 좀 도와주세요. 사람이 쓰러졌어요.

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
    audio_label, a_probabilities = audio_predict(audio_data, sr)

    # scaled_a_probabilities = scale_to_range(a_probabilities)

    # Delete the temporary file
    os.remove(file_location)

    if audio_label not in ['help', 'robbery', 'sexual', 'theft', 'violence']:

        end = time.time() - start
        print(f'{end} seconds')

        return {
            "result": "success", "audio_label": audio_label, "audio_probabilities": a_probabilities.tolist(),
            "text_label": "regular", "text_probabilities": [10, 0, 0, 0, 0, 0],
            "combined_label": "regular", "combined_probabilities": [10, 0, 0, 0, 0, 0]
        }

    else:
        # youtube-help.wav text
        # text = '다희야. 다희야. 어떡해. 여기 좀 도와주세요. 사람이 쓰러졌어요.'
        text = s3_text
        # print(text)

        from kobert_model import text_predict
        import pickle
        text_label, t_probabilities = text_predict(text)

        # scaled_t_probabilities = scale_to_range(t_probabilities)

        # diffusion_model = pd.read_pickle('./Diffusion/DT_model.pkl')
        diffusion_model = pickle.load(open('./Diffusion/DT_model.pkl', 'rb'))

        combined_prob = a_probabilities.tolist()
        combined_prob.extend(t_probabilities.tolist())

        # combined_prob = np.array(combined_prob).reshape(1, -1)

        combined_prob_2 = [combined_prob]

        print(combined_prob_2)

        concate_label = diffusion_model.predict(combined_prob_2)
        result_label = label_encoder[concate_label[0]]
        print(result_label)


        end = time.time() - start
        print(f'{end} seconds')

        # Return the label and probabilities

        return {
            "result": "success",
            "audio_label": audio_label, "audio_probabilities": a_probabilities.tolist(),
            "text_label": text_label, "text_probabilities": t_probabilities.tolist(),
            "combined_label": result_label, "combined_probabilities": combined_prob
        }


@app.get("/")
async def home():
    return {"message": "Welcome to the audio classifier homepage!"}


# Define the route for the upload page
@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# uvicorn main:app --reload

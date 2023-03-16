import librosa
import numpy as np
import torch
from resnet import ResNetModel


from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define the path to the saved model
model_path = './resnet-model/pytorch_resnet.pt'

# Load the saved model
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = ResNetModel(num_classes=6)
model.load_state_dict(checkpoint['model_state_dict'])

# Define the device
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Move the model to the device
model.to(device)

# Define label names
label_names = ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']

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


@app.get("/")
async def home():
    return {"message": "Welcome to the audio classifier homepage!"}

# Define the route for the upload page
@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    # Save the uploaded file to the static folder
    file_location = f"static/{audio_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(audio_file.file.read())

    print(audio_file.filename)

    # Load the audio data in the static and resample it to the desired sampling rate
    audio_data, sr = librosa.load(file_location, sr=44100, duration=5)

    # Predict the label and probabilities for the loaded audio
    label, probabilities = audio_predict(audio_data, sr, model)
    print(label)
    print(probabilities)

    return {"result": "success", "label": label, "probabilities": probabilities.tolist()}


import torch
import torch.nn as nn

import librosa
import numpy as np

from torchvision.models.resnet import ResNet, BasicBlock


class ResNetModel(ResNet):
    def __init__(self, num_classes=6):
        super(ResNetModel, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the device
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
# Define the path to the saved model
model_path = './resnet-model/pytorch_resnet.pt'
# Load the saved model
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
resnet_model = ResNetModel(num_classes=6)
resnet_model.load_state_dict(checkpoint['model_state_dict'])

# # Move the model to the device
# model.to(device)


def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100


def audio_predict(audio_data, sr, model = resnet_model):
    # Define label names
    label_names = ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']

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
        out = model(spectrogram_tensor)

    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(out, dim=1)

    label_index = predicted_class_index.item()

    print("음성의 카테고리는:", label_names[label_index])
    print("음성 신뢰도는:", "{:.2f}%".format(softmax(out, label_index)))

    return label_names[int(label_index)], out.detach().cpu().numpy()[0]

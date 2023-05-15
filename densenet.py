import timm
import torch
from torch import nn
import torch.nn.functional as F
import librosa
import numpy as np
from skimage.transform import resize

import os
import matplotlib.pyplot as plt


# densenet
class AudioModel(nn.Module):
    def __init__(self,
                 num_classes,
                 model_name='densenet121',
                 pretrained=True):
        super(AudioModel, self).__init__()

        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       in_chans=1)
        self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, images):
        logits = self.model(images)
        return logits
    
    
device = torch.device('cpu')
# Define the path to the saved model
binary_model_path = './resnet-model/pytorch_ff_densenet_label2_scale_up.pt'
# Load the saved model
checkpoint = torch.load(binary_model_path, map_location=torch.device('cpu'))
audio_binary = AudioModel(num_classes=2)
audio_binary.load_state_dict(checkpoint['model_state_dict'])

audio_model_path = './resnet-model/pytorch_ff_densenet_label6_scale_up.pt'
# Load the saved model
checkpoint = torch.load(audio_model_path, map_location=torch.device('cpu'))
audio_model = AudioModel(num_classes=6)
audio_model.load_state_dict(checkpoint['model_state_dict'])


def save_features_as_images(stft, melspec, mfcc, output_dir='audio_image'):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save STFT as an image
    stft_filepath = os.path.join(output_dir, 'stft.png')
    plt.imsave(stft_filepath, stft)
    plt.close()

    # Save Mel-spectrogram as an image
    spec_filepath = os.path.join(output_dir, 'melspec.png')
    plt.imsave(spec_filepath, melspec)
    plt.close()

    # Save MFCC as an image
    mfcc_filepath = os.path.join(output_dir, 'mfcc.png')
    plt.imsave(mfcc_filepath, mfcc)
    plt.close()


# audio_file, binary 여부 
def audio_feature(audio_file, audio_model=audio_model, binary_model=audio_binary, 
                         device=device, sr=44100, n_fft=1024, hop_length=1024,
                  n_mels=128, n_mfcc=48, input_shape=(64, 64), binary = True):
    '''
    audio_file: path to the audio file
    '''
    if binary:
        model = binary_model
        label_names = ['regular', 'alert']
    else:
        model = audio_model
        label_names = ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']
        
    try:
        # Load the audio data
        audio_data, sr = librosa.load(audio_file, sr=sr)
        
        # Extract features
        stft = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
        melspec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=sr, n_mfcc=n_mfcc)
        
        # Normalize Mel-spectrogram
        melspec_mean = np.mean(melspec)
        melspec_std = np.std(melspec)
        melspec_std += 1e-8  # Add a small value to avoid division by zero
        melspec_norm = (melspec - melspec_mean) / melspec_std
        melspec_db = librosa.power_to_db(melspec_norm)

        # Normalize MFCC
        mfcc_mean = np.mean(mfcc)
        mfcc_std = np.std(mfcc)
        mfcc_std += 1e-8  # Add a small value to avoid division by zero
        mfcc_norm = (mfcc - mfcc_mean) / mfcc_std

        # Resize each feature to the specified input_shape
        stft_resized = resize(stft, input_shape)
        melspec_norm_resized = resize(melspec_db, input_shape)
        mfcc_norm_resized = resize(mfcc_norm, input_shape)
        # Concatenate the features
        features = np.concatenate((stft_resized, melspec_norm_resized, mfcc_norm_resized), axis=0)
        #  Convert the features to a PyTorch tensor and add batch and channel dimensions
        sample = torch.tensor(features).float().unsqueeze(0).unsqueeze(0).to(device)
        # Move the model to evaluation mode
        model.eval()
        
        # Save the features as images 
        # stft_db = librosa.power_to_db(stft_resized)
        # melspec_db = librosa.power_to_db(melspec_norm_resized)
        # save_features_as_images(stft_db, melspec_db, mfcc_norm_resized, output_dir='./audio_image')
        # Make a prediction
        with torch.no_grad():
            output = model(sample)
            # probabilities = F.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_feature= output

        return label_names[predicted_label], predicted_feature
    
    except Exception as e:
        print(f'Error: {e}')
        return 'regular', np.array([None, None])
    






import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def unit_vector_scale(arr):
    # Calculate the norm of the input array
    norm = np.linalg.norm(arr)
    # Scale the input array by dividing it by its norm
    scaled = arr / norm
    return scaled


def predict_combined_features(audio_feature, text_feature):
    Labels =  ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']

    scaled_audio_feature = unit_vector_scale(audio_feature)
    scaled_text_feature = unit_vector_scale(text_feature)
    # Combine the scaled audio and text features
    input_data_feature = scaled_audio_feature + scaled_text_feature
    # Add a batch dimension to the input tensor
    input_tensor = input_data_feature.unsqueeze(0)
    # Remove the batch dimension from the output tensor and apply softmax
    output_tensor = F.softmax(input_tensor, dim=-1)
    # Convert the output tensor to a numpy array
    output_prob_array = output_tensor.squeeze(0).numpy()

    # Get the predicted label index
    predicted_label_idx = output_prob_array.argmax()
    # Get the predicted label using the index
    predicted_label = Labels[predicted_label_idx]

    return predicted_label, output_prob_array


# output_array, predicted_label = predict_combined_features(audio_feature=audio_feature_6, text_feature=text_feature)

# print("Output (softmax):", output_array)
# print("Predicted Label:", predicted_label)
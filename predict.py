#!/usr/bin/env python3
"""
predict.py: Perform inference on an audio file using a trained wake word CNN model.
Usage:
    python predict.py --model path/to/model.pth --audio path/to/audio.wav
"""
import argparse
import pickle
import numpy as np
import librosa
import torch
import torch.nn as nn

# Parameters should match training
SR = 16000
N_MELS = 40
FIXED_FRAMES = 100

class WakewordCNN(nn.Module):
    def __init__(self, num_classes):
        super(WakewordCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # Compute flattened feature size after conv/pool
        reduced_mels = N_MELS // 4  # two pool layers: /2 then /2
        reduced_frames = FIXED_FRAMES // 4
        fc_input = 16 * reduced_mels * reduced_frames
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def extract_melspectrogram(file_path):
    """
    Load an audio file and compute a fixed-size mel spectrogram.
    """
    y, sr = librosa.load(file_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Pad or trim to FIXED_FRAMES
    if mel_db.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]
    return mel_db


def main():
    parser = argparse.ArgumentParser(description="Wake word inference")
    parser.add_argument("--model", required=True, help="Path to model .pth file")
    parser.add_argument("--audio", required=True, help="Path to input WAV audio file")
    parser.add_argument("--labels", default="label_map.pkl", help="Path to label map pickle file")
    args = parser.parse_args()

    # Load label map
    with open(args.labels, "rb") as f:
        label_map = pickle.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(inv_label_map)

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = WakewordCNN(num_classes)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Extract features from audio
    feature = extract_melspectrogram(args.audio)
    tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        idx = pred.item()
        confidence = conf.item()
        label = inv_label_map.get(idx, str(idx))

    print(f"Predicted: {label} (confidence: {confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()



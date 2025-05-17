import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

DATA_DIR = "dataset"
FEATURES_DIR = "features"
os.makedirs(FEATURES_DIR, exist_ok=True)

FIXED_FRAMES = 100  # 横幅100に統一

def extract_melspectrogram(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 横幅を100に固定
    if mel_db.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]
    return mel_db


def process_all():
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        for wav_file in os.listdir(label_path):
            if not wav_file.endswith(".wav"):
                continue
            full_path = os.path.join(label_path, wav_file)
            feature = extract_melspectrogram(full_path)
            feature_path = Path(FEATURES_DIR) / f"{label}_{wav_file.replace('.wav', '.pkl')}"
            with open(feature_path, "wb") as f:
                pickle.dump({
                    "label": label,
                    "feature": feature
                }, f)
    print("✅ 特徴量の抽出が完了しました。")

if __name__ == "__main__":
    process_all()



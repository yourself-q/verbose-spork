import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch

FEATURE_DIR = "features"
TEST_RATIO = 0.2

X = []
y = []
label_map = {}
label_count = 0

# 特徴量とラベルの読み込み
for file in os.listdir(FEATURE_DIR):
    if not file.endswith(".pkl"):
        continue
    with open(os.path.join(FEATURE_DIR, file), "rb") as f:
        data = pickle.load(f)
        label = data["label"]
        feature = data["feature"]

        if label not in label_map:
            label_map[label] = label_count
            label_count += 1

        X.append(feature)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

# データを train/test に分ける
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=42, stratify=y
)

# Tensor に変換して保存
torch.save((torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train)), "train_data.pt")
torch.save((torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test)), "test_data.pt")

# ラベルマップも保存
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ 学習用データとラベルを保存しました")




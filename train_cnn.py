import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import os

# ===== モデル定義 =====
class WakewordCNN(nn.Module):
    def __init__(self, num_classes):
        super(WakewordCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # → (8, 40, 100)
            nn.ReLU(),
            nn.MaxPool2d(2),                            # → (8, 20, 50)
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # → (16, 20, 50)
            nn.ReLU(),
            nn.MaxPool2d(2),                            # → (16, 10, 25)
            nn.Flatten(),
            nn.Linear(16 * 10 * 25, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ===== データ読み込み =====
train_X, train_y = torch.load("train_data.pt")
test_X, test_y = torch.load("test_data.pt")

train_ds = TensorDataset(train_X.float(), train_y)
test_ds = TensorDataset(test_X.float(), test_y)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# ===== モデル・学習設定 =====
num_classes = len(set(train_y.tolist()))
model = WakewordCNN(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 学習ループ =====
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"🟢 Epoch {epoch+1} | Loss: {total_loss:.4f}")

# ===== 評価 =====
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total * 100
print(f"✅ Accuracy: {correct}/{total} = {accuracy:.2f}%")

# ===== モデル保存 =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/wakeword_cnn_{timestamp}.pth"
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"📦 Saved model to: {model_path}")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os

from tcn_model import TCN_Attention

# =====================================================
# 1. 配置
# =====================================================
X_PATH = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\tcn_input\X.npy"
Y_PATH = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\tcn_input\y.npy"

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH = "tcn_system_attention.pth"

# =====================================================
# 2. Dataset
# =====================================================
class TCNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # TCN 输入格式：(C, T)
        return self.X[idx].transpose(0, 1), self.y[idx]

# =====================================================
# 3. 加载数据（全部）
# =====================================================
X = np.load(X_PATH)
y = np.load(Y_PATH)

dataset = TCNDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=True, drop_last=True)

print(f"Loaded {len(dataset)} sequences for system training")

# =====================================================
# 4. 模型
# =====================================================
model = TCN_Attention(input_dim=36).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================================================
# 5. 训练
# =====================================================
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

    # 保存 system model（以 loss 为准）
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✔ Saved system model to {SAVE_PATH}")

print("\nTraining finished.")
print(f"Best system model saved as: {SAVE_PATH}")

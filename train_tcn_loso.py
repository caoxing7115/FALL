import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd

from tcn_model import TCN_Attention

# ===================== 配置 =====================
DATA_ROOT = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\tcn_input_loso"

EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== Dataset =====================
class TCNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # TCN 输入：(C, T)
        return self.X[idx].transpose(0, 1), self.y[idx]

# ===================== 读取所有 subject =====================
subjects = sorted(
    [d for d in os.listdir(DATA_ROOT) if d.startswith("subject")],
    key=lambda x: int(x[-2:])
)

print(f"Found {len(subjects)} subjects for LOSOCV")

results = []

# ===================== LOSOCV =====================
for test_subject in subjects:
    print(f"\n===== LOSOCV | Test subject: {test_subject} =====")

    # -------- 读取测试集 --------
    test_X = np.load(os.path.join(DATA_ROOT, test_subject, "X.npy"))
    test_y = np.load(os.path.join(DATA_ROOT, test_subject, "y.npy"))

    # -------- 构建训练集（其余 subjects） --------
    train_X_list, train_y_list = [], []

    for subj in subjects:
        if subj == test_subject:
            continue
        train_X_list.append(
            np.load(os.path.join(DATA_ROOT, subj, "X.npy"))
        )
        train_y_list.append(
            np.load(os.path.join(DATA_ROOT, subj, "y.npy"))
        )

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    train_dataset = TCNDataset(train_X, train_y)
    test_dataset = TCNDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===================== 模型 =====================
    model = TCN_Attention(input_dim=36).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ===================== 训练 =====================
    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # ===================== 测试 =====================
    model.eval()
    all_gt, all_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out, _ = model(x)
            pred = out.argmax(dim=1).cpu().numpy()

            all_pred.extend(pred)
            all_gt.extend(y.numpy())

    # ===================== 指标 =====================
    acc = accuracy_score(all_gt, all_pred)
    prec = precision_score(all_gt, all_pred, zero_division=0)
    rec = recall_score(all_gt, all_pred, zero_division=0)
    f1 = f1_score(all_gt, all_pred, zero_division=0)

    res = {
        "subject": test_subject,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1
    }
    print(res)
    results.append(res)

# ===================== 汇总 =====================
df = pd.DataFrame(results)

# 只保留 subject01–30
df_eval = df[df["subject"].apply(lambda x: int(x[-2:]) <= 30)]

print("\n===== LOSOCV Final Results (subject01–30) =====")
print(df_eval)

print("\nMean ± Std:")
print(df_eval[["acc", "prec", "rec", "f1"]].mean())
print(df_eval[["acc", "prec", "rec", "f1"]].std())

df_eval.to_csv("losocv_results_final.csv", index=False)


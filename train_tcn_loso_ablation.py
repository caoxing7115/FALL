import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

from tcn_model import TCN_Attention
from tcn_model_no_attn import TCN_NoAttention

# ================= 配置 =================
DATA_ROOT = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\tcn_input_loso"
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TCNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].transpose(0, 1), self.y[idx]


def run_loso(model_class, model_name):
    subjects = sorted(
        [d for d in os.listdir(DATA_ROOT) if d.startswith("subject")],
        key=lambda x: int(x[-2:])
    )

    results = []

    for test_subject in subjects:
        # load test
        test_X = np.load(os.path.join(DATA_ROOT, test_subject, "X.npy"))
        test_y = np.load(os.path.join(DATA_ROOT, test_subject, "y.npy"))

        train_X, train_y = [], []
        for subj in subjects:
            if subj == test_subject:
                continue
            train_X.append(np.load(os.path.join(DATA_ROOT, subj, "X.npy")))
            train_y.append(np.load(os.path.join(DATA_ROOT, subj, "y.npy")))

        train_X = np.concatenate(train_X, axis=0)
        train_y = np.concatenate(train_y, axis=0)

        train_loader = DataLoader(
            TCNDataset(train_X, train_y),
            batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            TCNDataset(test_X, test_y),
            batch_size=BATCH_SIZE, shuffle=False
        )

        model = model_class(input_dim=36).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # -------- train --------
        model.train()
        for _ in range(EPOCHS):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        # -------- test --------
        model.eval()
        all_gt, all_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                pred = out.argmax(dim=1).cpu().numpy()
                all_pred.extend(pred)
                all_gt.extend(y.numpy())

        results.append({
            "model": model_name,
            "subject": test_subject,
            "acc": accuracy_score(all_gt, all_pred),
            "prec": precision_score(all_gt, all_pred, zero_division=0),
            "rec": recall_score(all_gt, all_pred, zero_division=0),
            "f1": f1_score(all_gt, all_pred, zero_division=0)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_no_attn = run_loso(TCN_NoAttention, "TCN")
    df_attn    = run_loso(TCN_Attention, "TCN + Attention")

    df = pd.concat([df_no_attn, df_attn], ignore_index=True)

    print("\n===== Mean ± Std =====")
    print(df.groupby("model")[["acc", "prec", "rec", "f1"]].mean())
    print(df.groupby("model")[["acc", "prec", "rec", "f1"]].std())

    df.to_csv("ablation_loso_results.csv", index=False)

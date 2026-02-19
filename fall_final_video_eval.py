import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
from tqdm import tqdm

from tcn_model import TCN_Attention

# =====================================================
# 1. 配置
# =====================================================
VIDEO_ROOT = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\urfd_rgb"
GT_CSV = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\urfd_rgb\video_labels.csv"

YOLO_MODEL_PATH = r"best.pt"
TCN_MODEL_PATH = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\tcn_system_attention.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 30
CONF_THRES = 0.5

MIN_FALL_RATIO = 0.5
MIN_FALL_ABS = 10

# =====================================================
# 2. 模型加载
# =====================================================
print("Loading YOLO11-Pose...")
yolo = YOLO(YOLO_MODEL_PATH).to(DEVICE)

print("Loading TCN_Attention...")
tcn = TCN_Attention(input_dim=36, num_classes=2).to(DEVICE)
state_dict = torch.load(TCN_MODEL_PATH, map_location=DEVICE)
tcn.load_state_dict(state_dict)
tcn.eval()

# =====================================================
# 3. 辅助函数
# =====================================================
def extract_keypoints(results):
    if results.keypoints is None:
        return None

    kpts = results.keypoints.xy[0]  # [17, 2]
    if kpts.shape[0] != 17:
        return None

    kpts = kpts.cpu().numpy().reshape(-1)  # [34]

    # ===== 补齐到 36 维（关键）=====
    kpts_36 = np.zeros(36, dtype=np.float32)
    kpts_36[:34] = kpts
    return kpts_36


def normalize_kpts(kpts):
    kpts = kpts.copy()
    xs = kpts[0:34:2]
    ys = kpts[1:34:2]

    xs = (xs - xs.mean()) / (xs.std() + 1e-6)
    ys = (ys - ys.mean()) / (ys.std() + 1e-6)

    kpts[0:34:2] = xs
    kpts[1:34:2] = ys
    return kpts


def tcn_predict(sequence):
    """
    sequence: [T, 36]
    TCN expects: [B, C, T]
    """
    x = (
        torch.tensor(sequence, dtype=torch.float32)
        .unsqueeze(0)          # [1, T, 36]
        .transpose(1, 2)       # [1, 36, T] ✅ 关键
        .to(DEVICE)
    )

    with torch.no_grad():
        out, _ = tcn(x)
        prob = torch.softmax(out, dim=1)[0, 1].item()

    return 1 if prob > CONF_THRES else 0


# =====================================================
# 4. 视频级评估
# =====================================================
gt_df = pd.read_csv(GT_CSV)
video_results = []

for cls in ["falls", "adls"]:
    folder = os.path.join(VIDEO_ROOT, cls)

    for video_name in tqdm(os.listdir(folder), desc=f"{cls}"):
        cap = cv2.VideoCapture(os.path.join(folder, video_name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fall_threshold = max(int(MIN_FALL_RATIO * fps), MIN_FALL_ABS)

        seq_buffer = deque(maxlen=SEQ_LEN)
        consecutive_fall = 0
        max_consecutive_fall = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, conf=0.3, verbose=False)[0]
            if len(results.boxes) == 0:
                seq_buffer.clear()
                consecutive_fall = 0
                continue

            kpts = extract_keypoints(results)
            if kpts is None:
                continue

            kpts = normalize_kpts(kpts)
            seq_buffer.append(kpts)

            is_fall = 0
            if len(seq_buffer) == SEQ_LEN:
                is_fall = tcn_predict(np.array(seq_buffer))

            if is_fall:
                consecutive_fall += 1
                max_consecutive_fall = max(max_consecutive_fall, consecutive_fall)
            else:
                consecutive_fall = 0

        cap.release()

        video_pred = 1 if max_consecutive_fall >= fall_threshold else 0
        video_results.append({"video": video_name, "pred": video_pred})

# =====================================================
# 5. 指标
# =====================================================
df = pd.DataFrame(video_results).merge(gt_df, on="video")

TP = ((df.pred == 1) & (df.label == 1)).sum()
FP = ((df.pred == 1) & (df.label == 0)).sum()
FN = ((df.pred == 0) & (df.label == 1)).sum()
TN = ((df.pred == 0) & (df.label == 0)).sum()

acc = (TP + TN) / (TP + TN + FP + FN)
prec = TP / (TP + FP + 1e-6)
rec = TP / (TP + FN + 1e-6)
f1 = 2 * prec * rec / (prec + rec + 1e-6)

print("\n===== Video-level Evaluation =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

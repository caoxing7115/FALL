import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

# ================== 路径配置 ==================
MODEL_PATH = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best.pt"

URFD_ROOT = Path(r"D:\yolov8-pose-fall-detection\TCN\urfd_classified")
VIDEO_ROOT = URFD_ROOT / "temp_videos"
OUTPUT_ROOT = URFD_ROOT / "tcn_input"

SEQ_LEN = 64
NUM_KPTS = 17
KPT_DIM = NUM_KPTS * 2   # 34
OUT_DIM = 36             # 34 + vel + acc

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ================== 模型 ==================
model = YOLO(MODEL_PATH)

# ================== 工具函数 ==================
def normalize_keypoints(kpts, w, h):
    kpts = kpts.copy()
    kpts[0::2] /= w
    kpts[1::2] /= h
    return np.clip(kpts, 0.0, 1.0)


def pad_or_truncate(seq, target_len=64, dim=34):
    T = seq.shape[0]
    if T >= target_len:
        return seq[:target_len]
    pad = np.zeros((target_len - T, dim), dtype=np.float32)
    return np.vstack([seq, pad])


def compute_motion_features(seq):
    """
    seq: (64, 34)
    """
    vel = np.linalg.norm(seq[1:] - seq[:-1], axis=1)
    vel = np.pad(vel, (1, 0))
    acc = vel[1:] - vel[:-1]
    acc = np.pad(acc, (1, 0))
    return vel.reshape(-1, 1), acc.reshape(-1, 1)


def extract_pose_sequence(video_path):
    cap = cv2.VideoCapture(str(video_path))
    seq = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = model(frame, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            kpts = results.keypoints.xy[0].cpu().numpy().reshape(-1)
            if kpts.shape[0] != KPT_DIM:
                kpts = np.zeros(KPT_DIM, dtype=np.float32)
        else:
            kpts = np.zeros(KPT_DIM, dtype=np.float32)

        kpts = normalize_keypoints(kpts, w, h)
        seq.append(kpts)

    cap.release()

    if len(seq) == 0:
        seq = np.zeros((SEQ_LEN, KPT_DIM), dtype=np.float32)
    else:
        seq = pad_or_truncate(np.stack(seq), SEQ_LEN, KPT_DIM)

    return seq


# ================== 主流程 ==================
X, y = [], []

# -------- FALLS --------
for video in tqdm((VIDEO_ROOT / "falls").glob("*.mp4"), desc="Processing FALLS"):
    pose = extract_pose_sequence(video)
    vel, acc = compute_motion_features(pose)
    fused = np.hstack([pose, vel, acc])

    X.append(fused)
    y.append(1)

# -------- ADLS --------
for video in tqdm((VIDEO_ROOT / "adls").glob("*.mp4"), desc="Processing ADLS"):
    pose = extract_pose_sequence(video)
    vel, acc = compute_motion_features(pose)
    fused = np.hstack([pose, vel, acc])

    X.append(fused)
    y.append(0)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

np.save(OUTPUT_ROOT / "X.npy", X)
np.save(OUTPUT_ROOT / "y.npy", y)

print("=" * 60)
print("TCN 数据构建完成")
print(f"X shape: {X.shape}  # (N, 64, 36)")
print(f"y shape: {y.shape}")

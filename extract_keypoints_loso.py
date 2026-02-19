import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

# ================== 配置 ==================
MODEL_PATH = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best.pt"

URFD_ROOT = Path(r"D:\yolov8-pose-fall-detection\TCN\urfd_classified")
VIDEO_ROOT = URFD_ROOT / "temp_videos"
OUTPUT_ROOT = URFD_ROOT / "tcn_input_loso"

SEQ_LEN = 64
NUM_KPTS = 17
KPT_DIM = NUM_KPTS * 2     # 34
OUT_DIM = 36              # 34 + vel + acc

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

# ================== 工具函数 ==================
def normalize_keypoints(kpts, w, h):
    kpts = kpts.copy()
    kpts[0::2] /= w
    kpts[1::2] /= h
    return np.clip(kpts, 0.0, 1.0)

def pad_or_truncate(seq, target_len=SEQ_LEN, dim=KPT_DIM):
    T = seq.shape[0]
    if T >= target_len:
        return seq[:target_len]
    pad = np.zeros((target_len - T, dim), dtype=np.float32)
    return np.vstack([seq, pad])

def compute_motion_features(seq):
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

        if results.keypoints is not None and results.keypoints.xy is not None:
            kpts_all = results.keypoints.xy.cpu().numpy()  # (N,17,2)

            if kpts_all.shape[0] > 0:
                kpts = kpts_all[0].reshape(-1)
                if kpts.shape[0] != KPT_DIM:
                    kpts = np.zeros(KPT_DIM, dtype=np.float32)
            else:
                kpts = np.zeros(KPT_DIM, dtype=np.float32)
        else:
            kpts = np.zeros(KPT_DIM, dtype=np.float32)

        kpts = normalize_keypoints(kpts, w, h)
        seq.append(kpts)

    cap.release()

    if len(seq) == 0:
        seq = np.zeros((SEQ_LEN, KPT_DIM), dtype=np.float32)
    else:
        seq = pad_or_truncate(np.array(seq), SEQ_LEN, KPT_DIM)

    return seq


def parse_subject(video_name):
    # fall-01_xxx 或 adl-01_xxx
    prefix = video_name.split("_")[0]   # fall-01 / adl-01
    sid = prefix.split("-")[1]
    return f"subject{sid}"

# ================== 主流程 ==================
subject_data = {}

for label_name, label in [("falls", 1), ("adls", 0)]:
    folder = VIDEO_ROOT / label_name
    for video in tqdm(folder.glob("*.mp4"), desc=f"Processing {label_name.upper()}"):
        subject = parse_subject(video.name)

        if subject not in subject_data:
            subject_data[subject] = {"X": [], "y": []}

        pose = extract_pose_sequence(video)
        vel, acc = compute_motion_features(pose)
        fused = np.hstack([pose, vel, acc])

        subject_data[subject]["X"].append(fused)
        subject_data[subject]["y"].append(label)

# ================== 保存 ==================
for subject, data in subject_data.items():
    out_dir = OUTPUT_ROOT / subject
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "X.npy", np.asarray(data["X"], dtype=np.float32))
    np.save(out_dir / "y.npy", np.asarray(data["y"], dtype=np.int64))

print("=" * 60)
print("LOSOCV 数据构建完成")
print(f"Subjects: {len(subject_data)}")

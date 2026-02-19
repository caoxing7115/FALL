import os
import csv

VIDEO_ROOT = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\urfd_rgb"
SAVE_PATH = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\urfd_rgb\video_labels.csv"

rows = [("video", "label")]

# Fall videos
fall_dir = os.path.join(VIDEO_ROOT, "falls")
for name in sorted(os.listdir(fall_dir)):
    if name.endswith(".mp4"):
        rows.append((name, 1))

# ADL videos
adl_dir = os.path.join(VIDEO_ROOT, "adls")
for name in sorted(os.listdir(adl_dir)):
    if name.endswith(".mp4"):
        rows.append((name, 0))

with open(SAVE_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Saved video_labels.csv to: {SAVE_PATH}")

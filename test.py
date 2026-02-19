import cv2
import numpy as np
from ultralytics import YOLO

# 骨架连接关系
SKELETON_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# 加载模型
model = YOLO("best.pt")  # 或 yolov8n-pose.pt

# 读取单张图像
image_path = r'D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\datasets\train\images\000000014537.jpg'
frame = cv2.imread(image_path)

# 推理
results = model(frame)

for result in results:
    if not hasattr(result, "keypoints") or result.keypoints is None:
        continue

    keypoints = result.keypoints.data[0].cpu().numpy()  # shape: (17, 3)

    # 画关键点
    for x, y, conf in keypoints:
        if conf > 0.5:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # 画骨架线
    for a, b in SKELETON_PAIRS:
        if keypoints[a][2] > 0.5 and keypoints[b][2] > 0.5:
            x1, y1 = map(int, keypoints[a][:2])
            x2, y2 = map(int, keypoints[b][:2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 保存标注后的图像
cv2.imwrite("D:/yolov8-pose-fall-detection/annotated_image.jpg", frame)
cv2.imshow("Pose Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

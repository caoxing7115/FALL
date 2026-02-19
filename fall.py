import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# 预定义骨骼连接对
SKELETON_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# 跌倒检测函数：膝盖高于臀部为跌倒
def is_falling_v5(keypoints, keypoints_history, min_fall_frames=5):
    if keypoints is None or len(keypoints) != 17:
        return False

    keypoints = np.asarray(keypoints)
    critical_indices = [11, 12, 13, 14]  # 臀部+膝盖

    if np.any(keypoints[critical_indices, 2] < 0.5):
        return False

    hips_center = np.mean(keypoints[11:13, :2], axis=0)
    knees_center = np.mean(keypoints[13:15, :2], axis=0)

    is_fall_condition = knees_center[1] < hips_center[1]  # y小说明膝盖更高
    keypoints_history.append(is_fall_condition)

    return sum(keypoints_history) >= min_fall_frames

# 初始化模型
model = YOLO("best.pt")  # 替换为你的模型路径

# 摄像头或视频流（手机摄像头必须使用 /video）
# cap = cv2.VideoCapture("http://192.168.186.8:8080/video")
cap = cv2.VideoCapture(r"D:\618\视频\视频10 21\手机QQ视频_20251021124624.mp4")
if not cap.isOpened():
    print("❌ 视频流打开失败，请检查 URL 或网络")
    exit()

# 保存输出视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/yolov8-pose-fall-detection/output1021.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

keypoints_history = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 视频帧读取失败或已结束")
        break

    results = model(frame)
    if not results or not hasattr(results[0], "keypoints") or results[0].keypoints is None:
        continue

    result = results[0]
    keypoints_data = result.keypoints.data

    # 安全性判断：无关键点数据
    if keypoints_data is None or keypoints_data.shape[0] == 0:
        continue

    keypoints_list = keypoints_data[0].cpu().numpy()

    # 如果不是17个关键点，跳过
    if keypoints_list.shape[0] != 17:
        print(f"⚠️ Detected invalid keypoints shape: {keypoints_list.shape}")
        continue

    # 检测框处理
    if result.boxes is not None:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, score = box[:5]
            if score > 0.5:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                falling = is_falling_v5(keypoints_list, keypoints_history)

                color = (0, 0, 255) if falling else (0, 255, 0)
                label = "Falling!" if falling else f"Person {score:.2%}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 关键点圆点绘制
    for kp in keypoints_list:
        if kp[2] > 0.5:
            cv2.circle(frame, tuple(kp[:2].astype(int)), 5, (0, 255, 0), -1)

    # 骨骼连接线
    for a, b in SKELETON_PAIRS:
        if a >= len(keypoints_list) or b >= len(keypoints_list):
            continue  # 防止越界
        kp_a, kp_b = keypoints_list[a], keypoints_list[b]
        if kp_a[2] > 0.5 and kp_b[2] > 0.5:
            x1, y1 = map(int, kp_a[:2])
            x2, y2 = map(int, kp_b[:2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 显示 & 保存
    out.write(frame)
    cv2.imshow('Fall Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import time

# 预定义骨骼连接对（按照COCO格式关键点索引）
SKELETON_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# 初始化模型
model = YOLO("best.pt")  # 替换为你的姿态估计模型路径

# 打开视频流或摄像头
# cap = cv2.VideoCapture("http://192.168.186.8:8080/video")
cap = cv2.VideoCapture(r"D:\618\视频\视频10 21\手机QQ视频_20251021124624.mp4")
if not cap.isOpened():
    print("❌ 视频流打开失败，请检查路径或网络")
    exit()

# 设置视频保存
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/yolov8-pose-fall-detection/output1022-2.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

# 历史帧队列，用于稳定性判断
max_history = 5
knee_history = deque(maxlen=max_history)        # 膝盖高于臀部
horizontal_history = deque(maxlen=max_history)  # 身体水平（宽>高）
torso_history = deque(maxlen=max_history)       # 躯干接近水平

fall_detected = False  # 用于只记录新跌倒事件
fall_event_count = 0   # 跌倒事件计数

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
    if keypoints_data is None or keypoints_data.shape[0] == 0:
        continue

    # 仅取第一个检测到的人作为目标（不需要多人检测）
    keypoints_list = keypoints_data[0].cpu().numpy()  # 形状为 (17, 3)
    if keypoints_list.shape[0] != 17:
        # 如果关键点不全，跳过
        continue

    # 提取关节点坐标
    points = keypoints_list[:, :2]
    confs = keypoints_list[:, 2]

    # 膝盖高于臀部条件
    knee_cond = False
    # indices: 左臀11, 右臀12, 左膝13, 右膝14
    if (confs[11] > 0.5 and confs[12] > 0.5 and
        confs[13] > 0.5 and confs[14] > 0.5):
        hips_center_y = np.mean([keypoints_list[11][1], keypoints_list[12][1]])
        knees_center_y = np.mean([keypoints_list[13][1], keypoints_list[14][1]])
        # 坐标系：y轴向下，膝盖y值 < 臀部y值 表明膝盖位置更靠上（实际人跌坐或跪下）
        if knees_center_y < hips_center_y:
            knee_cond = True
    knee_history.append(knee_cond)

    # 身体边界框宽高比条件
    horizontal_cond = False
    # 取所有置信度高的关键点计算边界盒
    valid_pts = points[confs > 0.5]
    if valid_pts.shape[0] > 0:
        x_min, y_min = np.min(valid_pts, axis=0)
        x_max, y_max = np.max(valid_pts, axis=0)
        body_height = y_max - y_min
        body_width = x_max - x_min
        # 如果宽度大于高度（身体横着），则可能跌倒
        if body_width > body_height:
            horizontal_cond = True
    horizontal_history.append(horizontal_cond)

    # 躯干倾斜角度条件
    torso_cond = False
    # 计算肩部中心与臀部中心连线与水平的夹角
    # indices: 左肩5, 右肩6, 左臀11, 右臀12
    if confs[5] > 0.5 and confs[6] > 0.5 and confs[11] > 0.5 and confs[12] > 0.5:
        shoulder_center = np.mean([points[5], points[6]], axis=0)
        hip_center = np.mean([points[11], points[12]], axis=0)
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = abs(angle_rad * 180 / math.pi)
        # 角度接近0或180度时，躯干接近水平
        if angle_deg < 45 or angle_deg > 135:
            torso_cond = True
    torso_history.append(torso_cond)

    # 综合判断跌倒：任一条件在连续多帧出现
    fall_threshold = 3  # 连续帧阈值
    falling = False
    if (sum(knee_history) >= fall_threshold or
        sum(horizontal_history) >= fall_threshold or
        sum(torso_history) >= fall_threshold):
        falling = True

    # 处理检测框和标注
    if result.boxes is not None:
        boxes = result.boxes.data.cpu().numpy()
        if boxes.shape[0] > 0:
            # 选取置信度最高的检测框
            best_box = boxes[np.argmax(boxes[:, 4])]
            x1, y1, x2, y2, score = best_box[:5]
            if score > 0.5:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                color = (0, 0, 255) if falling else (0, 255, 0)
                label = "Falling!" if falling else f"Person {score:.2%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 绘制关键点和骨骼连线
    for kp in keypoints_list:
        if kp[2] > 0.5:
            cv2.circle(frame, tuple(kp[:2].astype(int)), 3, (0, 255, 0), -1)
    for a, b in SKELETON_PAIRS:
        if a < 17 and b < 17:
            kp_a, kp_b = keypoints_list[a], keypoints_list[b]
            if kp_a[2] > 0.5 and kp_b[2] > 0.5:
                x1, y1 = map(int, kp_a[:2])
                x2, y2 = map(int, kp_b[:2])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 当检测到新跌倒事件时，记录日志和保存图片
    if falling and not fall_detected:
        fall_event_count += 1
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_s = timestamp_ms / 1000.0
        log_msg = f"Event {fall_event_count}: Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, time {timestamp_s:.2f}s - Fall detected\n"
        print(log_msg.strip())
        with open("fall_log.txt", "a") as log_f:
            log_f.write(log_msg)
        # 保存当前帧为图片
        img_name = f"fall_event_{fall_event_count}.jpg"
        cv2.imwrite(img_name, frame)
        fall_detected = True
    elif not falling:
        # 重置跌倒状态，准备检测下一次事件
        fall_detected = False

    # 写入输出视频并显示
    out.write(frame)
    cv2.imshow('Fall Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

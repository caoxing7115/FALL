import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import torch
import torch.nn as nn


# ================== 1. TCN 模型定义 (新版) ==================
# 这是你提供的 TCN 代码结构
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.relu(out)
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim=35, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_dim, 64, 3, dilation=1),
            TemporalBlock(64, 128, 3, dilation=2),
            TemporalBlock(128, 128, 3, dilation=4),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = x.mean(dim=2)
        return self.fc(x)


# ================== 2. SimpleCNN 模型定义 (旧版兼容) ==================
# 这是根据你的报错信息反推出来的模型结构，为了匹配 best_tcn.pth
class SimpleCNN(nn.Module):
    def __init__(self, input_dim=35, num_classes=2):
        super().__init__()
        # 结构反推：conv1.0(Conv) -> conv1.1(BN) -> ReLU
        # 你的报错显示 fc 输入是 256，所以最后一层卷积输出通道必然是 256
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)  # Global Average Pooling
        return self.fc(x)


# ================== 初始化配置 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE}")

# 1. 加载 YOLO 模型
# 确保 best.pt 在当前目录下
model_yolo = YOLO("best.pt")

# 2. 智能加载分类模型 (TCN 或 CNN)
MODEL_PATH = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best_tcn.pth"
tcn_model = None
model_type = "Unknown"

try:
    print(f"正在尝试加载模型: {MODEL_PATH} ...")

    # 尝试加载为 TCN
    model_candidate = TCN(input_dim=35, num_classes=2).to(DEVICE)
    model_candidate.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    tcn_model = model_candidate
    model_type = "TCN (Advanced)"
    print("✅ 成功识别为 TCN 模型结构")

except Exception as e_tcn:
    print(f"⚠️ 不是 TCN 结构，尝试作为 SimpleCNN 加载... (错误: {e_tcn})")
    try:
        # 尝试加载为 SimpleCNN
        model_candidate = SimpleCNN(input_dim=35, num_classes=2).to(DEVICE)
        model_candidate.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        tcn_model = model_candidate
        model_type = "SimpleCNN (Compatible)"
        print("✅ 成功识别为 SimpleCNN 模型结构 (兼容旧权重)")

    except Exception as e_cnn:
        print(f"❌ 模型加载彻底失败。请检查 .pth 文件是否损坏或对应代码版本。")
        print(f"错误详情: {e_cnn}")
        # 如果没有模型，程序虽然继续运行但无法进行时序检测
        pass

if tcn_model:
    tcn_model.eval()

# 参数配置
WINDOW_SIZE = 64
history_buffer = deque(maxlen=WINDOW_SIZE)

# 骨骼连接 (用于绘图)
SKELETON_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# 视频输入
video_path = r"D:\618\视频\视频10 21\手机QQ视频_20251021124624.mp4"
# video_path = 0 # 如果想用摄像头，解开这行
cap = cv2.VideoCapture(video_path)

# 历史状态队列 (原逻辑)
max_history = 5
knee_history = deque(maxlen=max_history)
horizontal_history = deque(maxlen=max_history)
torso_history = deque(maxlen=max_history)
fall_detected = False
fall_event_count = 0


# 特征预处理函数
def preprocess_keypoints(keypoints, img_w, img_h):
    # 归一化坐标
    pts = keypoints[:, :2]
    pts[:, 0] = pts[:, 0] / img_w
    pts[:, 1] = pts[:, 1] / img_h
    features = pts.flatten()  # 34维

    # 补充第35维 (这里用平均置信度)
    avg_conf = np.mean(keypoints[:, 2])
    features = np.append(features, avg_conf)

    return features.astype(np.float32)


print("开始检测...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    img_h, img_w = frame.shape[:2]
    results = model_yolo(frame, verbose=False)

    has_person = False

    # YOLO 检测处理
    if results and results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
        has_person = True
        keypoints_list = results[0].keypoints.data[0].cpu().numpy()  # 取第一个人

        # === 1. 数据收集 ===
        if keypoints_list.shape[0] == 17:
            feat = preprocess_keypoints(keypoints_list, img_w, img_h)
            history_buffer.append(feat)

        # === 2. 传统规则逻辑 ===
        points = keypoints_list[:, :2]
        confs = keypoints_list[:, 2]

        # 规则1: 膝盖
        knee_cond = False
        if confs[11] > 0.5 and confs[12] > 0.5 and confs[13] > 0.5 and confs[14] > 0.5:
            hips_y = np.mean([keypoints_list[11][1], keypoints_list[12][1]])
            knees_y = np.mean([keypoints_list[13][1], keypoints_list[14][1]])
            if knees_y < hips_y: knee_cond = True
        knee_history.append(knee_cond)

        # 规则2: 宽高比
        horizontal_cond = False
        valid_pts = points[confs > 0.5]
        if valid_pts.shape[0] > 0:
            box_h = np.max(valid_pts[:, 1]) - np.min(valid_pts[:, 1])
            box_w = np.max(valid_pts[:, 0]) - np.min(valid_pts[:, 0])
            if box_w > box_h: horizontal_cond = True
        horizontal_history.append(horizontal_cond)

        # 规则3: 角度
        torso_cond = False
        if confs[5] > 0.5 and confs[6] > 0.5 and confs[11] > 0.5 and confs[12] > 0.5:
            sh_c = np.mean(points[5:7], axis=0)
            hip_c = np.mean(points[11:13], axis=0)
            dx, dy = hip_c[0] - sh_c[0], hip_c[1] - sh_c[1]
            angle = abs(math.atan2(dy, dx) * 180 / math.pi)
            if angle < 45 or angle > 135: torso_cond = True
        torso_history.append(torso_cond)

        # === 绘图 ===
        for kp in keypoints_list:
            if kp[2] > 0.5: cv2.circle(frame, tuple(kp[:2].astype(int)), 3, (0, 255, 0), -1)
        for a, b in SKELETON_PAIRS:
            if keypoints_list[a][2] > 0.5 and keypoints_list[b][2] > 0.5:
                cv2.line(frame, tuple(keypoints_list[a][:2].astype(int)),
                         tuple(keypoints_list[b][:2].astype(int)), (255, 0, 0), 2)

    else:
        # 无人时填充0，保持时间序列连续性
        if len(history_buffer) > 0:
            history_buffer.append(np.zeros(35, dtype=np.float32))

    # === 3. 时序模型推理 (TCN/CNN) ===
    tcn_label = "Buffering..."
    tcn_alert = False

    if tcn_model and len(history_buffer) == WINDOW_SIZE:
        input_np = np.array(history_buffer)
        # (Batch=1, Time=64, Channels=35) -> 变为 (1, 35, 64)
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).permute(0, 2, 1).to(DEVICE)

        with torch.no_grad():
            output = tcn_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_cls = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_cls].item()

        # 假设 1 是跌倒类别
        if pred_cls == 1:
            tcn_label = f"FALL ({conf:.2f})"
            tcn_alert = True
        else:
            tcn_label = f"Normal ({conf:.2f})"

    # === 4. 综合展示 ===
    # 规则结果
    rule_fall = (sum(knee_history) >= 3 or sum(horizontal_history) >= 3 or sum(torso_history) >= 3)

    # 界面绘制
    cv2.putText(frame, f"Model: {model_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 规则状态
    cv2.putText(frame, f"Rule: {'FALL' if rule_fall else 'Normal'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if rule_fall else (0, 255, 0), 2)

    # 时序模型状态
    cv2.putText(frame, f"AI: {tcn_label}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if tcn_alert else (255, 255, 0), 2)

    # 简单的事件记录
    if (rule_fall or tcn_alert) and not fall_detected:
        print(f"⚠️ Fall Detected! (Rule: {rule_fall}, AI: {tcn_alert})")
        fall_detected = True
    elif not rule_fall and not tcn_alert:
        fall_detected = False

    cv2.imshow('Fall Detection System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
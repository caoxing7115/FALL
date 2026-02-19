import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from ultralytics import YOLO
import math
import os

# ----------------- 配置区 -----------------
VIDEO_SOURCE = r"D:\yolov8-pose-fall-detection\video_1.mp4"
POSE_WEIGHTS = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best.pt"
TCN_WEIGHTS = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best_tcn_attention.pth"
SEQ_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------


# ==================== TCN + Attention ====================
class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, x):
        return x[:, :, :-self.size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return out + res


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class TCN_Attention(nn.Module):
    def __init__(self, input_dim=36, num_classes=2):
        super().__init__()
        channels = [64, 128, 256]
        layers = []
        for i, ch in enumerate(channels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, ch, 3, 2 ** i))
        self.tcn = nn.Sequential(*layers)
        self.attn = TemporalAttention(channels[-1])
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        feat = self.tcn(x)
        pooled = self.attn(feat)
        return self.fc(pooled)


# ==================== FallDetector ====================
class FallDetector:
    def __init__(self, pose_weights, tcn_weights, device="cpu", seq_len=64):
        self.device = device
        self.seq_len = seq_len

        self.pose_model = YOLO(pose_weights)

        self.tcn = TCN_Attention(input_dim=36).to(device)
        state = torch.load(tcn_weights, map_location=device)
        self.tcn.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        self.tcn.eval()

        self.buffer = deque(maxlen=seq_len)

        self.state = "NORMAL"
        self.trigger = 0
        self.recover_counter = 0
        self.kneeling_counter = 0

        self.angle_buf = deque(maxlen=5)
        self.vel_buf = deque(maxlen=5)

        self.prev_head = None

        self.skeleton = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 6), (11, 12), (5, 11), (6, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def normalize(self, kps, box):
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        feat = []
        for x, y in kps:
            feat.extend([(x - cx) / w, (y - cy) / h])
        return np.array(feat, dtype=np.float32)

    def body_state(self, kps, prev_hip_y):
        shoulder = (kps[5] + kps[6]) / 2
        hip = (kps[11] + kps[12]) / 2
        dx = abs(shoulder[0] - hip[0])
        dy = abs(shoulder[1] - hip[1])
        angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 90
        vel = hip[1] - prev_hip_y if prev_hip_y is not None else 0
        return angle, vel, hip[1]

    def draw(self, frame, kps, box, fall):
        h, w = frame.shape[:2]
        color = (0, 0, 255) if fall else (0, 255, 0)

        # bbox
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        def valid(p):
            x, y = int(p[0]), int(p[1])
            if x <= 0 or y <= 0:
                return False
            if x >= w or y >= h:
                return False
            return True

        # keypoints
        for x, y in kps:
            if valid((x, y)):
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        # skeleton
        for a, b in self.skeleton:
            if a < len(kps) and b < len(kps):
                if valid(kps[a]) and valid(kps[b]):
                    cv2.line(
                        frame,
                        (int(kps[a][0]), int(kps[a][1])),
                        (int(kps[b][0]), int(kps[b][1])),
                        color, 2
                    )

    def step(self, frame, prev_hip_y):
        res = self.pose_model(frame, verbose=False)[0]
        if res.boxes is None or res.keypoints is None:
            return frame, False, 0.0, "No person", prev_hip_y

        boxes = res.boxes.xyxy.cpu().numpy()
        kps_all = res.keypoints.xy.cpu().numpy()
        idx = np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        box = boxes[idx]
        kps = kps_all[idx]

        angle, vel, hip_y = self.body_state(kps, prev_hip_y)
        self.angle_buf.append(angle)
        self.vel_buf.append(vel)
        angle_s = np.mean(self.angle_buf)
        vel_s = np.mean(self.vel_buf)

        head = kps[0]
        head_vel = 0
        if self.prev_head is not None:
            head_vel = np.linalg.norm(head - self.prev_head)
        self.prev_head = head

        is_kneeling = (kps[13][1] < kps[11][1]) or (kps[14][1] < kps[12][1])

        feat = np.zeros(36, np.float32)
        feat[:34] = self.normalize(kps, box)
        self.buffer.append(feat)
        while len(self.buffer) < self.seq_len:
            self.buffer.append(feat)

        tcn_prob = 0
        if len(self.buffer) == self.seq_len:
            inp = torch.tensor(np.array(self.buffer).T).unsqueeze(0).to(self.device)
            with torch.no_grad():
                tcn_prob = F.softmax(self.tcn(inp), dim=1)[0, 1].item()

        # ================= 状态机 =================
        if self.state == "FALL":
            if angle_s < 20 and abs(vel_s) < 1:
                self.recover_counter += 1
                if self.recover_counter > 20:
                    self.state = "NORMAL"
                    self.recover_counter = 0
            else:
                self.recover_counter = 0
            self.draw(frame, kps, box, True)
            return frame, True, tcn_prob, "FALL (hold)", hip_y

        # kneeling → 跌倒候选
        if is_kneeling:
            self.kneeling_counter += 1
        else:
            self.kneeling_counter = 0

        if is_kneeling and self.kneeling_counter >= 3 and vel_s > 4:
            self.state = "FALL"
            self.draw(frame, kps, box, True)
            return frame, True, tcn_prob, "FALL (kneeling)", hip_y

        # 正常 TCN 路径
        score = tcn_prob
        if vel_s > 8 and angle_s > 45:
            score += 0.15
        if head_vel > 5:
            score += 0.1

        threshold = 0.85
        if angle_s < 40:
            threshold = 0.9

        if score > threshold:
            self.trigger += 1
        else:
            self.trigger = max(0, self.trigger - 1)

        if self.trigger >= 4:
            self.state = "FALL"
            self.trigger = 0
            self.draw(frame, kps, box, True)
            return frame, True, tcn_prob, "FALL", hip_y

        self.draw(frame, kps, box, False)
        return frame, False, tcn_prob, "Normal", hip_y


# ==================== main ====================
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    detector = FallDetector(POSE_WEIGHTS, TCN_WEIGHTS, DEVICE, SEQ_LEN)
    prev_hip_y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, fall, prob, msg, prev_hip_y = detector.step(frame, prev_hip_y)

        cv2.putText(frame, f"{msg} | TCN={prob:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if fall else (0, 255, 0), 2)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

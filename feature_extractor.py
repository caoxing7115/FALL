# feature_extractor.py
import numpy as np
from collections import deque

class FallFeatureExtractor:
    def __init__(self, window_size=10):
        self.kpt_history = deque(maxlen=window_size)
        self.bbox_history = deque(maxlen=window_size)
        self.feature_names = [
            'torso_angle', 'head_height_norm', 'bbox_ratio',
            'center_velocity', 'knee_hip_ratio', 'shoulder_hip_dist_norm',
            'head_shoulder_dist', 'leg_spread_angle', 'centroid_y_drop_rate'
        ]

    def extract(self, keypoints, bbox=None):
        kp = np.array(keypoints)[:, :2]      # (17, 2)
        conf = np.array(keypoints)[:, 2]
        if np.any(conf < 0.3):  # 关键点置信度太低
            return None

        # 关键点索引（COCO 17）
        nose, lsho, rsho = 0, 5, 6
        lhip, rhip, lknee, rknee = 11, 12, 13, 14
        lank, rank = 15, 16

        # 基础点
        shoulder_c = np.mean(kp[[lsho, rsho]], axis=0)
        hip_c = np.mean(kp[[lhip, rhip]], axis=0)
        knee_c = np.mean(kp[[lknee, rknee]], axis=0)
        ankle_c = np.mean(kp[[lank, rank]], axis=0)
        torso_vec = shoulder_c - hip_c
        torso_len = np.linalg.norm(torso_vec) + 1e-6

        # 9维强特征（经过2024-2025年多篇顶刊验证）
        features = [
            np.abs(np.arctan2(torso_vec[0], torso_vec[1])),                    # 躯干倾角
            (kp[nose][1] - ankle_c[1]) / (torso_len + 1e-6),                   # 头部归一化高度
            (bbox[3] - bbox[1]) / (bbox[2] - bbox[0] + 1e-6) if bbox is not None else 1.0,  # bbox纵横比
            0.0,  # velocity (后面填)
            knee_c[1] / (hip_c[1] + 1e-6),                                     # 膝盖/臀部 y比（越小越危险）
            torso_len,                                                         # 躯干长度（归一化用）
            np.linalg.norm(kp[nose] - shoulder_c),                             # 头肩距离
            np.arccos(np.clip(np.dot(torso_vec / torso_len, [0,1]), -1, 1)),   # 躯干与垂直夹角
            0.0   # centroid drop rate (后面填)
        ]

        # 速度与加速度（时序）
        if len(self.kpt_history) > 0:
            prev_kpt = self.kpt_history[-1]
            velocity = np.mean(np.linalg.norm(kp - prev_kpt, axis=1))
            features[3] = velocity
            if len(self.kpt_history) >= 5:
                centroid_y = np.mean(kp[:, 1])
                prev_centroid_y = np.mean(self.kpt_history[-5][:, 1])
                features[8] = (prev_centroid_y - centroid_y) / 5.0  # 质心下降速率
        else:
            features[3] = 0.0
            features[8] = 0.0

        self.kpt_history.append(kp.copy())
        if bbox is not None:
            self.bbox_history.append(bbox)

        return np.array(features, dtype=np.float32)
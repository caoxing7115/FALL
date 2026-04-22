import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
import math
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# 全局配置（与原代码保持一致）
# =====================================================
VIDEO_ROOT = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\urfd_rgb"
GT_CSV = r"D:\yolov8-pose-fall-detection\TCN\urfd_classified\urfd_rgb\video_labels.csv"
YOLO_MODEL_PATH = r"best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 核心参数
SKELETON_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]
MAX_HISTORY = 5  # 历史帧队列长度
FALL_THRESHOLD = 3  # 连续帧触发跌倒的阈值
CONF_THRESHOLD = 0.5  # 关键点置信度阈值

# 加载YOLO Pose模型
print("Loading YOLO Pose Model...")
model = YOLO(YOLO_MODEL_PATH).to(DEVICE)

# =====================================================
# 消融实验配置定义
# =====================================================
# 定义所有消融组合（key: 实验名称, value: (是否启用膝盖条件, 是否启用水平条件, 是否启用躯干条件)）
ABLATION_CONFIGS = {
    # 单条件消融
    "only_knee": (True, False, False),
    "only_horizontal": (False, True, False),
    "only_torso": (False, False, True),
    # 两两组合消融
    "knee+horizontal": (True, True, False),
    "knee+torso": (True, False, True),
    "horizontal+torso": (False, True, True),
    # 三者组合（原算法）
    "original_all": (True, True, True)
}

# 对比算法名称
COMPARE_ALGORITHMS = ["baseline_pose_only", "baseline_centroid_height", "baseline_skeleton_angle"]


# =====================================================
# 核心检测函数（不同算法/消融配置）
# =====================================================
def detect_fall_ablation(frame, model, knee_history, horizontal_history, torso_history, use_knee, use_horizontal,
                         use_torso):
    """
    消融实验专用检测函数（可配置启用/禁用不同条件）
    """
    results = model(frame, conf=0.3, verbose=False)[0]

    # 检查关键点是否存在
    if not hasattr(results, "keypoints") or results.keypoints is None:
        return False, False

    keypoints_data = results.keypoints.data
    if keypoints_data is None or keypoints_data.shape[0] == 0:
        return False, False

    # 仅取第一个检测到的人
    keypoints_list = keypoints_data[0].cpu().numpy()
    if keypoints_list.shape[0] != 17:
        return False, False

    points = keypoints_list[:, :2]
    confs = keypoints_list[:, 2]

    # 1. 膝盖高于臀部条件（可选启用）
    knee_cond = False
    if use_knee:
        if (confs[11] > CONF_THRESHOLD and confs[12] > CONF_THRESHOLD and
                confs[13] > CONF_THRESHOLD and confs[14] > CONF_THRESHOLD):
            hips_center_y = np.mean([keypoints_list[11][1], keypoints_list[12][1]])
            knees_center_y = np.mean([keypoints_list[13][1], keypoints_list[14][1]])
            if knees_center_y < hips_center_y:
                knee_cond = True
    knee_history.append(knee_cond)

    # 2. 身体边界框宽高比条件（可选启用）
    horizontal_cond = False
    if use_horizontal:
        valid_pts = points[confs > CONF_THRESHOLD]
        if valid_pts.shape[0] > 0:
            x_min, y_min = np.min(valid_pts, axis=0)
            x_max, y_max = np.max(valid_pts, axis=0)
            body_height = y_max - y_min
            body_width = x_max - x_min
            if body_width > body_height:
                horizontal_cond = True
    horizontal_history.append(horizontal_cond)

    # 3. 躯干倾斜角度条件（可选启用）
    torso_cond = False
    if use_torso:
        if (confs[5] > CONF_THRESHOLD and confs[6] > CONF_THRESHOLD and
                confs[11] > CONF_THRESHOLD and confs[12] > CONF_THRESHOLD):
            shoulder_center = np.mean([points[5], points[6]], axis=0)
            hip_center = np.mean([points[11], points[12]], axis=0)
            dx = hip_center[0] - shoulder_center[0]
            dy = hip_center[1] - shoulder_center[1]
            angle_rad = math.atan2(dy, dx)
            angle_deg = abs(angle_rad * 180 / math.pi)
            if angle_deg < 45 or angle_deg > 135:
                torso_cond = True
    torso_history.append(torso_cond)

    # 综合判断：启用的条件中任一满足阈值帧
    falling = False
    active_conditions = []
    if use_knee:
        active_conditions.append(sum(knee_history) >= FALL_THRESHOLD)
    if use_horizontal:
        active_conditions.append(sum(horizontal_history) >= FALL_THRESHOLD)
    if use_torso:
        active_conditions.append(sum(torso_history) >= FALL_THRESHOLD)

    if active_conditions and any(active_conditions):
        falling = True

    return falling, True


def detect_fall_baseline_pose_only(frame, model, *args):
    """
    基线1：仅Pose检测（无跌倒判断，始终返回False）
    用于验证纯姿态检测的基线性能
    """
    results = model(frame, conf=0.3, verbose=False)[0]
    has_keypoints = hasattr(results, "keypoints") and results.keypoints is not None
    return False, has_keypoints  # 始终不检测跌倒


def detect_fall_baseline_centroid_height(frame, model, centroid_history, *args):
    """
    基线2：基于人体质心高度的跌倒检测（主流方法）
    原理：质心高度骤降超过阈值判定为跌倒
    """
    results = model(frame, conf=0.3, verbose=False)[0]

    # 检查关键点
    if not hasattr(results, "keypoints") or results.keypoints is None:
        return False, False

    keypoints_data = results.keypoints.data
    if keypoints_data is None or keypoints_data.shape[0] == 0:
        return False, False

    keypoints_list = keypoints_data[0].cpu().numpy()
    if keypoints_list.shape[0] != 17:
        return False, False

    confs = keypoints_list[:, 2]
    valid_pts = keypoints_list[confs > CONF_THRESHOLD]
    if valid_pts.shape[0] < 5:  # 至少5个有效关键点
        return False, False

    # 计算人体质心（所有有效关键点的均值）
    centroid_y = np.mean(valid_pts[:, 1])
    centroid_history.append(centroid_y)

    # 质心高度骤降判断：当前高度比历史平均低30%以上
    falling = False
    if len(centroid_history) >= MAX_HISTORY:
        avg_centroid = np.mean(list(centroid_history)[:-1])
        if centroid_y < avg_centroid * 0.7:  # 高度下降30%
            falling = True

    return falling, True


def detect_fall_baseline_skeleton_angle(frame, model, angle_history, *args):
    """
    基线3：传统骨骼角度跌倒检测（头部-躯干-腿部角度）
    原理：头部与臀部的垂直角度 + 腿部与躯干的角度综合判断
    """
    results = model(frame, conf=0.3, verbose=False)[0]

    # 检查关键点
    if not hasattr(results, "keypoints") or results.keypoints is None:
        return False, False

    keypoints_data = results.keypoints.data
    if keypoints_data is None or keypoints_data.shape[0] == 0:
        return False, False

    keypoints_list = keypoints_data[0].cpu().numpy()
    if keypoints_list.shape[0] != 17:
        return False, False

    points = keypoints_list[:, :2]
    confs = keypoints_list[:, 2]

    # 检查关键关键点
    required_joints = [0, 11, 12, 13, 14]  # 鼻子、左右臀、左右膝
    if not all(confs[j] > CONF_THRESHOLD for j in required_joints):
        return False, False

    # 计算头部（鼻子）与臀部中心的垂直角度
    nose = points[0]
    hip_center = np.mean([points[11], points[12]], axis=0)
    leg_center = np.mean([points[13], points[14]], axis=0)

    # 头部-臀部角度
    angle_hip_nose = math.atan2(hip_center[1] - nose[1], hip_center[0] - nose[0])
    angle_hip_nose_deg = abs(math.degrees(angle_hip_nose))

    # 臀部-腿部角度
    angle_hip_leg = math.atan2(leg_center[1] - hip_center[1], leg_center[0] - hip_center[0])
    angle_hip_leg_deg = abs(math.degrees(angle_hip_leg))

    # 综合角度判断
    angle_cond = (angle_hip_nose_deg < 30 or angle_hip_nose_deg > 150) and \
                 (angle_hip_leg_deg < 30 or angle_hip_leg_deg > 150)
    angle_history.append(angle_cond)

    falling = sum(angle_history) >= FALL_THRESHOLD
    return falling, True


# =====================================================
# 通用评估函数
# =====================================================
def evaluate_algorithm(alg_name, detect_func, **kwargs):
    """
    通用评估函数：支持不同检测算法的评估
    :param alg_name: 算法名称
    :param detect_func: 检测函数
    :param kwargs: 检测函数需要的额外参数
    :return: 评估指标字典
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating Algorithm: {alg_name}")
    print(f"{'=' * 60}")

    # 加载标注文件
    gt_df = pd.read_csv(GT_CSV)
    preds = []

    # 遍历falls和adls文件夹
    for cls in ["falls", "adls"]:
        folder = os.path.join(VIDEO_ROOT, cls)
        for video in tqdm(os.listdir(folder), desc=f"Processing {cls}"):
            video_path = os.path.join(folder, video)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告：无法打开视频 {video_path}")
                preds.append((video, 0))
                continue

            # 根据算法初始化历史队列
            if alg_name in ABLATION_CONFIGS.keys():
                # 消融实验：初始化三个历史队列
                knee_history = deque(maxlen=MAX_HISTORY)
                horizontal_history = deque(maxlen=MAX_HISTORY)
                torso_history = deque(maxlen=MAX_HISTORY)
                history_params = (knee_history, horizontal_history, torso_history)
            elif alg_name == "baseline_pose_only":
                history_params = ()
            elif alg_name == "baseline_centroid_height":
                centroid_history = deque(maxlen=MAX_HISTORY)
                history_params = (centroid_history,)
            elif alg_name == "baseline_skeleton_angle":
                angle_history = deque(maxlen=MAX_HISTORY)
                history_params = (angle_history,)
            else:
                history_params = ()

            video_fall_detected = False  # 视频级别跌倒标记

            # 逐帧处理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行检测
                if alg_name in ABLATION_CONFIGS.keys():
                    use_knee, use_horizontal, use_torso = ABLATION_CONFIGS[alg_name]
                    falling, is_valid = detect_func(
                        frame, model, *history_params, use_knee, use_horizontal, use_torso
                    )
                else:
                    falling, is_valid = detect_func(frame, model, *history_params)

                # 有效检测且跌倒则标记视频
                if falling and is_valid:
                    video_fall_detected = True

            # 释放视频流
            cap.release()

            # 视频级别预测：1=跌倒，0=正常
            video_pred = 1 if video_fall_detected else 0
            preds.append((video, video_pred))

    # 计算评估指标
    df_pred = pd.DataFrame(preds, columns=["video", "pred"])
    df = df_pred.merge(gt_df, on="video")

    # 计算TP/FP/FN/TN
    TP = ((df.pred == 1) & (df.label == 1)).sum()
    FP = ((df.pred == 1) & (df.label == 0)).sum()
    FN = ((df.pred == 0) & (df.label == 1)).sum()
    TN = ((df.pred == 0) & (df.label == 0)).sum()

    # 计算精确率、召回率、F1-score
    precision = TP / (TP + FP + 1e-6)  # 加小值避免除零
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # 输出当前算法结果
    print(f"\n{alg_name} 评估结果:")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"F1分数 (F1-score) : {f1:.4f}")
    print(f"混淆矩阵: TP={TP} FP={FP} FN={FN} TN={TN}")

    return {
        "algorithm": alg_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }


# =====================================================
# 执行消融实验和对比实验
# =====================================================
def run_ablation_experiment():
    """执行完整的消融实验和对比实验"""
    all_results = []

    # 1. 执行消融实验（原算法的条件组合）
    for alg_name in ABLATION_CONFIGS.keys():
        result = evaluate_algorithm(alg_name, detect_fall_ablation)
        all_results.append(result)

    # 2. 执行对比算法实验
    compare_funcs = {
        "baseline_pose_only": detect_fall_baseline_pose_only,
        "baseline_centroid_height": detect_fall_baseline_centroid_height,
        "baseline_skeleton_angle": detect_fall_baseline_skeleton_angle
    }
    for alg_name, detect_func in compare_funcs.items():
        result = evaluate_algorithm(alg_name, detect_func)
        all_results.append(result)

    # 3. 汇总并保存结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    # 保存结果到CSV
    save_path = "ablation_experiment_results.csv"
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 60}")
    print("消融实验完整结果已保存到:", save_path)
    print(f"{'=' * 60}")

    # 打印汇总表格
    print("\n消融实验汇总结果（按F1分数排序）:")
    print(results_df[["algorithm", "precision", "recall", "f1"]].to_string(index=False))

    return results_df


# =====================================================
# 主函数
# =====================================================
if __name__ == "__main__":
    # 执行消融实验
    final_results = run_ablation_experiment()

    # 可选：绘制结果可视化（需安装matplotlib）
    try:
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文
        plt.rcParams["axes.unicode_minus"] = False

        # 提取消融实验结果（排除对比基线）
        ablation_results = final_results[final_results["algorithm"].isin(ABLATION_CONFIGS.keys())]
        compare_results = final_results[final_results["algorithm"].isin(COMPARE_ALGORITHMS)]

        # 绘制F1分数对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 消融实验F1对比
        ax1.bar(ablation_results["algorithm"], ablation_results["f1"], color="skyblue")
        ax1.set_title("消融实验 - 不同条件组合F1分数")
        ax1.set_xlabel("算法配置")
        ax1.set_ylabel("F1分数")
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(ablation_results["f1"]):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')

        # 对比算法F1对比
        ax2.bar(compare_results["algorithm"], compare_results["f1"], color="lightcoral")
        ax2.set_title("对比实验 - 主流算法F1分数")
        ax2.set_xlabel("对比算法")
        ax2.set_ylabel("F1分数")
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(compare_results["f1"]):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha='center')

        plt.tight_layout()
        plt.savefig("ablation_experiment_visualization.png", dpi=300, bbox_inches="tight")
        print("\n可视化结果已保存到: ablation_experiment_visualization.png")
        plt.show()

    except ImportError:
        print("\n提示：未安装matplotlib，跳过结果可视化")
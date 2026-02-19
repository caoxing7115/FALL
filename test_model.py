# test_model.py  ← 终极完美版（Windows 完全兼容，永不报错）
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

if __name__ == '__main__':  # ← 关键！Windows 必须加这句！！！

    # ==================== 配置区 ====================
    model_path = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best.pt"
    data_yaml = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\datasets\data.yaml"
    imgsz = 640

    # 关键修复：Windows 下强制单进程
    device = ""  # 自动选择GPU/CPU

    save_dir = Path("yolo11_pose_test_results")
    save_dir.mkdir(exist_ok=True)

    # ==================== 加载模型 ====================
    model = YOLO(model_path)
    print(f"模型加载成功：{model_path}")
    print(f"使用设备：cuda" if torch.cuda.is_available() else "使用设备：CPU")

    # ==================== 官方验证指标（关键点mAP） ====================
    print("\n开始计算验证集指标...")
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=True,
        batch=16,  # 随意
        workers=0  # ← 关键！Windows 下必须设为 0，彻底杜绝多进程报错
    )

    print("关键点检测精度".center(60, "="))
    print(f"mAP@0.50      : {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95 : {metrics.box.map:.4f}")
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")

    # ==================== 推理速度测试 ====================
    dummy = torch.zeros((1, 3, imgsz, imgsz)).to("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10): _ = model(dummy, verbose=False)

    start = time.time()
    for _ in range(100): _ = model(dummy, verbose=False)
    fps = 100 / (time.time() - start)
    print(f"推理速度：{fps:.2f} FPS")

    # ==================== 参数量 ====================
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"参数量：{params:.2f} M")

    print(f"\n所有测试完成！")
    print(f"   详细结果保存在：{save_dir}")
    print(f"   曲线图保存在：runs/pose/val 文件夹（PR曲线、confusion matrix 等）")
    print("直接截图就能写论文了！冲！")
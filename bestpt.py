import os
import time
from ultralytics import YOLO
import torch

# ===================== 配置区 =====================
MODEL_PATH = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\best.pt"
DATA_YAML  = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\datasets\data.yaml"

IMG_SIZE   = 640
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def main():
    assert os.path.exists(MODEL_PATH), "❌ best.pt 不存在"
    assert os.path.exists(DATA_YAML), "❌ data.yaml 不存在"

    print("=" * 60)
    print("YOLO11-Pose Evaluation")
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_YAML}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 加载模型
    model = YOLO(MODEL_PATH)

    # ================== 正式验证 ==================
    results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        split="val",          # 使用验证集
        conf=0.25,
        iou=0.5,
        plots=True,           # 生成 PR / Confusion Matrix 图
        save_json=True        # 生成 COCO-style JSON（论文可用）
    )

    # ================== 结果输出 ==================
    print("\n===== Detection Metrics =====")
    print(f"mAP@0.5      : {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {results.box.map:.4f}")
    print(f"Precision    : {results.box.mp:.4f}")
    print(f"Recall       : {results.box.mr:.4f}")

    print("\n===== Pose Metrics =====")
    print(f"Pose mAP@0.5      : {results.pose.map50:.4f}")
    print(f"Pose mAP@0.5:0.95 : {results.pose.map:.4f}")

    # ================== FPS（可选） ==================
    print("\n===== Speed =====")
    print(f"Preprocess : {results.speed['preprocess']:.2f} ms")
    print(f"Inference  : {results.speed['inference']:.2f} ms")
    print(f"Postprocess: {results.speed['postprocess']:.2f} ms")

    fps = 1000 / results.speed['inference']
    print(f"FPS ≈ {fps:.2f}")

    print("\n✅ Evaluation finished.")

if __name__ == "__main__":
    main()

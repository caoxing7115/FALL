import os
from ultralytics import YOLO
# import torch
# data_loader = torch.utils.data.DataLoader(..., pin_memory=True)
if __name__ == "__main__":
    data_path = r"D:\yolov8-pose-fall-detection\project\yolo11\ultralytics-main\ultralytics-main\datasets\data.yaml"

    # 检查路径是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Starting training with data file: {data_path}")
    print(f"Using model: yolo11n-pose.pt")

    try:
        model = YOLO("yolo11n-pose.pt")
        model.train(data=data_path, epochs=100,batch=8)
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
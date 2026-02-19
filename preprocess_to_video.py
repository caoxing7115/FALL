import cv2
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(r"D:\yolov8-pose-fall-detection\TCN\urfd_classified").resolve()
OUT_DIR = ROOT_DIR / "temp_videos"
FPS = 30

OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_images(img_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in exts:
        images.extend(img_dir.glob(ext))
    return sorted(images, key=lambda x: x.stem)


def images_to_video(img_dir: Path, save_dir: Path, video_name: str):
    images = collect_images(img_dir)
    if not images:
        return False

    first = cv2.imread(str(images[0]))
    if first is None:
        return False

    h, w = first.shape[:2]
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"{video_name}.mp4"
    vw = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    valid = 0
    for img in images:
        frame = cv2.imread(str(img))
        if frame is not None:
            vw.write(frame)
            valid += 1

    vw.release()

    if valid > 0:
        print(f"生成视频：{out_path}")
        return True
    return False


def process_category(category: str):
    cat_root = ROOT_DIR / category
    save_root = OUT_DIR / category

    for seq_dir in tqdm(list(cat_root.iterdir()), desc=f"Processing {category}"):
        if not seq_dir.is_dir():
            continue

        # fall-01 / adl-01
        seq_name = seq_dir.name

        # 🔥 自动扫描 cam 子目录
        for cam_dir in seq_dir.iterdir():
            if not cam_dir.is_dir():
                continue

            images = collect_images(cam_dir)
            if not images:
                continue

            video_name = f"{seq_name}_{cam_dir.name}"
            images_to_video(cam_dir, save_root, video_name)


if __name__ == "__main__":
    process_category("falls")
    process_category("adls")

    print("\n视频输出目录：")
    print(OUT_DIR)

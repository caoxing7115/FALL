import os
import shutil

root_dir = r"D:\yolov8-pose-fall-detection\TCN\adl_deal"

for folder in os.listdir(root_dir):
    outer_path = os.path.join(root_dir, folder)

    if not os.path.isdir(outer_path):
        continue

    items = os.listdir(outer_path)

    # 条件 1：外层目录下只有一个元素
    if len(items) != 1:
        continue

    inner_name = items[0]
    inner_path = os.path.join(outer_path, inner_name)

    # 条件 2：该元素必须是文件夹
    if not os.path.isdir(inner_path):
        continue

    print(f"[处理] {outer_path}")
    print(f"  └─ 展开 {inner_name}")

    for item in os.listdir(inner_path):
        src = os.path.join(inner_path, item)
        dst = os.path.join(outer_path, item)

        # 防止覆盖
        if os.path.exists(dst):
            print(f"    [跳过] 已存在：{item}")
            continue

        shutil.move(src, dst)

    # 删除空的内层目录
    os.rmdir(inner_path)
    print(f"  [完成] 已扁平化\n")

print("✅ 所有异常嵌套目录处理完成")

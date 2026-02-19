import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ================== 路径配置 ==================
URFD_ROOT = Path(r"D:\yolov8-pose-fall-detection\TCN\urfd_classified")
TCN_INPUT_DIR = URFD_ROOT / "tcn_input"
TCN_FUSED_DIR = URFD_ROOT / "tcn_fused"

TCN_FUSED_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 64

# ================== 工具函数 ==================
def load_kpts(prefix):
    """
    自动加载对应前缀的 npy（避免硬编码 cam0-rgb）
    """
    candidates = list(TCN_INPUT_DIR.glob(f"{prefix}*.npy"))
    if len(candidates) == 0:
        return None
    return np.load(candidates[0])


def interp_acc(acc_values, target_len):
    """
    将加速度序列插值到 target_len
    """
    if len(acc_values) == 0:
        return np.zeros(target_len, dtype=np.float32)

    x_old = np.linspace(0, 1, len(acc_values))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, acc_values).astype(np.float32)


def fuse_sequence(kpts_seq, acc_seq):
    """
    kpts_seq: (64, 34)
    acc_seq:  (64,)
    """
    acc_seq = acc_seq.reshape(-1, 1)
    return np.hstack([kpts_seq, acc_seq])


# ================== Falls ==================
print("Processing FALLS ...")

for seq in tqdm(range(1, 31)):
    seq_name = f"fall-{seq:02d}"

    acc_path = URFD_ROOT / "falls" / seq_name / f"{seq_name}-acc.csv"
    if not acc_path.exists():
        print(f"[WARN] 缺少加速度文件: {acc_path}")
        continue

    kpts = load_kpts(f"fall_{seq:02d}_")
    if kpts is None:
        print(f"[WARN] 缺少关键点 npy: fall_{seq:02d}_*.npy")
        continue

    acc_df = pd.read_csv(acc_path)
    if "SV_total" not in acc_df.columns:
        print(f"[WARN] SV_total 不存在: {acc_path}")
        continue

    acc_interp = interp_acc(acc_df["SV_total"].values, SEQ_LEN)
    fused = fuse_sequence(kpts, acc_interp)

    out_path = TCN_FUSED_DIR / f"fall_{seq:02d}.npy"
    np.save(out_path, fused)


# ================== ADLs ==================
print("Processing ADLS ...")

for seq in tqdm(range(1, 41)):
    seq_name = f"adl-{seq:02d}"

    acc_path = URFD_ROOT / "adls" / seq_name / f"{seq_name}-acc.csv"
    if not acc_path.exists():
        print(f"[WARN] 缺少加速度文件: {acc_path}")
        continue

    kpts = load_kpts(f"adl_{seq:02d}_")
    if kpts is None:
        print(f"[WARN] 缺少关键点 npy: adl_{seq:02d}_*.npy")
        continue

    acc_df = pd.read_csv(acc_path)
    if "SV_total" not in acc_df.columns:
        print(f"[WARN] SV_total 不存在: {acc_path}")
        continue

    acc_interp = interp_acc(acc_df["SV_total"].values, SEQ_LEN)
    fused = fuse_sequence(kpts, acc_interp)

    out_path = TCN_FUSED_DIR / f"adl_{seq:02d}.npy"
    np.save(out_path, fused)


print("=" * 50)
print("TCN 融合数据生成完成")
print(f"输出目录: {TCN_FUSED_DIR}")

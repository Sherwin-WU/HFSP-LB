# examples/analyze_two_level_log.py

import os
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def analyze_log(log_path: str) -> None:
    if not os.path.isfile(log_path):
        print(f"[ERROR] log file not found: {log_path}")
        return

    outer_eps = []
    ep_rewards = []
    makespans = []
    buffer_strs = []

    with open(log_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # 检查列名
        fieldnames = reader.fieldnames or []
        print(f"[INFO] columns: {fieldnames}")

        has_buffers = "buffers" in fieldnames

        for row in reader:
            try:
                outer_eps.append(int(row["outer_ep"]))
            except Exception:
                continue
            ep_rewards.append(float(row["ep_reward_U"]))
            makespans.append(float(row["final_makespan"]))
            if has_buffers:
                buffer_strs.append(row["buffers"])

    # === 1. 打印 buffers 分布 ===
    if buffer_strs:
        counter = Counter(buffer_strs)
        print("=== Buffer distribution (by exact vector) ===")
        for buf, cnt in counter.most_common():
            print(f"{buf:>8s} : {cnt}")
    else:
        print("[WARN] no 'buffers' column found or it's empty; "
              "did you rerun training after adding logging?")

    # === 2. 画训练曲线：outer_ep vs final_makespan ===
    if not outer_eps:
        print("[WARN] log file has no data rows.")
        return

    outer_eps = np.array(outer_eps)
    makespans = np.array(makespans)
    ep_rewards = np.array(ep_rewards)

    # 简单滑动平均，窗口可以按需要调整
    window = min(50, len(makespans))
    kernel = np.ones(window, dtype=float) / window
    ma_makespan = np.convolve(makespans, kernel, mode="valid")
    ma_reward = np.convolve(ep_rewards, kernel, mode="valid")
    x_ma = outer_eps[window - 1 :]

    # 图1：makespan
    plt.figure()
    plt.plot(outer_eps, makespans, alpha=0.3, label="makespan (raw)")
    plt.plot(x_ma, ma_makespan, label=f"makespan (MA, window={window})")
    plt.xlabel("outer episode")
    plt.ylabel("final makespan")
    plt.title("Upper-level training: makespan")
    plt.legend()
    plt.tight_layout()

    # 图2：ep_reward_U
    plt.figure()
    plt.plot(outer_eps, ep_rewards, alpha=0.3, label="ep_reward_U (raw)")
    plt.plot(x_ma, ma_reward, label=f"ep_reward_U (MA, window={window})")
    plt.xlabel("outer episode")
    plt.ylabel("ep_reward_U")
    plt.title("Upper-level training: episode reward")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # 换成你实际的 log 路径：
    # 例如：results/j50s3m3/two_level_dqn/20251121_153000/train_log.csv
    log_path = r"results/j50s3m3/upper_arch_ablation/algo_dqn_buffer_uniform_seed0/train_log.csv"
    analyze_log(log_path)

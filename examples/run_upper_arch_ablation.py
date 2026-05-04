# examples/run_upper_arch_ablation.py

import os
from pathlib import Path

import numpy as np
import torch
import random

# 直接复用 train_two_level 里的 ROOT_DIR / train_two_level / helper 函数
from train_two_level import (
    ROOT_DIR,
    train_two_level,
    load_j50s3m3_instances,
    build_two_level_config_for_j50s3m3,
)


def set_global_seed(seed: int) -> None:
    """
    为当前进程设置全局随机种子：
    - Python random
    - NumPy
    - PyTorch (CPU + CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 固定算例规模：j50s3m3
    experiment_name = "j50s3m3"

    # 这整个脚本是架构消融：upper-arch-ablation
    algo_name = "upper_arch_ablation"

    # 三种上层算法
    algo_list = ["d3qn"] #["dqn", "ddqn", "d3qn"]

    # 三种 replay buffer
    replay_list = ["uniform", "per", "nstep"] #["uniform", "per", "nstep"]

    # 每个组合跑若干个随机种子（这里设为 10 个）
    seeds = list(range(10))   # 0,1,...,9

    # 统一的结果根目录：results/j50s3m3/upper_arch_ablation/...
    results_root = Path(ROOT_DIR) / "results" / experiment_name / algo_name
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] all results will be saved under: {results_root}")

    # === 1) 只加载一次算例，所有组合共用 ===
    train_instances, val_instances, test_instances = load_j50s3m3_instances()

    # === 2) 三重循环跑 3×3×len(seeds) 个实验 ===
    for algo_type in algo_list:
        for replay_type in replay_list:
            for seed in seeds:
                exp_name = f"algo_{algo_type}_buffer_{replay_type}_seed{seed}"
                out_dir = results_root / exp_name
                out_dir.mkdir(parents=True, exist_ok=True)

                print("=" * 80)
                print(
                    f"[RUN] algo_type={algo_type}, replay_type={replay_type}, seed={seed}"
                )
                print(f"[RUN] out_dir={out_dir}")

                # 2.1 设置随机种子
                set_global_seed(seed)

                # 2.2 构造本次运行的配置 cfg
                cfg = build_two_level_config_for_j50s3m3(
                    algo_type=algo_type,
                    replay_type=replay_type,
                    seed=seed,
                    # 如果想临时缩短训练，只需修改这里的 num_outer_episodes
                    # 比如先用 300 调试，再改回 1500 正式跑。
                    num_outer_episodes=400,
                    device="cuda",
                )

                # 2.3 开始训练当前组合
                train_two_level(
                    cfg=cfg,
                    train_instances=train_instances,
                    val_instances=val_instances,
                    test_instances=test_instances,
                    out_dir=str(out_dir),
                )

    print("[INFO] all arch-ablation runs finished.")


if __name__ == "__main__":
    main()

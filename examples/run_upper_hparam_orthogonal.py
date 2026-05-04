# examples/run_upper_hparam_orthogonal.py
"""
第二组实验：上层超参正交试验 (D3QN + uniform 为基线)

- 使用 L27(3^13) 的一个 9 因子子表 (A~I)，每个因子 3 个水平；
- 固定算法: algo_type = "d3qn", replay_type = "uniform"；
- 每个正交组合跑 10 个随机种子；
- 每个 run 仍然训练 400 outer episodes，并保存 best_val / last / offline_eval 等。

输出目录结构示例:
    results/j50s3m3/hparam_orthogonal/
        design_01_row1_seed0/
        design_01_row1_seed1/
        ...
        design_27_row27_seed9/

注意：
    - 因子 I (hidden_dim) 需要在 UpperAgentConfig 中增加字段，
      并在构建 DuelingDQNNet 时使用 (见说明)。
"""

from pathlib import Path
from typing import Dict

from train_two_level import (
    build_two_level_config_for_j50s3m3,
    load_j50s3m3_instances,
    train_two_level,
)

# ========= 实验配置 =========

RESULTS_ROOT = Path("results/j50s3m3/hparam_orthogonal")

# 正交表：27 行，每行包含 row 编号以及 A~I 的水平 (1/2/3)
ORTHOGONAL_ROWS = [
    dict(row=1, A=1, B=1, C=1, D=1, E=1, F=1, G=1, H=1, I=1),
    dict(row=2, A=1, B=1, C=2, D=1, E=2, F=2, G=1, H=3, I=3),
    dict(row=3, A=1, B=1, C=3, D=1, E=3, F=3, G=1, H=2, I=2),
    dict(row=4, A=1, B=2, C=1, D=2, E=1, F=2, G=3, H=1, I=2),
    dict(row=5, A=1, B=2, C=2, D=2, E=2, F=3, G=3, H=3, I=1),
    dict(row=6, A=1, B=2, C=3, D=2, E=3, F=1, G=3, H=2, I=3),
    dict(row=7, A=1, B=3, C=1, D=3, E=1, F=3, G=2, H=1, I=3),
    dict(row=8, A=1, B=3, C=2, D=3, E=2, F=1, G=2, H=3, I=2),
    dict(row=9, A=1, B=3, C=3, D=3, E=3, F=2, G=2, H=2, I=1),
    dict(row=10, A=2, B=1, C=1, D=2, E=2, F=1, G=2, H=2, I=1),
    dict(row=11, A=2, B=1, C=2, D=2, E=3, F=2, G=2, H=1, I=3),
    dict(row=12, A=2, B=1, C=3, D=2, E=1, F=3, G=2, H=3, I=2),
    dict(row=13, A=2, B=2, C=1, D=3, E=2, F=2, G=1, H=2, I=2),
    dict(row=14, A=2, B=2, C=2, D=3, E=3, F=3, G=1, H=1, I=1),
    dict(row=15, A=2, B=2, C=3, D=3, E=1, F=1, G=1, H=3, I=3),
    dict(row=16, A=2, B=3, C=1, D=1, E=2, F=3, G=3, H=2, I=3),
    dict(row=17, A=2, B=3, C=2, D=1, E=3, F=1, G=3, H=1, I=2),
    dict(row=18, A=2, B=3, C=3, D=1, E=1, F=2, G=3, H=3, I=1),
    dict(row=19, A=3, B=1, C=1, D=3, E=3, F=1, G=3, H=3, I=1),
    dict(row=20, A=3, B=1, C=2, D=3, E=1, F=2, G=3, H=2, I=3),
    dict(row=21, A=3, B=1, C=3, D=3, E=2, F=3, G=3, H=1, I=2),
    dict(row=22, A=3, B=2, C=1, D=1, E=3, F=2, G=2, H=3, I=2),
    dict(row=23, A=3, B=2, C=2, D=1, E=1, F=3, G=2, H=2, I=1),
    dict(row=24, A=3, B=2, C=3, D=1, E=2, F=1, G=2, H=1, I=3),
    dict(row=25, A=3, B=3, C=1, D=2, E=3, F=3, G=1, H=3, I=3),
    dict(row=26, A=3, B=3, C=2, D=2, E=1, F=1, G=1, H=2, I=2),
    dict(row=27, A=3, B=3, C=3, D=2, E=2, F=2, G=1, H=1, I=1),
]

# 每个组合的随机种子
SEEDS = list(range(10))  # 0..9

# 各因子 level -> 真实数值的映射
GAMMA_LEVEL = {1: 0.97, 2: 0.985, 3: 0.99}
LR_LEVEL = {1: 5e-5, 2: 1e-4, 3: 5e-4}
BATCH_LEVEL = {1: 32, 2: 64, 3: 128}
BUFFER_CAP_LEVEL = {1: 10_000, 2: 30_000, 3: 50_000}
TARGET_UPDATE_LEVEL = {1: 50, 2: 100, 3: 200}
EPS_START_LEVEL = {1: 0.3, 2: 0.5, 3: 0.7}
EPS_END_LEVEL = {1: 0.01, 2: 0.05, 3: 0.1}
EPS_DECAY_LEVEL = {1: 0.992, 2: 0.995, 3: 0.998}
HIDDEN_DIM_LEVEL = {1: 64, 2: 128, 3: 256}

# ========= 帮助函数 =========


def apply_levels_to_config(cfg, levels: Dict[str, int]) -> None:
    """
    根据 A~I 的 level (1/2/3) 设置 upper_agent_cfg 中的具体超参。
    """
    ua = cfg.upper_agent_cfg

    ua.gamma = GAMMA_LEVEL[levels["A"]]
    ua.lr = LR_LEVEL[levels["B"]]
    ua.batch_size = BATCH_LEVEL[levels["C"]]
    ua.buffer_capacity = BUFFER_CAP_LEVEL[levels["D"]]
    ua.target_update_interval = TARGET_UPDATE_LEVEL[levels["E"]]
    ua.epsilon_start = EPS_START_LEVEL[levels["F"]]
    ua.epsilon_end = EPS_END_LEVEL[levels["G"]]
    ua.epsilon_decay_rate = EPS_DECAY_LEVEL[levels["H"]]
    # 因子 I: Dueling 网络宽度
    if hasattr(ua, "hidden_dim"):
        ua.hidden_dim = HIDDEN_DIM_LEVEL[levels["I"]]



def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # 载入 j50s3m3 的 train/val/test 实例
    train_instances, val_instances, test_instances = load_j50s3m3_instances()

    for design in ORTHOGONAL_ROWS:
        row_id = design["row"]
        levels = {k: v for k, v in design.items() if k in "ABCDEFGHI"}

        for seed in SEEDS:
            # 构建基础配置 (D3QN + uniform)
            cfg = build_two_level_config_for_j50s3m3(
                algo_type="d3qn",
                replay_type="uniform",
                seed=seed,
                num_outer_episodes=400,
                device="cuda",
            )

            # 覆盖上层超参
            apply_levels_to_config(cfg, levels)

            # 设定输出目录
            out_dir = (
                RESULTS_ROOT
                / f"design_{row_id:02d}_row{row_id}_seed{seed}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[INFO] start training: row={row_id}, seed={seed}, "
                f"out_dir={out_dir}"
            )

            # 运行训练；train_two_level 内部会完成保存 best_val / last / offline_eval 等
            train_two_level(
                cfg,
                train_instances,
                val_instances,
                test_instances,
                out_dir,
            )


if __name__ == "__main__":
    main()

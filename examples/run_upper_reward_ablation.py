# examples/run_upper_reward_ablation.py
"""
Step3: 上层 reward 消融试验批量脚本。

固定：
    - 结构：D3QN + uniform replay
    - 算例：j50s3m3
    - 上层超参：使用第二组 L27 结果中 row24 对应的 Standard-Hyperparams
    - 外层训练轮数：400

只改变：
    - lambda_B (buffer_cost_weight)
    - deadlock_penalty

网格：
    lambda_list   = [0.0, 0.5, 1.0]
    penalty_list  = [500.0, 1000.0, 2000.0]
    seeds         = range(10)

结果目录：
    results/j50s3m3/reward_ablation/
        d3qn_uniform_lam{λ}_pen{P}_seed{seed}/
            cfg.json
            train_log.csv
            offline_buffer_eval_val.csv
            upper_q_last.pth
            upper_q_best_val.pth
            ...

注意：
    - 本脚本假定 train_two_level.py 中已经提供：
        * build_two_level_config_for_j50s3m3
        * load_j50s3m3_instances
        * train_two_level
    - 并且 UpperAgentConfig 至少包含：
        * gamma, lr, batch_size, buffer_capacity,
          target_update_interval, epsilon_start, epsilon_end,
          epsilon_decay_rate, buffer_cost_weight
      若你在 UpperAgentConfig 中额外加了 hidden_dim / deadlock_penalty 字段，
      本脚本会自动设置（用 hasattr 检查）。
"""

from __future__ import annotations

import os
from pathlib import Path

from train_two_level import (
    build_two_level_config_for_j50s3m3,
    load_j50s3m3_instances,
    train_two_level,
)


# ========= 可调部分：reward 网格 =========
lambda_list = [0.0, 0.5, 1.0]          # 缓冲成本权重 λ_B
penalty_list = [500.0, 1000.0, 2000.0] # deadlock 惩罚
seeds = list(range(10))                # 每种组合的随机种子

# ========= 固定结构 & 标准超参（row24） =========
STANDARD_HPARAMS = dict(
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    buffer_capacity=10_000,
    target_update_interval=100,
    epsilon_start=0.3,
    epsilon_end=0.05,
    epsilon_decay_rate=0.992,
    # 若 UpperAgentConfig 有 hidden_dim 字段，这里会用到
    hidden_dim=256,
)

ALGO_TYPE = "d3qn"
REPLAY_TYPE = "uniform"
NUM_OUTER_EPISODES = 400
DEVICE = "cuda"  # 需要改成 "cpu" 时可以这里改

# ========= 结果目录 =========
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "results" / "j50s3m3" / "reward_ablation"


def apply_standard_hparams(upper_cfg) -> None:
    """把 row24 的超参写入 upper_agent_cfg。"""
    for k, v in STANDARD_HPARAMS.items():
        if hasattr(upper_cfg, k):
            setattr(upper_cfg, k, v)

    # 确保 algo/replay 类型正确
    if hasattr(upper_cfg, "algo_type"):
        upper_cfg.algo_type = ALGO_TYPE
    if hasattr(upper_cfg, "replay_type"):
        upper_cfg.replay_type = REPLAY_TYPE


def format_lambda(lam: float) -> str:
    """把 lambda 格式化成适合文件名的短字符串，例如 0.5 -> '0p5'。"""
    if float(lam).is_integer():
        return str(int(lam))
    return str(lam).replace(".", "p")


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # 统一加载算例（避免每个 run 反复读盘）
    train_instances, val_instances, test_instances = load_j50s3m3_instances()
    print(
        f"[INFO] Loaded instances: "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )

    for lam in lambda_list:
        lam_str = format_lambda(lam)
        for pen in penalty_list:
            pen_int = int(pen)
            for seed in seeds:
                run_name = f"{ALGO_TYPE}_{REPLAY_TYPE}_lam{lam_str}_pen{pen_int}_seed{seed}"
                out_dir = RESULTS_ROOT / run_name

                # 若该组合已跑过且有 offline_buffer_eval_val.csv，则跳过
                if (out_dir / "offline_buffer_eval_val.csv").exists():
                    print(f"[SKIP] {run_name} already done (offline_buffer_eval_val.csv found).")
                    continue

                print(f"\n===== Running: {run_name} =====")

                # 1) 构造基础 cfg（结构固定为 D3QN + uniform）
                cfg = build_two_level_config_for_j50s3m3(
                    algo_type=ALGO_TYPE,
                    replay_type=REPLAY_TYPE,
                    seed=seed,
                    num_outer_episodes=NUM_OUTER_EPISODES,
                    device=DEVICE,
                )

                # 2) 写入标准超参
                apply_standard_hparams(cfg.upper_agent_cfg)

                # 3) 设置 reward 超参：lambda_B 与 deadlock_penalty
                cfg.upper_agent_cfg.buffer_cost_weight = float(lam)

                # 如果 UpperAgentConfig 有 deadlock_penalty 字段，则同步设置
                if hasattr(cfg.upper_agent_cfg, "deadlock_penalty"):
                    cfg.upper_agent_cfg.deadlock_penalty = float(pen)

                # 4) 运行训练 + offline 评估
                os.makedirs(out_dir, exist_ok=True)
                train_two_level(
                    cfg=cfg,
                    train_instances=train_instances,
                    val_instances=val_instances,
                    test_instances=test_instances,
                    out_dir=str(out_dir),
                )

    print("\n[INFO] All reward ablation runs finished.")


if __name__ == "__main__":
    main()

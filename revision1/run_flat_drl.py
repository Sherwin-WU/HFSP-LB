# revision1/run_flat_drl.py
from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
REVISION1_DIR = PROJECT_ROOT / "revision1"

if str(REVISION1_DIR) not in sys.path:
    sys.path.insert(0, str(REVISION1_DIR))

from flat_drl_baseline import run_flat_drl_for_experiment  # noqa: E402


# ============================================================
# CONFIG：revision1 / Flat-DRL
# ============================================================

# 先跑代表实例；后面再扩展到 12 类
EXPERIMENTS = [
    #"j50s3",
    "j50s4",
    "j50s5",
    "j80s3",
    "j80s4",
    "j80s5",
    "j160s3",
    "j160s4",
    "j160s5",
    "j200s3",
    "j200s4",
    #"j200s5",
]

# 正式版建议 5 seeds
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

DEVICE_STR = "cuda"

# 冒烟时可改成 20 或 50；正式版恢复 400
NUM_OUTER_EPISODES = 200

METHOD_NAME = "flat_drl_single_agent"

# Flat-DRL 默认沿用当前主 reward 口径
DISPATCH_REWARD_SCHEME = "dense_epi"
DISPATCH_EPI_REWARD_WEIGHT = 1.0


def main() -> None:
    print("=" * 80)
    print("[REVISION1] run Flat-DRL supplementary experiments")
    print(f"[INFO] PROJECT_ROOT   = {PROJECT_ROOT}")
    print(f"[INFO] EXPERIMENTS    = {EXPERIMENTS}")
    print(f"[INFO] SEEDS          = {SEEDS}")
    print(f"[INFO] DEVICE         = {DEVICE_STR}")
    print(f"[INFO] OUTER_EPISODES = {NUM_OUTER_EPISODES}")
    print(f"[INFO] METHOD_NAME    = {METHOD_NAME}")
    print(f"[INFO] REWARD_SCHEME  = {DISPATCH_REWARD_SCHEME}")
    print("=" * 80)

    for exp_name in EXPERIMENTS:
        print("\n" + "-" * 80)
        print(
            f"[RUN] experiment={exp_name}, method={METHOD_NAME}, "
            f"seeds={SEEDS}, outer_episodes={NUM_OUTER_EPISODES}"
        )
        print("-" * 80)

        run_flat_drl_for_experiment(
            experiment_name=exp_name,
            seeds=SEEDS,
            device_str=DEVICE_STR,
            method_name=METHOD_NAME,
            num_outer_episodes=NUM_OUTER_EPISODES,
            dispatch_reward_scheme=DISPATCH_REWARD_SCHEME,
            dispatch_epi_reward_weight=DISPATCH_EPI_REWARD_WEIGHT,
        )

    print("\n[DONE] revision1 Flat-DRL experiments finished.")


if __name__ == "__main__":
    main()
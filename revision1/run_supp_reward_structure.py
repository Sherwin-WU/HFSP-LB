# revision1/run_supp_reward_structure.py
from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# 路径处理：确保可以从 revision1/ 调到 examples/group4_two_level.py
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from group4_two_level import run_group4_for_experiment  # noqa: E402


# ============================================================
# CONFIG：这轮 revision1 reward structure 补实验
# ============================================================

# 建议先从两个代表规模做起
EXPERIMENTS = [
    "j50s3",
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
    "j200s5",
]

# 正式版建议 5 个 seed；若先试跑可改成 [0]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #, 1, 2, 3, 4

# 设备
DEVICE_STR = "cuda"

# 外层训练轮数
# 先试跑可设 20 或 50；正式跑再恢复 400
NUM_OUTER_EPISODES = 200

# 三种 reward structure
METHODS = [
    ("group4_two_level_dense_epi", "dense_epi"),
    ("group4_two_level_terminal_only", "terminal_only"),
    ("group4_two_level_dense_only", "dense_only"),
]

# shared episodic reward 的权重
LOWER_EPI_REWARD_WEIGHT = 1.0


def main() -> None:
    print("=" * 80)
    print("[REVISION1] reward structure supplementary experiments")
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] EXPERIMENTS  = {EXPERIMENTS}")
    print(f"[INFO] SEEDS        = {SEEDS}")
    print(f"[INFO] DEVICE       = {DEVICE_STR}")
    print(f"[INFO] OUTER_EP     = {NUM_OUTER_EPISODES}")
    print("=" * 80)

    for exp_name in EXPERIMENTS:
        for method_name, reward_scheme in METHODS:
            print("\n" + "-" * 80)
            print(
                f"[RUN] experiment={exp_name}, "
                f"method={method_name}, "
                f"reward_scheme={reward_scheme}, "
                f"seeds={SEEDS}"
            )
            print("-" * 80)

            run_group4_for_experiment(
                experiment_name=exp_name,
                seeds=SEEDS,
                device_str=DEVICE_STR,
                method_name=method_name,
                num_outer_episodes=NUM_OUTER_EPISODES,
                lower_reward_scheme=reward_scheme,
                lower_epi_reward_weight=LOWER_EPI_REWARD_WEIGHT,
            )

    print("\n[DONE] revision1 reward structure experiments finished.")


if __name__ == "__main__":
    main()
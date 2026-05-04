# examples/run_supp_reward_structure.py
from __future__ import annotations

from group4_two_level import run_group4_for_experiment

# ===== 可调 CONFIG =====
EXPERIMENTS = [
    "j50s3",
    "j200s5",
]

SEEDS = [0, 1, 2, 3, 4]
DEVICE_STR = "cuda"
NUM_OUTER_EPISODES = 400

METHODS = [
    ("group4_two_level_dense_epi", "dense_epi"),
    ("group4_two_level_terminal_only", "terminal_only"),
    ("group4_two_level_dense_only", "dense_only"),
]

# 若希望后面做“terminal alignment 强弱”测试，可再扫这个权重
LOWER_EPI_REWARD_WEIGHT = 1.0


def main() -> None:
    for exp_name in EXPERIMENTS:
        for method_name, reward_scheme in METHODS:
            print("=" * 80)
            print(
                f"[RUN] exp={exp_name}, method={method_name}, "
                f"reward_scheme={reward_scheme}, seeds={SEEDS}"
            )
            run_group4_for_experiment(
                experiment_name=exp_name,
                seeds=SEEDS,
                device_str=DEVICE_STR,
                method_name=method_name,
                num_outer_episodes=NUM_OUTER_EPISODES,
                lower_reward_scheme=reward_scheme,
                lower_epi_reward_weight=LOWER_EPI_REWARD_WEIGHT,
            )


if __name__ == "__main__":
    main()
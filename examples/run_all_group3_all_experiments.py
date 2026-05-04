# examples/run_all_group3_all_experiments.py

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

from group3_dispatch_fixedbuf import run_group3_for_experiment


NUM_JOBS_LIST = [50, 80, 160, 200]
NUM_STAGES_LIST = [3, 4, 5]

SEEDS = list(range(10))          # Group3 需要 10 个种子
DEVICE = "cuda"
NUM_EPISODES = 400
MAX_STEPS = 2_000
EVAL_INTERVAL = 2_000
LOG_INTERVAL = 100

# 这里的 fixed_buffers 将来可以从 Group1 的 summary 里读出来；
# 现在先占位用一个 dict 或之后再填。
BEST_BUFFERS = {
     "j50s3": [5, 4],
     "j50s4": [5, 4, 4],
     "j50s5": [5, 4, 4, 5],
     "j80s3": [5, 4],
     "j80s4": [5, 5, 5],
     "j80s5": [5, 5, 4, 5],
     "j160s3": [5, 5],
     "j160s4": [5, 3, 5],
     "j160s5": [5, 5, 5, 5],
     "j200s3": [4, 5],
     "j200s4": [5, 5, 4],
     "j200s5": [5, 4, 4, 5],
}


@dataclass
class ExperimentConfig:
    name: str
    num_jobs: int
    num_stages: int


def build_experiment_list() -> List[ExperimentConfig]:
    exps: List[ExperimentConfig] = []
    for nj in NUM_JOBS_LIST:
        for ns in NUM_STAGES_LIST:
            name = f"j{nj}s{ns}"
            exps.append(ExperimentConfig(name=name, num_jobs=nj, num_stages=ns))
    return exps


def main():
    exps = build_experiment_list()
    print("[INFO] Experiments to run (Group3 - fixed buffer D3QN dispatch):")
    for e in exps:
        print(f"  - {e.name}: jobs={e.num_jobs}, stages={e.num_stages}")

    for e in exps:
        if e.name not in BEST_BUFFERS:
            print(f"[WARN] No fixed_buffers specified for {e.name}, skip.")
            continue

        fixed_buf = BEST_BUFFERS[e.name]

        print("\n" + "=" * 80)
        print(
            f"[MASTER] Running Group3 for experiment {e.name} "
            f"(jobs={e.num_jobs}, stages={e.num_stages}, fixed_buffers={fixed_buf})"
        )
        print("=" * 80)

        run_group3_for_experiment(
            experiment_name=e.name,
            fixed_buffers=fixed_buf,
            seeds=SEEDS,
            device_str=DEVICE,
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            eval_interval=EVAL_INTERVAL,
            log_interval=LOG_INTERVAL,
        )


if __name__ == "__main__":
    main()

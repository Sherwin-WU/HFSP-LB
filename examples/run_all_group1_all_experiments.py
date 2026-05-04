# examples/run_all_group1_all_experiments.py

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

# 导入 Group1 函数
from group1_rule_baseline import run_group1_for_experiment


NUM_JOBS_LIST = [50, 80, 160, 200]
NUM_STAGES_LIST = [3, 4, 5]

RULES = ["fifo", "spt", "lpt", "srpt"]   # Group1 固定四条规则
BASE_SEED = 0
MAX_STEPS = 10_000


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
    print("[INFO] Experiments to run (Group1 - rule baseline):")
    for e in exps:
        print(f"  - {e.name}: jobs={e.num_jobs}, stages={e.num_stages}")

    for e in exps:
        print("\n" + "=" * 80)
        print(f"[MASTER] Running Group1 for experiment {e.name} "
              f"(jobs={e.num_jobs}, stages={e.num_stages})")
        print("=" * 80)

        run_group1_for_experiment(
            experiment_name=e.name,
            rules=RULES,
            base_seed=BASE_SEED,
            max_steps=MAX_STEPS,
        )


if __name__ == "__main__":
    main()

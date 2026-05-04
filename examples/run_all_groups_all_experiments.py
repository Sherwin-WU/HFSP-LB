# examples/run_all_groups_all_experiments.py
"""
总控脚本 (只跑 Group4，两层 BA+DA)：

算例定义：
  - 工件数量 num_jobs ∈ {50, 80, 160, 200}
  - 阶段数   num_stages ∈ {3, 4, 5}
  - 实验名格式：j{jobs}s{stages}，例如 j50s3, j80s4 等

前置条件：
  已经通过 examples/gen_instances.py 生成好：
    experiments/raw/j{jobs}s{stages}/{train,val,test}/inst_*.pkl

本脚本行为：
  - 对每个算例 exp_name 调用 group4_two_level.run_group4_for_experiment()
  - 每个算例 10 个种子
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

# ---------- 路径设置：把 src/ 和 examples/ 加入 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

# ---------- 导入 group4 函数 ----------
from group4_two_level import run_group4_for_experiment


# ============================================================
# 实验 / 运行配置
# ============================================================

NUM_JOBS_LIST = [50, 80, 160, 200]
NUM_STAGES_LIST = [3, 4, 5]

SEEDS = list(range(10))          # 每个算例 10 个种子
DEVICE = "cuda"
NUM_OUTER_EPISODES = 400
METHOD_NAME = "group4_two_level"


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


# ============================================================
# main：遍历所有算例，调用 Group4
# ============================================================

def main():
    exps = build_experiment_list()
    print("[INFO] Experiments to run (Group4 only):")
    for e in exps:
        print(f"  - {e.name}: jobs={e.num_jobs}, stages={e.num_stages}")

    for e in exps:
        print("\n" + "=" * 80)
        print(f"[MASTER] Running Group4 for experiment {e.name} "
              f"(jobs={e.num_jobs}, stages={e.num_stages})")
        print("=" * 80)

        run_group4_for_experiment(
            experiment_name=e.name,
            seeds=SEEDS,
            device_str=DEVICE,
            method_name=METHOD_NAME,
            num_outer_episodes=NUM_OUTER_EPISODES,
        )


if __name__ == "__main__":
    main()

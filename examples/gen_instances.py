# examples/gen_instances.py

import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from instances.generators import FlowShopGeneratorConfig, generate_random_instance
from instances.io import save_instance_pickle


def parse_experiment_name(name: str) -> tuple[int, int]:
    """
    解析实验名，支持两种格式：
      1) 新格式：'j50s3'  -> (50, 3)
      2) 兼容旧格式：'j50s3m3' -> (50, 3)

    注意：设备数不再从名字里解析，而是在生成实例时对每个 stage
    独立从 {3,4,5} 采样。
    """
    try:
        assert name[0] == "j"
        # 先按有没有 'm' 来区分
        if "m" in name:
            # 旧格式 j{jobs}s{stages}m{machines}
            j_part, rest = name[1:].split("s", 1)
            s_part, _ = rest.split("m", 1)
            num_jobs = int(j_part)
            num_stages = int(s_part)
        else:
            # 新格式 j{jobs}s{stages}
            j_part, s_part = name[1:].split("s", 1)
            num_jobs = int(j_part)
            num_stages = int(s_part)
        return num_jobs, num_stages
    except Exception as e:
        raise ValueError(f"无法解析实验名: {name}") from e


def generate_split(
    exp_name: str,
    split: str,
    num_instances: int,
    base_seed: int = 0,
) -> None:
    """
    在 experiments/raw/<exp_name>/<split> 下生成 num_instances 个算例。
    split ∈ {"train","val","test"}。

    约束：
      - 工件数 num_jobs 由 exp_name 决定，如 'j50s3' -> 50
      - 阶段数 num_stages 由 exp_name 决定，如 'j50s3' -> 3
      - 每个实例、每个阶段的设备数 machines_per_stage[k] ~ Uniform{3,4,5}
      - 加工时间区间 [1, 30]，same_proc_time_across_machines=True
    """
    data_root = Path(ROOT_DIR) / "experiments" / "raw" / exp_name / split
    data_root.mkdir(parents=True, exist_ok=True)

    num_jobs, num_stages = parse_experiment_name(exp_name)

    import numpy as np
    rng = np.random.default_rng(base_seed)

    for idx in range(num_instances):
        # 每个实例独立采样 machines_per_stage
        machines_per_stage = [int(rng.integers(3, 6)) for _ in range(num_stages)]

        gen_cfg = FlowShopGeneratorConfig(
            num_jobs=num_jobs,
            num_stages=num_stages,
            machines_per_stage=machines_per_stage,
            proc_time_low=1,
            proc_time_high=30,
            same_proc_time_across_machines=True,
            seed=int(rng.integers(0, 10**9)),
        )

        # 使用同一个 rng，保证 generate_random_instance 内部使用该随机源
        instance = generate_random_instance(gen_cfg, rng)

        filename = f"inst_{idx:04d}.pkl"
        save_instance_pickle(data_root / filename, instance)

        if (idx + 1) % 100 == 0 or idx == num_instances - 1:
            print(f"[{exp_name}/{split}] generated {idx+1}/{num_instances}")


def main():
    """
    一键生成 12 种算例：
      - num_jobs ∈ {50, 80, 160, 200}
      - num_stages ∈ {3, 4, 5}
    实验名格式统一为 'j{jobs}s{stages}'，例如：
      j50s3, j50s4, j50s5, j80s3, ... , j200s5
    """
    num_jobs_list = [50, 80, 160, 200]
    num_stages_list = [3, 4, 5]

    # train/val/test 数量，可以按需调整
    n_train, n_val, n_test = 1000, 20, 20

    base_seed = 0

    for i, nj in enumerate(num_jobs_list):
        for j, ns in enumerate(num_stages_list):
            exp_name = f"j{nj}s{ns}"
            print(f"\n=== Generating instances for {exp_name} (jobs={nj}, stages={ns}) ===")

            # 为不同算例设置不同的 base_seed，防止数据集完全一致
            seed_offset = (i * len(num_stages_list) + j) * 10000

            generate_split(exp_name, "train", n_train, base_seed=base_seed + seed_offset + 0)
            generate_split(exp_name, "val",   n_val,   base_seed=base_seed + seed_offset + 1)
            generate_split(exp_name, "test",  n_test,  base_seed=base_seed + seed_offset + 2)


if __name__ == "__main__":
    main()

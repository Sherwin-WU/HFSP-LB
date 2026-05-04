# examples/check_upper_arch_ablation_runs.py
"""
小工具：检查 upper_arch_ablation 批量实验中，哪些 (algo, replay) 组合
还缺哪些 seed（以 offline_buffer_eval_val.csv 是否存在为准）。

使用方式：
    1. 根据实际情况修改 ROOT_DIR / ALGO_LIST / REPLAY_LIST / EXPECTED_SEEDS。
    2. python check_upper_arch_ablation_runs.py
"""

from pathlib import Path
from typing import List, Dict

# ======== 配置区域 ========
ROOT_DIR = Path("results/j50s3m3/upper_arch_ablation")

# 你当前批量跑用到的算法 / replay 类型
ALGO_LIST = ["dqn", "ddqn", "d3qn"]
REPLAY_LIST = ["uniform", "per", "nstep"]

# 期望每个组合应该有的随机种子
EXPECTED_SEEDS = list(range(10))   # 如果你用 1~10，可以改成 [1,2,...,10]
# =========================


def parse_seed_from_name(name: str) -> int | None:
    """
    从目录名中解析 seed，例如:
        algo_d3qn_buffer_nstep_seed7  -> 7
    解析失败时返回 None。
    """
    if "seed" not in name:
        return None
    try:
        return int(name.split("seed")[-1])
    except ValueError:
        return None


def find_complete_seeds_for_combo(
    root: Path, algo: str, replay: str
) -> List[int]:
    """
    在 root 下查找某个 (algo, replay) 组合中，哪些 seed 已经“完整”（有 offline_buffer_eval_val.csv）。
    """
    prefix = f"algo_{algo}_buffer_{replay}_seed"
    complete_seeds: List[int] = []

    if not root.exists():
        print(f"[WARN] ROOT_DIR 不存在: {root}")
        return complete_seeds

    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue

        seed = parse_seed_from_name(d.name)
        if seed is None:
            continue

        offline_path = d / "offline_buffer_eval_val.csv"
        if offline_path.exists():
            complete_seeds.append(seed)
        else:
            # 目录存在但缺 offline_eval，说明 run 没完整跑完
            print(f"[INFO] 发现未完成 run: {d} (缺 offline_buffer_eval_val.csv)")

    complete_seeds = sorted(set(complete_seeds))
    return complete_seeds


def main():
    print(f"[INFO] 扫描目录: {ROOT_DIR}")
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {ROOT_DIR}")

    missing_summary: Dict[str, Dict[str, List[int]]] = {}

    for algo in ALGO_LIST:
        missing_summary[algo] = {}
        for replay in REPLAY_LIST:
            complete = find_complete_seeds_for_combo(ROOT_DIR, algo, replay)
            expected_set = set(EXPECTED_SEEDS)
            complete_set = set(complete)
            missing = sorted(expected_set - complete_set)

            print("-" * 60)
            print(f"组合: algo={algo}, replay={replay}")
            print(f"  已完成 seeds: {complete if complete else '无'}")
            print(f"  期望 seeds:   {sorted(expected_set)}")
            print(f"  缺失 seeds:   {missing if missing else '无 (已齐全)'}")

            missing_summary[algo][replay] = missing

    print("\n====== 汇总建议（按组合给出可用于 seeds=... 的列表） ======")
    for algo in ALGO_LIST:
        for replay in REPLAY_LIST:
            missing = missing_summary[algo][replay]
            if not missing:
                continue
            print(
                f"algo={algo}, replay={replay} -> 需要补跑的 seeds = {missing}"
            )

    print(
        "\n提示：你可以根据上面的输出，临时把 examples/run_upper_arch_ablation.py\n"
        "里的 seeds 列表改成对应组合缺失的 seeds，然后只跑这些组合；\n"
        "已经有 offline_buffer_eval_val.csv 的 run 不需要重跑。"
    )


if __name__ == "__main__":
    main()

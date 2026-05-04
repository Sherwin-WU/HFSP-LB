# examples/run_rule_baseline_j50s3m3.py
"""
在 j50s3m3 的 val 集上，对多种规则 (fifo/spt/lpt/srpt)
进行 “固定缓存 + 规则调度” 的离线枚举评估。

输出：
  1) rule_baseline_details.csv  —— 所有 rule × buffers 组合的结果
  2) rule_baseline_summary.csv  —— 每个 rule 最优 buffers 的摘要

评价指标：
  - avg_makespan：只在 non-deadlock episode 上取平均；
                  若该 rule 下某个 buffers 对所有实例都 deadlock，则记为 +inf。
  - deadlock_rate：deadlock 次数 / 总实例数。
"""

import os
import sys
from pathlib import Path
import csv
import math
from datetime import datetime
import itertools
from typing import List, Dict, Any

import numpy as np

# ---- 路径设置：把 src/ 加进 sys.path ----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from instances.io import load_instances_from_dir
from instances.types import InstanceData
from envs.buffer_design_env import compute_buffer_upper_bounds
from envs.ffs_core_env import simulate_instance_with_job_rule


# ============================================================
# 1. 数据加载：j50s3m3 val instances
# ============================================================

def load_j50s3m3_val_instances() -> List[InstanceData]:
    experiment_name = "j50s3m3"
    data_root = os.path.join(ROOT_DIR, "experiments", "raw", experiment_name)
    val_dir = os.path.join(data_root, "val")
    val_instances = load_instances_from_dir(val_dir)
    print(f"[INFO] Loaded j50s3m3 val instances: {len(val_instances)}")
    return val_instances


# ============================================================
# 2. 上界计算与 buffer 枚举
# ============================================================

def compute_global_buffer_upper_bounds_for_val(
    instances: List[InstanceData],
) -> List[int]:
    """
    对给定实例集合，按 compute_buffer_upper_bounds 计算每个缓冲段的全局上界。
    """
    if not instances:
        return []

    base_bounds = compute_buffer_upper_bounds(instances[0])
    if not base_bounds:
        return []

    global_bounds = list(base_bounds)
    for inst in instances[1:]:
        bounds = compute_buffer_upper_bounds(inst)
        if not bounds:
            continue
        n = min(len(global_bounds), len(bounds))
        for k in range(n):
            global_bounds[k] = max(global_bounds[k], bounds[k])

    return global_bounds


def enumerate_candidate_buffers(upper_bounds: List[int]) -> List[List[int]]:
    """
    给定每个缓冲段的上界，例如 [3, 3]，枚举所有 0..a_k 的组合：
      -> [[0,0], [0,1], ..., [3,3]]
    """
    if not upper_bounds:
        return []
    ranges = [range(b + 1) for b in upper_bounds]
    return [list(buf) for buf in itertools.product(*ranges)]


# ============================================================
# 3. 单个 buffer × 规则 的评估
# ============================================================

def evaluate_one_buffer_for_rule(
    rule: str,
    buffers: List[int],
    instances: List[InstanceData],
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """
    在给定 rule + buffers + 实例集合上做评估。

    返回：
      {
        "rule": str,
        "buffers": List[int],
        "avg_makespan": float,   # 只对非 deadlock 实例取平均，若全 deadlock 则为 inf
        "deadlock_rate": float,
        "episodes": int,
        "num_deadlocks": int,
      }
    """
    num_instances = len(instances)
    if num_instances == 0:
        return dict(
            rule=rule,
            buffers=list(buffers),
            avg_makespan=math.inf,
            deadlock_rate=0.0,
            episodes=0,
            num_deadlocks=0,
        )

    sum_makespan_non_dl = 0.0
    cnt_non_dl = 0
    num_deadlocks = 0

    for idx, inst in enumerate(instances):
        metrics = simulate_instance_with_job_rule(
            instance=inst,
            buffers=buffers,
            job_rule=rule,
            machine_rule="min_index",
            max_steps=max_steps,
            seed=seed + idx,
        )
        mk = float(metrics.get("makespan", math.inf))
        dl = bool(metrics.get("deadlock", False))

        if dl:
            num_deadlocks += 1
        else:
            sum_makespan_non_dl += mk
            cnt_non_dl += 1

    if cnt_non_dl > 0:
        avg_makespan = sum_makespan_non_dl / float(cnt_non_dl)
    else:
        avg_makespan = math.inf

    deadlock_rate = num_deadlocks / float(num_instances)

    return dict(
        rule=rule,
        buffers=list(buffers),
        avg_makespan=avg_makespan,
        deadlock_rate=deadlock_rate,
        episodes=num_instances,
        num_deadlocks=num_deadlocks,
    )


# ============================================================
# 4. 对单个 rule 枚举所有 buffers 的评估
# ============================================================

def evaluate_all_buffers_for_rule(
    rule: str,
    candidate_buffers: List[List[int]],
    val_instances: List[InstanceData],
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    对给定 rule，在所有 candidate_buffers 上做评估。

    返回列表，每个元素为一个 dict，包含：
      - rule
      - buffers
      - avg_makespan
      - deadlock_rate
      - episodes
      - num_deadlocks
      - buffer_index
    """
    results: List[Dict[str, Any]] = []

    print(f"[RULE] Evaluating rule='{rule}' on {len(candidate_buffers)} buffer candidates...")
    for idx, bufs in enumerate(candidate_buffers):
        res = evaluate_one_buffer_for_rule(
            rule=rule,
            buffers=bufs,
            instances=val_instances,
            max_steps=max_steps,
            seed=seed + 1000 * idx,  # 简单区分一下随机种子
        )
        res["buffer_index"] = idx
        results.append(res)

        if (idx + 1) % 10 == 0 or idx == len(candidate_buffers) - 1:
            print(
                f"  [PROG] rule={rule}, evaluated {idx + 1}/{len(candidate_buffers)} "
                f"candidates; last avg_ms={res['avg_makespan']:.3f}, dl_rate={res['deadlock_rate']:.3f}"
            )

    return results


# ============================================================
# 5. 写 CSV：details + summary
# ============================================================

def write_details_csv(
    all_results: List[Dict[str, Any]],
    out_dir: str | Path,
) -> None:
    """
    写入 rule_baseline_details.csv：
      rule, buffers, avg_makespan, deadlock_rate, episodes, num_deadlocks, buffer_index, rank_within_rule
    """
    if not all_results:
        print("[WARN] write_details_csv: all_results is empty, skip.")
        return

    # 先按 rule 分组，再在组内排序以得到 rank
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rule_baseline_details.csv"

    # 对 (rule, deadlock_rate, avg_makespan) 排序，便于查看
    all_results_sorted = sorted(
        all_results,
        key=lambda r: (r["rule"], r["deadlock_rate"], r["avg_makespan"]),
    )

    # 为每个 rule 分配 rank_within_rule
    rank_counter: Dict[str, int] = {}
    for r in all_results_sorted:
        rule = r["rule"]
        rank_counter.setdefault(rule, 0)
        rank_counter[rule] += 1
        r["rank_within_rule"] = rank_counter[rule]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rule",
                "buffers",
                "avg_makespan",
                "deadlock_rate",
                "episodes",
                "num_deadlocks",
                "buffer_index",
                "rank_within_rule",
            ]
        )
        for row in all_results_sorted:
            buffer_str = " ".join(str(b) for b in row["buffers"])
            writer.writerow(
                [
                    row["rule"],
                    buffer_str,
                    row["avg_makespan"],
                    row["deadlock_rate"],
                    row["episodes"],
                    row["num_deadlocks"],
                    row["buffer_index"],
                    row["rank_within_rule"],
                ]
            )

    print(f"[SAVE] rule_baseline_details.csv -> {csv_path}")


def write_summary_csv(
    best_per_rule: List[Dict[str, Any]],
    out_dir: str | Path,
) -> None:
    """
    写入 rule_baseline_summary.csv：
      rule, best_buffers, avg_makespan, deadlock_rate, episodes, num_deadlocks
    """
    if not best_per_rule:
        print("[WARN] write_summary_csv: best_per_rule is empty, skip.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rule_baseline_summary.csv"

    # 按 rule 名排序一下
    best_sorted = sorted(best_per_rule, key=lambda r: r["rule"])

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rule",
                "best_buffers",
                "avg_makespan",
                "deadlock_rate",
                "episodes",
                "num_deadlocks",
            ]
        )
        for row in best_sorted:
            buffer_str = " ".join(str(b) for b in row["buffers"])
            writer.writerow(
                [
                    row["rule"],
                    buffer_str,
                    row["avg_makespan"],
                    row["deadlock_rate"],
                    row["episodes"],
                    row["num_deadlocks"],
                ]
            )

    print(f"[SAVE] rule_baseline_summary.csv -> {csv_path}")


# ============================================================
# 6. main()
# ============================================================

def main():
    # ---- 基本配置 ----
    experiment_name = "j50s3m3"
    RULES = ["fifo", "spt", "lpt", "srpt"]
    base_seed = 0
    max_steps = 10_000  # 给规则调度的最大步数上限

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        ROOT_DIR,
        "results",
        experiment_name,
        "rule_baseline",
        run_id,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Results will be saved to: {out_dir}")

    # ---- 1. 加载 val 实例 ----
    val_instances = load_j50s3m3_val_instances()
    if not val_instances:
        print("[ERROR] No val instances loaded, abort.")
        return

    # ---- 2. 计算全局 buffer 上界，并枚举候选 buffers ----
    upper_bounds = compute_global_buffer_upper_bounds_for_val(val_instances)
    if not upper_bounds:
        print("[ERROR] Failed to compute buffer upper bounds, abort.")
        return

    candidate_buffers = enumerate_candidate_buffers(upper_bounds)
    if not candidate_buffers:
        print("[ERROR] No candidate buffers generated, abort.")
        return

    print(
        f"[INFO] Global buffer upper bounds = {upper_bounds}, "
        f"num_candidates = {len(candidate_buffers)}"
    )

    # ---- 3. 对每个 rule 分别评估 ----
    all_results: List[Dict[str, Any]] = []
    best_per_rule: List[Dict[str, Any]] = []

    for rule in RULES:
        # 对当前 rule 评估所有 candidate buffers
        res_rule = evaluate_all_buffers_for_rule(
            rule=rule,
            candidate_buffers=candidate_buffers,
            val_instances=val_instances,
            max_steps=max_steps,
            seed=base_seed,
        )
        all_results.extend(res_rule)

        # 找到该 rule 下的最优 buffers
        # 排序 key：先按 deadlock_rate，再按 avg_makespan
        res_rule_sorted = sorted(
            res_rule,
            key=lambda r: (r["deadlock_rate"], r["avg_makespan"]),
        )
        best = res_rule_sorted[0]
        best_per_rule.append(best)

        print(
            f"[BEST] rule={rule}, buffers={best['buffers']}, "
            f"avg_makespan={best['avg_makespan']:.3f}, "
            f"deadlock_rate={best['deadlock_rate']:.3f}, "
            f"episodes={best['episodes']}, "
            f"num_deadlocks={best['num_deadlocks']}"
        )

    # ---- 4. 写 CSV ----
    write_details_csv(all_results, out_dir)
    write_summary_csv(best_per_rule, out_dir)

    print("[DONE] rule baseline evaluation finished.")


if __name__ == "__main__":
    main()

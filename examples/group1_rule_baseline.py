# examples/group1_rule_baseline.py
"""
Group1：通用 rule baseline 评估（固定缓存 + 规则派工），函数版。

对外接口（给总控脚本调用）：

    run_group1_for_experiment(
        experiment_name: str,
        rules: List[str],
        base_seed: int = 0,
        max_steps: int = 10_000,
    )

行为：
  - 在 experiments/raw/<experiment_name>/val 上：
      * 先用 compute_buffer_upper_bounds 算出全局 buffer 上界
      * 枚举所有 candidate buffers
      * 对每条规则 rule ∈ rules，在所有 candidate buffers 上做仿真
      * 统计 avg_makespan（只在非 deadlock 上平均）和 deadlock_rate
  - 结果保存到：
      results/<experiment_name>/rule_baseline/<run_id>/
        - rule_baseline_details.csv
        - rule_baseline_summary.csv
"""

from __future__ import annotations

import os
import sys
import csv
import math
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ---------- 路径设置：把 src/ 加入 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ---------- 项目内部导入 ----------
from instances.io import load_instances_from_dir
from instances.types import InstanceData
from envs.buffer_design_env import compute_buffer_upper_bounds
from envs.ffs_core_env import simulate_instance_with_job_rule


# ============================================================
# 1. 加载 val 实例
# ============================================================

def _load_val_instances(experiment_name: str) -> List[InstanceData]:
    data_root = Path(ROOT_DIR) / "experiments" / "raw" / experiment_name
    val_dir = data_root / "val"
    if not val_dir.exists():
        print(f"[ERROR] val directory not found: {val_dir}")
        return []
    instances = load_instances_from_dir(str(val_dir))
    print(f"[DATA] Loaded val instances for {experiment_name}: {len(instances)}")
    return instances


# ============================================================
# 2. buffer 上界 + 枚举 candidate buffers
# ============================================================

def _compute_global_buffer_upper_bounds(
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


def _enumerate_candidate_buffers(upper_bounds: List[int]) -> List[List[int]]:
    """
    给定每个缓冲段的上界，例如 [3, 3]，枚举所有 0..a_k 的组合：
      -> [[0,0], [0,1], ..., [3,3]]
    """
    if not upper_bounds:
        return []
    ranges = [range(b + 1) for b in upper_bounds]
    return [list(buf) for buf in itertools.product(*ranges)]


# ============================================================
# 3. 单个 (rule, buffers) 的评估
# ============================================================

def _evaluate_one_buffer_for_rule(
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
        ms = float(metrics.get("makespan", math.inf))
        dl = bool(metrics.get("deadlock", False))

        if dl:
            num_deadlocks += 1
        else:
            if math.isfinite(ms):
                sum_makespan_non_dl += ms
                cnt_non_dl += 1

    if cnt_non_dl > 0:
        avg_makespan = sum_makespan_non_dl / cnt_non_dl
    else:
        # 该 rule + buffers 在所有实例上都 deadlock
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
# 4. 某个 rule 上跑所有 candidate buffers
# ============================================================

def _evaluate_all_buffers_for_rule(
    rule: str,
    candidate_buffers: List[List[int]],
    val_instances: List[InstanceData],
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    对给定 rule，在所有 candidate_buffers 上做评估。
    """
    results: List[Dict[str, Any]] = []

    print(f"[RULE] Evaluating rule='{rule}' on {len(candidate_buffers)} buffer candidates...")
    for idx, bufs in enumerate(candidate_buffers):
        res = _evaluate_one_buffer_for_rule(
            rule=rule,
            buffers=bufs,
            instances=val_instances,
            max_steps=max_steps,
            seed=seed + idx,  # 不同 buffers 稍微平移一下 seed
        )
        res["buffer_index"] = idx
        results.append(res)

    return results


# ============================================================
# 5. 写 CSV：details + summary
# ============================================================

def _write_details_csv(
    all_results: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    写入 rule_baseline_details.csv：
      rule, buffers, avg_makespan, deadlock_rate, episodes, num_deadlocks,
      buffer_index, rank_within_rule
    """
    if not all_results:
        print("[WARN] _write_details_csv: all_results is empty, skip.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rule_baseline_details.csv"

    # 对 (rule, deadlock_rate, avg_makespan) 排序
    all_sorted = sorted(
        all_results,
        key=lambda r: (r["rule"], r["deadlock_rate"], r["avg_makespan"]),
    )

    # 为每个 rule 分配 rank_within_rule
    rank_counter: Dict[str, int] = {}
    for r in all_sorted:
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
        for row in all_sorted:
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


def _write_summary_csv(
    best_per_rule: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    写入 rule_baseline_summary.csv：
      rule, best_buffers, avg_makespan, deadlock_rate, episodes, num_deadlocks
    """
    if not best_per_rule:
        print("[WARN] _write_summary_csv: best_per_rule is empty, skip.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rule_baseline_summary.csv"

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
# 6. 对外接口：run_group1_for_experiment
# ============================================================

def run_group1_for_experiment(
    experiment_name: str,
    rules: List[str],
    base_seed: int = 0,
    max_steps: int = 10_000,
) -> None:
    """
    Group1：在指定 experiment_name 的 val 集上，
    对 rules 中的每条规则做 “固定缓存 + 规则派工” 离线枚举评估。

    每次调用保存结果到：
      results/<experiment_name>/rule_baseline/<run_id>/
        - rule_baseline_details.csv
        - rule_baseline_summary.csv
    """
    rules = [r.strip().lower() for r in rules if r.strip()]
    if not rules:
        print("[ERROR] run_group1_for_experiment: empty rules, skip.")
        return

    print(
        f"[GROUP1] experiment_name={experiment_name}, "
        f"rules={rules}, base_seed={base_seed}, max_steps={max_steps}"
    )

    # 输出目录（带时间戳）
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(ROOT_DIR) / "results" / experiment_name / "rule_baseline" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GROUP1] Results will be saved under: {out_dir}")

    # 1) 加载 val instances
    val_instances = _load_val_instances(experiment_name)
    if not val_instances:
        print("[ERROR] No val instances loaded, abort Group1 for", experiment_name)
        return

    # 2) 计算全局 buffer 上界 + 枚举候选 buffers
    upper_bounds = _compute_global_buffer_upper_bounds(val_instances)
    if not upper_bounds:
        print("[ERROR] Failed to compute buffer upper bounds, abort Group1 for", experiment_name)
        return

    candidate_buffers = _enumerate_candidate_buffers(upper_bounds)
    if not candidate_buffers:
        print("[ERROR] No candidate buffers generated, abort Group1 for", experiment_name)
        return

    print(
        f"[GROUP1] Global buffer upper bounds = {upper_bounds}, "
        f"num_candidates = {len(candidate_buffers)}"
    )

    # 3) 对每个 rule 评估
    all_results: List[Dict[str, Any]] = []
    best_per_rule: List[Dict[str, Any]] = []

    for rule in rules:
        res_rule = _evaluate_all_buffers_for_rule(
            rule=rule,
            candidate_buffers=candidate_buffers,
            val_instances=val_instances,
            max_steps=max_steps,
            seed=base_seed,
        )
        all_results.extend(res_rule)

        # 按 deadlock_rate -> avg_makespan 排序，取第一条作为该 rule 的 best
        res_rule_sorted = sorted(
            res_rule,
            key=lambda r: (r["deadlock_rate"], r["avg_makespan"]),
        )
        best = res_rule_sorted[0]
        best_per_rule.append(best)

        print(
            f"[GROUP1][BEST] rule={rule}, buffers={best['buffers']}, "
            f"avg_makespan={best['avg_makespan']:.3f}, "
            f"deadlock_rate={best['deadlock_rate']:.3f}, "
            f"episodes={best['episodes']}, "
            f"num_deadlocks={best['num_deadlocks']}"
        )

    # 4) 写 CSV
    _write_details_csv(all_results, out_dir)
    _write_summary_csv(best_per_rule, out_dir)

    print(f"[GROUP1] Done for experiment {experiment_name}.")

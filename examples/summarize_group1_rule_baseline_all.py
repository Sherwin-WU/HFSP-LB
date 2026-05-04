# examples/summarize_group1_rule_baseline_all.py
"""
汇总所有算例的 Group1 rule_baseline 结果（每算例 × 每条规则各自的最优一行）。

扫描目录结构：
  results/<experiment_name>/rule_baseline/<run_id>/rule_baseline_summary.csv

行为：
  - 对每个 experiment_name：
      * 找到 rule_baseline 下“最新的” run_id 子目录
      * 读取其中的 rule_baseline_summary.csv
      * summary 里的每一行（一个 rule 的最优 buffers）加上 experiment、run_id 字段
  - 把所有算例的行拼成一个总表：
      results/summary/group1_rule_baseline_all_rules.csv
"""

from __future__ import annotations

import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(ROOT_DIR) / "results"


def _find_latest_rule_baseline_summary_for_experiment(
    experiment_name: str,
) -> Dict[str, Any] | None:
    """
    在 results/<experiment_name>/rule_baseline/ 下找到最新的 run_id 目录，
    读取其中的 rule_baseline_summary.csv。

    返回：
      {
        "experiment": experiment_name,
        "run_id": run_id_str,
        "rows": [ {col: value, ...}, ... ]
      }
    若找不到则返回 None。
    """
    exp_dir = RESULTS_DIR / experiment_name
    rb_root = exp_dir / "rule_baseline"
    if not rb_root.exists() or not rb_root.is_dir():
        return None

    subdirs = [d for d in rb_root.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # run_id 一般是 YYYYmmdd_HHMMSS，字符串排序即时间排序
    subdirs_sorted = sorted(subdirs, key=lambda p: p.name)
    latest_dir = subdirs_sorted[-1]
    run_id = latest_dir.name

    summary_path = latest_dir / "rule_baseline_summary.csv"
    if not summary_path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))

    if not rows:
        return None

    return dict(
        experiment=experiment_name,
        run_id=run_id,
        rows=rows,
    )


def _scan_all_experiments() -> List[str]:
    """
    扫描 results/ 目录下有哪些 experiment_name（一级子目录）。
    """
    if not RESULTS_DIR.exists():
        return []
    exps: List[str] = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir():
            exps.append(item.name)
    return sorted(exps)


def summarize_group1_rule_baseline_all() -> None:
    experiments = _scan_all_experiments()
    if not experiments:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Experiments found for Group1 summary:")
    for e in experiments:
        print(f"  - {e}")

    all_rule_rows: List[Dict[str, Any]] = []

    for exp in experiments:
        info = _find_latest_rule_baseline_summary_for_experiment(exp)
        if info is None:
            print(f"[WARN] No rule_baseline_summary.csv for experiment {exp}, skip.")
            continue

        experiment = info["experiment"]
        run_id = info["run_id"]
        rows = info["rows"]

        print(
            f"[INFO] Using rule_baseline_summary.csv from "
            f"results/{experiment}/rule_baseline/{run_id}"
        )

        for r in rows:
            # summary 的列：rule, best_buffers, avg_makespan, deadlock_rate, episodes, num_deadlocks
            row = {
                "experiment": experiment,
                "run_id": run_id,
                "rule": r.get("rule", ""),
                "best_buffers": r.get("best_buffers", ""),
                "avg_makespan": r.get("avg_makespan", ""),
                "deadlock_rate": r.get("deadlock_rate", ""),
                "episodes": r.get("episodes", ""),
                "num_deadlocks": r.get("num_deadlocks", ""),
            }
            all_rule_rows.append(row)

    summary_dir = RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "group1_rule_baseline_all_rules.csv"
    if not all_rule_rows:
        print("[WARN] No rows collected, nothing to write.")
        return

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "run_id",
                "rule",
                "best_buffers",
                "avg_makespan",
                "deadlock_rate",
                "episodes",
                "num_deadlocks",
            ]
        )
        for r in all_rule_rows:
            writer.writerow(
                [
                    r["experiment"],
                    r["run_id"],
                    r["rule"],
                    r["best_buffers"],
                    r["avg_makespan"],
                    r["deadlock_rate"],
                    r["episodes"],
                    r["num_deadlocks"],
                ]
            )

    print(f"[SAVE] Group1 all-rules summary -> {out_path}")


if __name__ == "__main__":
    summarize_group1_rule_baseline_all()

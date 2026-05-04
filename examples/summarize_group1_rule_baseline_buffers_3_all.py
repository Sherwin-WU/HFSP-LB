# examples/summarize_group1_rule_baseline_buffers_3_all.py
"""
汇总 Group1 中“缓存容量全为 3”的规则结果到一张表。

扫描目录结构：
  results/<experiment>/rule_baseline/<run_id>/rule_baseline_details.csv

对每个 experiment：
  - 找到 rule_baseline 下“最新的” run_id 子目录
  - 读取其中的 rule_baseline_details.csv
  - 只保留 buffers ∈ {[3,3], [3,3,3], [3,3,3,3]} 的行

最终输出：
  results/summary/group1_rule_baseline_buffers_3_all_rules.csv

输出列：
  experiment, run_id, rule, buffers, avg_makespan, deadlock_rate,
  episodes, num_deadlocks, buffer_index, rank_within_rule
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(ROOT_DIR) / "results"


def _scan_experiments() -> List[str]:
    """扫描 results/ 下有哪些 experiment（一级子目录名）。"""
    if not RESULTS_DIR.exists():
        return []
    exps: List[str] = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir():
            exps.append(item.name)
    return sorted(exps)


def _find_latest_rule_baseline_details_for_experiment(
    experiment_name: str,
) -> Dict[str, Any] | None:
    """
    在 results/<experiment_name>/rule_baseline/ 下找到最新的 run_id 目录，
    读取其中的 rule_baseline_details.csv。

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

    details_path = latest_dir / "rule_baseline_details.csv"
    if not details_path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    with details_path.open("r", newline="") as f:
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


def _parse_buffers_str(buf_str: str) -> List[int]:
    """
    把 buffers 字符串（如 '3 3 3' 或 '[3, 3, 3]'）解析成 List[int]。
    """
    s = buf_str.strip()
    # 去掉可能的方括号和逗号
    s = s.replace("[", "").replace("]", "").replace(",", " ")
    parts = [p for p in s.split() if p]
    res: List[int] = []
    for p in parts:
        try:
            res.append(int(p))
        except ValueError:
            pass
    return res


def summarize_group1_buffers_3_all() -> None:
    """
    主函数：筛选所有 buffers 为 [3,3] / [3,3,3] / [3,3,3,3] 的行，汇总到一张表。
    """
    experiments = _scan_experiments()
    if not experiments:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Experiments found for Group1 buffers=3 summary:")
    for e in experiments:
        print(f"  - {e}")

    target_patterns = {
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    }

    all_rows: List[Dict[str, Any]] = []

    for exp in experiments:
        info = _find_latest_rule_baseline_details_for_experiment(exp)
        if info is None:
            print(f"[WARN] No rule_baseline_details.csv for experiment {exp}, skip.")
            continue

        experiment = info["experiment"]
        run_id = info["run_id"]
        rows = info["rows"]

        print(
            f"[INFO] Using rule_baseline_details.csv from "
            f"results/{experiment}/rule_baseline/{run_id}"
        )

        for r in rows:
            buf_str = r.get("buffers", "")
            buf_list = _parse_buffers_str(buf_str)
            key = tuple(buf_list)
            if key not in target_patterns:
                continue

            # 兼容列名：
            # rule, buffers, avg_makespan, deadlock_rate, episodes,
            # num_deadlocks, buffer_index, rank_within_rule
            row = {
                "experiment": experiment,
                "run_id": run_id,
                "rule": r.get("rule", ""),
                "buffers": buf_str,
                "avg_makespan": r.get("avg_makespan", ""),
                "deadlock_rate": r.get("deadlock_rate", ""),
                "episodes": r.get("episodes", ""),
                "num_deadlocks": r.get("num_deadlocks", ""),
                "buffer_index": r.get("buffer_index", ""),
                "rank_within_rule": r.get("rank_within_rule", ""),
            }
            all_rows.append(row)

    summary_dir = RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "group1_rule_baseline_buffers_3_all_rules.csv"
    if not all_rows:
        print("[WARN] No rows with buffers in {[3,3],[3,3,3],[3,3,3,3]} found, nothing to write.")
        return

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "run_id",
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
        for r in all_rows:
            writer.writerow(
                [
                    r["experiment"],
                    r["run_id"],
                    r["rule"],
                    r["buffers"],
                    r["avg_makespan"],
                    r["deadlock_rate"],
                    r["episodes"],
                    r["num_deadlocks"],
                    r["buffer_index"],
                    r["rank_within_rule"],
                ]
            )

    print(f"[SAVE] Group1 buffers=3 summary -> {out_path}")


if __name__ == "__main__":
    summarize_group1_buffers_3_all()

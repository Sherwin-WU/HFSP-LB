# examples/summarize_group4_two_level_summary_all.py
"""
汇总 Group4（Joint D3QN_LB，两层缓冲+调度）已完成种子的 test 评估结果。

扫描目录结构：
  results/<experiment>/group4_two_level/seed<seed>_<run_id>/

对每个 (experiment, seed, run_id)：
  - 读取 eval_test_summary_detail.csv 或 eval_test_summary.csv
    * 若存在列 'split'，只保留 split == 'test' 的那一行
    * 否则默认取第一行
  - 输出一行汇总记录

输出文件：
  results/summary/group4_two_level_eval_summary_all.csv

列字段：
  experiment, seed, run_id,
  split, num_eval_episodes, avg_makespan, deadlock_rate, num_deadlocks
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------- 路径设置 ----------
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


def _parse_seed_and_runid_from_dirname(name: str) -> Tuple[int | None, str]:
    """
    从 seed 目录名 seed<seed>_<run_id> 中解析出 (seed, run_id)。

    例如：
      'seed3_20251202_184927' -> (3, '20251202_184927')
    """
    if not name.startswith("seed"):
        return None, ""
    rest = name[4:]
    if "_" in rest:
        seed_str, run_id = rest.split("_", 1)
    else:
        seed_str, run_id = rest, ""
    try:
        seed = int(seed_str)
    except ValueError:
        seed = None
    return seed, run_id


def _load_test_summary(summary_path: Path) -> Dict[str, Any] | None:
    """
    从 eval_test_summary_*.csv 里取出 test 行（若有 split 列），否则取第一行。
    返回包含原字段的 dict；若失败则返回 None。
    """
    if not summary_path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))

    if not rows:
        return None

    # 若有 split 列，优先找 test 行
    if "split" in rows[0]:
        for r in rows:
            if str(r.get("split", "")).strip().lower() == "test":
                return r
        return rows[0]
    else:
        return rows[0]


def summarize_group4_two_level_summary_all() -> None:
    experiments = _scan_experiments()
    if not experiments:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Experiments found for Group4 summary:")
    for e in experiments:
        print(f"  - {e}")

    all_rows: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_dir = RESULTS_DIR / exp
        method_dir = exp_dir / "group4_two_level"
        if not method_dir.exists() or not method_dir.is_dir():
            # 该 experiment 还没有 Group4 结果，直接跳过
            continue

        print(f"[INFO] Scanning Group4 results in {method_dir}")

        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed, run_id = _parse_seed_and_runid_from_dirname(seed_dir.name)
            if seed is None:
                continue

            # 兼容两种命名：eval_test_summary_detail.csv 或 eval_test_summary.csv
            summary_path1 = seed_dir / "eval_test_summary_detail.csv"
            summary_path2 = seed_dir / "eval_test_summary.csv"
            if summary_path1.exists():
                summary_path = summary_path1
            elif summary_path2.exists():
                summary_path = summary_path2
            else:
                print(f"[WARN] No eval_test_summary CSV in {seed_dir}, skip.")
                continue

            summary_row = _load_test_summary(summary_path)
            if summary_row is None:
                print(f"[WARN] Empty summary file: {summary_path}, skip.")
                continue

            split = summary_row.get("split", "test")
            num_eps = (
                summary_row.get("num_eval_episodes")
                or summary_row.get("episodes")
                or ""
            )
            avg_ms = summary_row.get("avg_makespan", "")
            dead_rate = summary_row.get("deadlock_rate", "")
            num_dead = summary_row.get("num_deadlocks", "")

            row = {
                "experiment": exp,
                "seed": seed,
                "run_id": run_id,
                "split": split,
                "num_eval_episodes": num_eps,
                "avg_makespan": avg_ms,
                "deadlock_rate": dead_rate,
                "num_deadlocks": num_dead,
            }
            all_rows.append(row)

    summary_dir = RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "group4_two_level_eval_summary_all.csv"
    if not all_rows:
        print("[WARN] No Group4 summary rows collected, nothing to write.")
        return

    fieldnames = [
        "experiment",
        "seed",
        "run_id",
        "split",
        "num_eval_episodes",
        "avg_makespan",
        "deadlock_rate",
        "num_deadlocks",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"[SAVE] Group4 summary over available seeds -> {out_path}")


if __name__ == "__main__":
    summarize_group4_two_level_summary_all()

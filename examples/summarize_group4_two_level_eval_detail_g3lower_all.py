# examples/summarize_group4_two_level_eval_detail_g3lower_all.py
"""
汇总 Group4 近似结果（upper=Group4, lower=Group3_dispatch）中
所有 eval_test_detail_g3lower.csv 到一张总表。

扫描目录结构：
  results/<experiment>/group4_two_level/seed<seed>_<run_id>/eval_test_detail_g3lower.csv

对每个 experiment / seed / run_id：
  - 读取 eval_test_detail_g3lower.csv
    表头：episode, instance_idx, buffers, makespan, deadlock
  - 每行前面加上 experiment, seed, run_id
  - 追加到总表

输出：
  results/summary/group4_two_level_eval_test_detail_g3lower_all.csv
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------- 路径设置 ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(ROOT_DIR) / "results"


def _scan_experiments() -> List[str]:
    """扫描 results/ 下有哪些 experiment（一级子目录名）。"""
    if not RESULTS_DIR.exists():
        return []
    return sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()])


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


def summarize_group4_two_level_eval_detail_g3lower_all() -> None:
    experiments = _scan_experiments()
    if not experiments:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Experiments found for Group4 g3lower eval_detail summary:")
    for e in experiments:
        print(f"  - {e}")

    all_rows: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_dir = RESULTS_DIR / exp
        method_dir = exp_dir / "group4_two_level"
        if not method_dir.exists() or not method_dir.is_dir():
            # 该算例还没有 Group4 结果
            continue

        print(f"[INFO] Scanning Group4 g3lower eval_detail in {method_dir}")

        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            seed, run_id = _parse_seed_and_runid_from_dirname(seed_dir.name)
            if seed is None:
                continue

            detail_path = seed_dir / "eval_test_detail_g3lower.csv"
            if not detail_path.exists():
                print(f"[WARN] eval_test_detail_g3lower.csv not found in {seed_dir}, skip.")
                continue

            print(
                f"[INFO] Reading detail_g3lower: exp={exp}, seed={seed}, run_id={run_id}"
            )

            with detail_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    row = {
                        "experiment": exp,
                        "seed": seed,
                        "run_id": run_id,
                        "episode": r.get("episode", ""),
                        "instance_idx": r.get("instance_idx", ""),
                        "buffers": r.get("buffers", ""),
                        "makespan": r.get("makespan", ""),
                        "deadlock": r.get("deadlock", ""),
                    }
                    all_rows.append(row)

    summary_dir = RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "group4_two_level_eval_test_detail_g3lower_all.csv"
    if not all_rows:
        print("[WARN] No eval_test_detail_g3lower rows collected, nothing to write.")
        return

    fieldnames = [
        "experiment",
        "seed",
        "run_id",
        "episode",
        "instance_idx",
        "buffers",
        "makespan",
        "deadlock",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"[SAVE] Group4 g3lower eval_test_detail all-in-one -> {out_path}")


if __name__ == "__main__":
    summarize_group4_two_level_eval_detail_g3lower_all()

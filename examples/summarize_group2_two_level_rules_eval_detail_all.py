# examples/summarize_group2_two_level_rules_eval_detail_all.py
"""
汇总 Group2（BA + Rule）所有 eval_test_detail 结果到一张大表。

扫描目录结构：
  results/<experiment>/group2_two_level_<rule>/seed<seed>_<run_id>/eval_test_detail.csv

行为：
  - 对每个 experiment（results 的一级子目录）：
      * 对每个方法目录 group2_two_level_<rule>：
      * 对每个 seed 目录 seed<seed>_<run_id>：
          - 若存在 eval_test_detail.csv，则读取其中所有行
          - 每行额外加上 experiment, rule, seed, run_id 四个字段
  - 所有行纵向拼接，输出到：
      results/summary/group2_two_level_rules_eval_test_detail_all.csv

注意：
  - 不做任何统计 / 均值，只是原样汇总。
  - 假设 eval_test_detail.csv 的表头为：
        episode, instance_idx, buffers, makespan, deadlock
"""

from __future__ import annotations

import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

# ---------- 路径设置 ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = Path(ROOT_DIR) / "results"


def _scan_experiments() -> List[str]:
    """
    扫描 results/ 下有哪些 experiment（一级子目录名）。
    """
    if not RESULTS_DIR.exists():
        return []
    exps: List[str] = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir():
            exps.append(item.name)
    return sorted(exps)


def _iter_group2_method_dirs(exp_dir: Path):
    """
    遍历某个 experiment 目录下的所有 group2_two_level_* 方法目录。
    """
    for item in exp_dir.iterdir():
        if item.is_dir() and item.name.startswith("group2_two_level_"):
            yield item


def _parse_rule_from_method_dirname(name: str) -> str:
    """
    从目录名 group2_two_level_<rule> 中解析出 rule。
    """
    prefix = "group2_two_level_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _parse_seed_and_runid_from_dirname(name: str):
    """
    从 seed 目录名 seed<seed>_<run_id> 中解析出 (seed, run_id)。

    示例：
      'seed3_20251202_184927' -> (3, '20251202_184927')
    """
    if not name.startswith("seed"):
        return None, None
    # 去掉前缀 'seed'
    rest = name[4:]
    # 按第一个 '_' 切一次
    if "_" in rest:
        seed_str, run_id = rest.split("_", 1)
    else:
        seed_str, run_id = rest, ""
    try:
        seed = int(seed_str)
    except ValueError:
        seed = None
    return seed, run_id


def summarize_group2_eval_test_detail_all() -> None:
    experiments = _scan_experiments()
    if not experiments:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Experiments found for Group2 eval_test_detail summary:")
    for e in experiments:
        print(f"  - {e}")

    all_rows: List[Dict[str, Any]] = []

    # ------------------ 主循环：exp -> rule -> seed ------------------
    for exp in experiments:
        exp_dir = RESULTS_DIR / exp

        for method_dir in _iter_group2_method_dirs(exp_dir):
            rule = _parse_rule_from_method_dirname(method_dir.name)

            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                seed, run_id = _parse_seed_and_runid_from_dirname(seed_dir.name)
                if seed is None:
                    # 不是 seed 目录，跳过
                    continue

                detail_path = seed_dir / "eval_test_detail.csv"
                if not detail_path.exists():
                    print(
                        f"[WARN] eval_test_detail.csv not found, skip: "
                        f"{detail_path}"
                    )
                    continue

                print(
                    f"[INFO] Reading detail: exp={exp}, rule={rule}, "
                    f"seed={seed}, run_id={run_id}"
                )

                with detail_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        row = dict(r)  # 原始列：episode, instance_idx, buffers, makespan, deadlock
                        # 加上标记字段
                        row["experiment"] = exp
                        row["rule"] = rule
                        row["seed"] = str(seed)
                        row["run_id"] = run_id
                        all_rows.append(row)

    # ------------------ 写总汇总表 ------------------
    summary_dir = RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "group2_two_level_rules_eval_test_detail_all.csv"
    if not all_rows:
        print("[WARN] No eval_test_detail rows collected, nothing to write.")
        return

    # 统一列顺序：先标记列，再原始 detail 列
    fieldnames = [
        "experiment",
        "rule",
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
            writer.writerow(
                {
                    "experiment": r.get("experiment", ""),
                    "rule": r.get("rule", ""),
                    "seed": r.get("seed", ""),
                    "run_id": r.get("run_id", ""),
                    "episode": r.get("episode", ""),
                    "instance_idx": r.get("instance_idx", ""),
                    "buffers": r.get("buffers", ""),
                    "makespan": r.get("makespan", ""),
                    "deadlock": r.get("deadlock", ""),
                }
            )

    print(f"[SAVE] Group2 eval_test_detail all-in-one -> {out_path}")


if __name__ == "__main__":
    summarize_group2_eval_test_detail_all()

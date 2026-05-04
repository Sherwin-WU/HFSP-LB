# examples/summarize_group2_ba_rule_j50s3m3.py
"""
组2结果汇总脚本：BA + Rule (Static BufferDesignEnv + Rule 下层)

假定目录结构为：
  results/<experiment_name>/ba_rule/<rule>/seed*/train_log.csv

脚本功能：
  1) 扫描所有 (rule, seed) run，读取 train_log.csv
  2) 在每个 run 的最后 K 个 outer_ep 上计算：
       - avg_makespan_lastK
       - deadlock_rate_lastK
       - avg_total_buffer_lastK
  3) 写出两张表到：
       results/<experiment_name>/ba_rule_summary/
         - group2_ba_rule_runs.csv          : 每个 (rule, seed) 一行
         - group2_ba_rule_rule_summary.csv  : 每个 rule 聚合多 seed 的 mean/std

用法（在项目根目录）：

  python examples/summarize_group2_ba_rule_j50s3m3.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import argparse
import csv
import math
import glob

import numpy as np  # 用于计算 mean/std

# ---- 路径设置：把 src/ 加进 sys.path（备用） ----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# ============================================================
# 1. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Group2 (BA+Rule) results over seeds."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="j50s3m3",
        help="Experiment name. Default: j50s3m3",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root dir for results (relative to project root). Default: results",
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=20,
        help="Number of last outer episodes to average over. Default: 20",
    )
    return parser.parse_args()


# ============================================================
# 2. 工具：读取 train_log.csv 并计算后 K 步指标
# ============================================================

def read_train_log(train_log_path: Path) -> List[Dict[str, Any]]:
    """
    读取 train_log.csv，返回每行一个 dict。
    字段应包括：
      outer_ep, final_makespan, deadlock, total_buffer, ...
    """
    rows: List[Dict[str, Any]] = []
    with train_log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_last_k_stats(
    rows: List[Dict[str, Any]],
    last_k: int,
) -> Dict[str, float]:
    """
    在 train_log 的后 last_k 行上，计算：
      - avg_makespan_lastK
      - deadlock_rate_lastK
      - avg_total_buffer_lastK

    若行数不足 last_k，则对全部行计算。
    若没有行，返回 NaN。
    """
    n = len(rows)
    if n == 0:
        return dict(
            avg_makespan_lastK=math.nan,
            deadlock_rate_lastK=math.nan,
            avg_total_buffer_lastK=math.nan,
        )

    start_idx = max(0, n - last_k)
    sub = rows[start_idx:]

    makespans: List[float] = []
    deadlocks: List[float] = []
    total_buffers: List[float] = []

    for r in sub:
        try:
            mk = float(r.get("final_makespan", math.nan))
        except ValueError:
            mk = math.nan
        try:
            dl = float(r.get("deadlock", math.nan))
        except ValueError:
            dl = math.nan
        try:
            tb = float(r.get("total_buffer", math.nan))
        except ValueError:
            tb = math.nan

        makespans.append(mk)
        deadlocks.append(dl)
        total_buffers.append(tb)

    def safe_mean(xs: List[float]) -> float:
        arr = np.array(xs, dtype=float)
        if arr.size == 0:
            return math.nan
        # 过滤掉 NaN
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return math.nan
        return float(arr.mean())

    avg_makespan = safe_mean(makespans)
    avg_deadlock_rate = safe_mean(deadlocks)  # deadlock 列原本就是 0/1，可直接当比例平均
    avg_total_buffer = safe_mean(total_buffers)

    return dict(
        avg_makespan_lastK=avg_makespan,
        deadlock_rate_lastK=avg_deadlock_rate,
        avg_total_buffer_lastK=avg_total_buffer,
    )


# ============================================================
# 3. 扫描所有 run
# ============================================================

def scan_group2_runs(
    experiment_name: str,
    results_root: str,
    last_k: int,
) -> List[Dict[str, Any]]:
    """
    扫描:
      <ROOT_DIR>/<results_root>/<experiment_name>/ba_rule/<rule>/seed*_* / train_log.csv

    返回每个 run 的统计结果列表：
      [
        {
          "rule": str,
          "seed": int,
          "run_dir": str,
          "num_episodes": int,
          "avg_makespan_lastK": float,
          "deadlock_rate_lastK": float,
          "avg_total_buffer_lastK": float,
        },
        ...
      ]
    """
    base_dir = Path(ROOT_DIR) / results_root / experiment_name / "ba_rule"
    if not base_dir.exists():
        print(f"[WARN] base_dir not found: {base_dir}")
        return []

    results: List[Dict[str, Any]] = []

    # 规则目录：results/<exp>/ba_rule/<rule>/
    for rule_dir in sorted(base_dir.glob("*")):
        if not rule_dir.is_dir():
            continue
        rule_name = rule_dir.name.lower()

        # seed run 目录：seedX_YYYYMMDD_HHMMSS
        pattern = str(rule_dir / "seed*_*")
        run_dirs = sorted(glob.glob(pattern))
        if not run_dirs:
            print(f"[INFO] No runs found under rule={rule_name}, dir={rule_dir}")
            continue

        for rd in run_dirs:
            run_path = Path(rd)
            train_log_path = run_path / "train_log.csv"
            if not train_log_path.exists():
                print(f"[WARN] train_log.csv not found: {train_log_path}")
                continue

            # 从目录名中粗略解析 seed（形如 seed3_2025xxxx）
            seed_str = run_path.name.split("_")[0]  # "seed3"
            try:
                if seed_str.startswith("seed"):
                    seed = int(seed_str[4:])
                else:
                    seed = int(seed_str)
            except ValueError:
                seed = -1

            rows = read_train_log(train_log_path)
            stats = compute_last_k_stats(rows, last_k=last_k)
            num_episodes = len(rows)

            run_info: Dict[str, Any] = dict(
                rule=rule_name,
                seed=seed,
                run_dir=str(run_path),
                num_episodes=num_episodes,
            )
            run_info.update(stats)
            results.append(run_info)

            print(
                f"[RUN] rule={rule_name}, seed={seed}, "
                f"episodes={num_episodes}, "
                f"avg_ms_lastK={stats['avg_makespan_lastK']:.3f}, "
                f"dl_rate_lastK={stats['deadlock_rate_lastK']:.3f}, "
                f"buf_lastK={stats['avg_total_buffer_lastK']:.3f}"
            )

    return results


# ============================================================
# 4. 写两张汇总表
# ============================================================

def write_runs_csv(
    runs: List[Dict[str, Any]],
    out_dir: Path,
) -> Path:
    """
    写明细表：每个 (rule, seed) 一行。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "group2_ba_rule_runs.csv"

    if not runs:
        print(f"[WARN] No runs to write in {csv_path}")
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rule",
                    "seed",
                    "run_dir",
                    "num_episodes",
                    "avg_makespan_lastK",
                    "deadlock_rate_lastK",
                    "avg_total_buffer_lastK",
                ]
            )
        return csv_path

    # 固定列顺序
    fieldnames = [
        "rule",
        "seed",
        "run_dir",
        "num_episodes",
        "avg_makespan_lastK",
        "deadlock_rate_lastK",
        "avg_total_buffer_lastK",
    ]

    # 按 rule, seed 排个序便于阅读
    runs_sorted = sorted(runs, key=lambda r: (r["rule"], r["seed"]))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs_sorted:
            writer.writerow(r)

    print(f"[SAVE] run-level summary -> {csv_path}")
    return csv_path


def write_rule_summary_csv(
    runs: List[Dict[str, Any]],
    out_dir: Path,
) -> Path:
    """
    按 rule 聚合多个 seed，写 rule-level 的 mean/std 表。
    输出列：
      rule, num_seeds,
      avg_makespan_mean, avg_makespan_std,
      deadlock_rate_mean, deadlock_rate_std,
      avg_total_buffer_mean, avg_total_buffer_std
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "group2_ba_rule_rule_summary.csv"

    if not runs:
        print(f"[WARN] No runs to aggregate in {csv_path}")
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rule",
                    "num_seeds",
                    "avg_makespan_mean",
                    "avg_makespan_std",
                    "deadlock_rate_mean",
                    "deadlock_rate_std",
                    "avg_total_buffer_mean",
                    "avg_total_buffer_std",
                ]
            )
        return csv_path

    # 按 rule 分组
    by_rule: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        rule = r["rule"]
        by_rule.setdefault(rule, []).append(r)

    fieldnames = [
        "rule",
        "num_seeds",
        "avg_makespan_mean",
        "avg_makespan_std",
        "deadlock_rate_mean",
        "deadlock_rate_std",
        "avg_total_buffer_mean",
        "avg_total_buffer_std",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rule, rlist in sorted(by_rule.items(), key=lambda x: x[0]):
            def collect(key: str) -> np.ndarray:
                vals: List[float] = []
                for rr in rlist:
                    v = rr.get(key, math.nan)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = math.nan
                    vals.append(v)
                arr = np.array(vals, dtype=float)
                arr = arr[~np.isnan(arr)]
                return arr

            ms_arr = collect("avg_makespan_lastK")
            dl_arr = collect("deadlock_rate_lastK")
            buf_arr = collect("avg_total_buffer_lastK")

            def safe_mean_std(arr: np.ndarray) -> tuple[float, float]:
                if arr.size == 0:
                    return math.nan, math.nan
                return float(arr.mean()), float(arr.std(ddof=0))

            ms_mean, ms_std = safe_mean_std(ms_arr)
            dl_mean, dl_std = safe_mean_std(dl_arr)
            buf_mean, buf_std = safe_mean_std(buf_arr)

            writer.writerow(
                dict(
                    rule=rule,
                    num_seeds=len(rlist),
                    avg_makespan_mean=ms_mean,
                    avg_makespan_std=ms_std,
                    deadlock_rate_mean=dl_mean,
                    deadlock_rate_std=dl_std,
                    avg_total_buffer_mean=buf_mean,
                    avg_total_buffer_std=buf_std,
                )
            )

    print(f"[SAVE] rule-level summary -> {csv_path}")
    return csv_path


# ============================================================
# 5. main
# ============================================================

def main():
    args = parse_args()

    experiment_name = args.experiment_name
    results_root = args.results_root
    last_k = args.last_k

    print(f"[INFO] Summarizing Group2 (BA+Rule) results")
    print(f"[INFO] experiment_name = {experiment_name}")
    print(f"[INFO] results_root    = {results_root}")
    print(f"[INFO] last_k episodes = {last_k}")

    # 扫描所有 run
    runs = scan_group2_runs(
        experiment_name=experiment_name,
        results_root=results_root,
        last_k=last_k,
    )

    # 输出目录：results/<exp>/ba_rule_summary/
    out_dir = Path(ROOT_DIR) / results_root / experiment_name / "ba_rule_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 run 级别和 rule 级别汇总
    write_runs_csv(runs, out_dir)
    write_rule_summary_csv(runs, out_dir)

    print("[DONE] Group2 summary finished.")


if __name__ == "__main__":
    main()

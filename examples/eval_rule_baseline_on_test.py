# examples/eval_rule_baseline_on_test.py
"""
在 j50s3m3 的 test 集上，评估 rule_baseline_summary.csv 中
每个 rule 的 best buffers 的真实表现。

用法：
  1) 自动查找最新 summary（推荐）：
     python examples/eval_rule_baseline_on_test.py

  2) 手动指定 summary：
     python examples/eval_rule_baseline_on_test.py \
       --summary_csv results/j50s3m3/rule_baseline/2025xxxx_xxxxxx/rule_baseline_summary.csv

输出：
  results/<experiment_name>/rule_baseline_test/<run_id>/rule_baseline_test.csv
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import csv
import math
from typing import List, Dict, Any

import argparse
import glob

import numpy as np  # noqa: F401  # 预留需要时使用

# ---- 路径设置：把 src/ 加进 sys.path ----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from instances.io import load_instances_from_dir  # type: ignore
from instances.types import InstanceData          # type: ignore
from envs.ffs_core_env import simulate_instance_with_job_rule  # type: ignore


# ============================================================
# 1. 加载 test instances
# ============================================================

def load_test_instances(experiment_name: str) -> List[InstanceData]:
    data_root = os.path.join(ROOT_DIR, "experiments", "raw", experiment_name)
    test_dir = os.path.join(data_root, "test")
    test_instances = load_instances_from_dir(test_dir)
    print(f"[INFO] Loaded {experiment_name} test instances: {len(test_instances)}")
    return test_instances


# ============================================================
# 2. 自动查找最新的 summary csv
# ============================================================

def find_latest_summary_csv(experiment_name: str) -> Path | None:
    """
    在 results/<exp>/rule_baseline/*/rule_baseline_summary.csv 中
    找到最近修改时间的那个文件。
    """
    base_dir = Path(ROOT_DIR) / "results" / experiment_name / "rule_baseline"
    pattern = str(base_dir / "*" / "rule_baseline_summary.csv")
    candidates = glob.glob(pattern)

    if not candidates:
        print(f"[WARN] No summary csv found under: {base_dir}")
        return None

    # 选 mtime 最大的
    latest_path = max(candidates, key=lambda p: os.path.getmtime(p))
    latest_path = Path(latest_path)
    print(f"[INFO] Auto-detected latest summary_csv: {latest_path}")
    return latest_path


# ============================================================
# 3. 读取 summary
# ============================================================

def parse_buffers_str(buf_str: str) -> List[int]:
    """
    把 summary 里的 best_buffers 字段（例如 "3 3" 或 "[3, 3]"）解析为 List[int]。
    """
    if not buf_str:
        return []

    s = buf_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    if not s:
        return []

    parts = s.split()
    bufs: List[int] = []
    for p in parts:
        try:
            bufs.append(int(p))
        except ValueError:
            p2 = p.strip(",")
            if p2:
                bufs.append(int(p2))
    return bufs


def load_rule_summary(summary_csv_path: Path) -> List[Dict[str, Any]]:
    """
    从 rule_baseline_summary.csv 中读取
      rule, best_buffers
    作为后续在 test 上评估的配置。
    """
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"summary_csv not found: {summary_csv_path}")

    configs: List[Dict[str, Any]] = []
    with summary_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rule = row.get("rule", "").strip().lower()
            if not rule:
                continue
            buf_str = row.get("best_buffers") or row.get("buffers") or ""
            buffers = parse_buffers_str(buf_str)
            configs.append(dict(rule=rule, buffers=buffers))

    if not configs:
        print(f"[WARN] No valid configs loaded from {summary_csv_path}")
    else:
        print("[INFO] Loaded best (rule, buffers) configs from summary:")
        for c in configs:
            print(f"  rule={c['rule']}, buffers={c['buffers']}")

    return configs


# ============================================================
# 4. 在 test 集上评估单个 (rule, buffers)
# ============================================================

def evaluate_on_test_for_rule(
    rule: str,
    buffers: List[int],
    test_instances: List[InstanceData],
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    num_instances = len(test_instances)
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

    for idx, inst in enumerate(test_instances):
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
# 5. 写 test 结果 CSV
# ============================================================

def write_test_csv(
    results: List[Dict[str, Any]],
    out_dir: str | Path,
) -> Path:
    if not results:
        print("[WARN] write_test_csv: empty results, skip.")
        return Path(out_dir) / "rule_baseline_test.csv"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rule_baseline_test.csv"

    results_sorted = sorted(results, key=lambda r: r["rule"])

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
            ]
        )
        for row in results_sorted:
            buf_str = " ".join(str(b) for b in row["buffers"])
            writer.writerow(
                [
                    row["rule"],
                    buf_str,
                    row["avg_makespan"],
                    row["deadlock_rate"],
                    row["episodes"],
                    row["num_deadlocks"],
                ]
            )

    print(f"[SAVE] rule_baseline_test.csv -> {csv_path}")
    return csv_path


# ============================================================
# 6. main()
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Path to rule_baseline_summary.csv. "
             "If not set, auto-detect the latest one under "
             "results/<experiment_name>/rule_baseline/*/.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="j50s3m3",
        help="Experiment name (default: j50s3m3)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10_000,
        help="Max steps for simulate_instance_with_job_rule (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for test csv. "
             "If not set, use results/<exp>/rule_baseline_test/<run_id>/",
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name
    max_steps = args.max_steps
    base_seed = args.seed

    # ---- 1. 确定 summary_csv 路径 ----
    if args.summary_csv is not None:
        summary_csv_path = Path(args.summary_csv)
        print(f"[INFO] Using user-specified summary_csv: {summary_csv_path}")
    else:
        summary_csv_path = find_latest_summary_csv(experiment_name)
        if summary_csv_path is None:
            print("[ERROR] Failed to auto-detect summary_csv, abort.")
            return

    # ---- 2. 输出目录 ----
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            ROOT_DIR,
            "results",
            experiment_name,
            "rule_baseline_test",
            run_id,
        )

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Test results will be saved to: {out_dir}")

    # ---- 3. 读取 summary configs ----
    configs = load_rule_summary(summary_csv_path)
    if not configs:
        print("[ERROR] No valid configs found in summary_csv, abort.")
        return

    # ---- 4. 加载 test 实例 ----
    test_instances = load_test_instances(experiment_name)
    if not test_instances:
        print("[ERROR] No test instances loaded, abort.")
        return

    # ---- 5. 对每个 (rule, buffers) 做 test 评估 ----
    all_results: List[Dict[str, Any]] = []
    for cfg in configs:
        rule = cfg["rule"]
        buffers = cfg["buffers"]
        print(f"[EVAL] rule={rule}, buffers={buffers} on test set ...")

        res = evaluate_on_test_for_rule(
            rule=rule,
            buffers=buffers,
            test_instances=test_instances,
            max_steps=max_steps,
            seed=base_seed,
        )
        all_results.append(res)

        print(
            f"       -> avg_makespan={res['avg_makespan']:.3f}, "
            f"deadlock_rate={res['deadlock_rate']:.3f}, "
            f"episodes={res['episodes']}, "
            f"num_deadlocks={res['num_deadlocks']}"
        )

    # ---- 6. 写 CSV ----
    write_test_csv(all_results, out_dir)

    print("[DONE] rule baseline evaluation on test set finished.")


if __name__ == "__main__":
    main()

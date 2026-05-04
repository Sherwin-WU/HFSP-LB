# examples/summarize_upper_arch_ablation.py

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 复用 train_two_level 里的 ROOT_DIR
from train_two_level import ROOT_DIR


def find_best_val_row(val_log_path: Path) -> Dict[str, float]:
    """
    从 val_log.csv 中找出“最佳一行”：
    - 按 (deadlock_rate, avg_makespan, avg_total_buffer) 字典序最小。
    返回一个 dict，包含 avg_makespan / avg_total_buffer / deadlock_rate。
    """
    if not val_log_path.exists():
        raise FileNotFoundError(f"val_log.csv not found: {val_log_path}")

    best_row: Dict[str, float] = {}
    best_key: Tuple[float, float, float] | None = None

    with val_log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                avg_ms = float(row["avg_makespan"])
                avg_buf = float(row["avg_total_buffer"])
                dead = float(row["deadlock_rate"])
            except (KeyError, ValueError):
                continue

            key = (dead, avg_ms, avg_buf)
            if best_key is None or key < best_key:
                best_key = key
                best_row = dict(
                    avg_makespan=avg_ms,
                    avg_total_buffer=avg_buf,
                    deadlock_rate=dead,
                    outer_ep=int(row.get("outer_ep", -1)),
                    episodes=int(row.get("episodes", 0)),
                )

    if best_key is None:
        raise RuntimeError(f"No valid rows found in {val_log_path}")

    return best_row


def summarize_arch_ablation():
    """
    扫描 results/j50s3m3/upper_arch_ablation 下所有子目录，
    汇总 3×3 (algo_type × replay_type) 的表现：
    - 对每个组合，在不同 seed 上计算 mean/std（avg_makespan, avg_total_buffer, deadlock_rate）。
    """
    experiment_name = "j50s3m3"
    algo_name = "upper_arch_ablation"
    results_root = Path(ROOT_DIR) / "results" / experiment_name / algo_name

    if not results_root.exists():
        raise FileNotFoundError(f"results root not found: {results_root}")

    print(f"[INFO] summarizing results under: {results_root}")

    # 我们预期的算法 & buffer 类型列表（用于固定表格顺序）
    algo_list = ["dqn", "ddqn", "d3qn"]
    replay_list = ["uniform", "per", "nstep"]

    # 收集所有 run 的结果
    # 每个元素：{algo_type, replay_type, seed, best_*}
    all_runs: List[Dict[str, float]] = []

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue

        cfg_path = run_dir / "cfg.json"
        val_log_path = run_dir / "val_log.csv"

        if not cfg_path.exists() or not val_log_path.exists():
            print(f"[WARN] skip {run_dir}, missing cfg.json or val_log.csv")
            continue

        # 1) 读取 cfg.json，获取 algo_type / replay_type / seed
        with cfg_path.open("r", encoding="utf-8") as f_cfg:
            cfg = json.load(f_cfg)

        # upper_agent_cfg 在 cfg.json 里的路径：cfg["upper_agent_cfg"]["algo_type"]
        upper_cfg = cfg.get("upper_agent_cfg", {})
        algo_type = str(upper_cfg.get("algo_type", "dqn")).lower()
        replay_type = str(upper_cfg.get("replay_type", "uniform")).lower()

        # seed 可以从 instance_cfg.seed 取，也可以从目录名解析，取其一即可
        inst_cfg = cfg.get("instance_cfg", {})
        seed = inst_cfg.get("seed", None)

        # 如果 seed 没写进 cfg.json，就试着从目录名解析：..._seedX
        if seed is None:
            name = run_dir.name
            if "seed" in name:
                try:
                    seed = int(name.split("seed")[-1])
                except ValueError:
                    seed = -1
            else:
                seed = -1

        # 2) 读取 val_log.csv，找出最佳一行
        try:
            best_row = find_best_val_row(val_log_path)
        except Exception as e:
            print(f"[WARN] failed to parse {val_log_path}: {e}")
            continue

        all_runs.append(
            dict(
                algo_type=algo_type,
                replay_type=replay_type,
                seed=int(seed),
                best_avg_makespan=float(best_row["avg_makespan"]),
                best_avg_total_buffer=float(best_row["avg_total_buffer"]),
                best_deadlock_rate=float(best_row["deadlock_rate"]),
            )
        )

    if not all_runs:
        print("[WARN] no runs found to summarize.")
        return

    # 按 (algo_type, replay_type) 分组
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for r in all_runs:
        key = (r["algo_type"], r["replay_type"])
        grouped.setdefault(key, []).append(r)

    # 输出表头
    print("\n=== Upper Architecture Ablation Summary (best on VAL, mean±std over seeds) ===")
    header = (
        f"{'algo':<6} {'replay':<8} "
        f"{'ms_mean':>10} {'ms_std':>10} "
        f"{'buf_mean':>10} {'buf_std':>10} "
        f"{'dead_mean':>10} {'dead_std':>10} "
        f"{'n_seeds':>7}"
    )
    print(header)
    print("-" * len(header))

    # 同时准备写到 CSV
    summary_rows = []

    for algo in algo_list:
        for replay in replay_list:
            key = (algo, replay)
            runs = grouped.get(key, [])

            if not runs:
                row_str = (
                    f"{algo:<6} {replay:<8} "
                    f"{'NaN':>10} {'NaN':>10} "
                    f"{'NaN':>10} {'NaN':>10} "
                    f"{'NaN':>10} {'NaN':>10} "
                    f"{0:>7}"
                )
                print(row_str)
                summary_rows.append(
                    dict(
                        algo_type=algo,
                        replay_type=replay,
                        n_seeds=0,
                        ms_mean=np.nan,
                        ms_std=np.nan,
                        buf_mean=np.nan,
                        buf_std=np.nan,
                        dead_mean=np.nan,
                        dead_std=np.nan,
                    )
                )
                continue

            ms_arr = np.array([r["best_avg_makespan"] for r in runs], dtype=np.float64)
            buf_arr = np.array([r["best_avg_total_buffer"] for r in runs], dtype=np.float64)
            dead_arr = np.array([r["best_deadlock_rate"] for r in runs], dtype=np.float64)

            ms_mean, ms_std = ms_arr.mean(), ms_arr.std(ddof=0)
            buf_mean, buf_std = buf_arr.mean(), buf_arr.std(ddof=0)
            dead_mean, dead_std = dead_arr.mean(), dead_arr.std(ddof=0)

            row_str = (
                f"{algo:<6} {replay:<8} "
                f"{ms_mean:10.3f} {ms_std:10.3f} "
                f"{buf_mean:10.3f} {buf_std:10.3f} "
                f"{dead_mean:10.3f} {dead_std:10.3f} "
                f"{len(runs):7d}"
            )
            print(row_str)

            summary_rows.append(
                dict(
                    algo_type=algo,
                    replay_type=replay,
                    n_seeds=len(runs),
                    ms_mean=float(ms_mean),
                    ms_std=float(ms_std),
                    buf_mean=float(buf_mean),
                    buf_std=float(buf_std),
                    dead_mean=float(dead_mean),
                    dead_std=float(dead_std),
                )
            )

    # 写入一个汇总 CSV
    summary_path = results_root / "summary_arch_ablation.csv"
    with summary_path.open("w", newline="") as f_sum:
        writer = csv.DictWriter(
            f_sum,
            fieldnames=[
                "algo_type",
                "replay_type",
                "n_seeds",
                "ms_mean",
                "ms_std",
                "buf_mean",
                "buf_std",
                "dead_mean",
                "dead_std",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\n[INFO] summary CSV written to: {summary_path}")


if __name__ == "__main__":
    summarize_arch_ablation()

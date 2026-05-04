# examples/summarize_upper_reward_ablation.py
"""
Step3: 上层 reward 消融试验 结果汇总脚本。

目录结构假定为：
    results/j50s3m3/reward_ablation/
        d3qn_uniform_lam0p5_pen1000_seed0/
            cfg.json
            offline_buffer_eval_val.csv
            ...
        d3qn_uniform_lam0p5_pen1000_seed1/
        ...

功能：
    - 对每个 run：
        * 从 cfg.json 中读出 lambda_B (buffer_cost_weight)
          和 deadlock_penalty
        * 从 offline_buffer_eval_val.csv 中挑出:
            - 若有 deadlock_rate == 0 的行，则在这些行中选 avg_makespan 最小的一行；
            - 否则，在全表中选 avg_makespan 最小的一行；
          得到 (avg_makespan, avg_total_buffer, deadlock_rate)

    - 按 (lambda_B, deadlock_penalty) 聚合所有 seed：
        * 计算 ms_mean/std, buf_mean/std, dead_mean/std

    - 输出 CSV：
        results/j50s3m3/reward_ablation/summary_reward_ablation.csv

    - 在终端打印按 dead_mean → ms_mean 排序的前若干个组合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


ROOT_DIR = Path("results/j50s3m3/reward_ablation")
SUMMARY_CSV = ROOT_DIR / "summary_reward_ablation.csv"


@dataclass
class RunMetrics:
    lambda_B: float
    deadlock_penalty: float
    seed: int
    avg_makespan: float
    avg_total_buffer: float
    deadlock_rate: float


@dataclass
class RewardSummary:
    lambda_B: float
    deadlock_penalty: float
    num_seeds: int
    ms_mean: float
    ms_std: float
    buf_mean: float
    buf_std: float
    dead_mean: float
    dead_std: float


def parse_seed_from_dirname(name: str) -> int | None:
    """
    从目录名解析 seed，约定为 ..._seed{N}。
    """
    parts = name.split("_")
    for p in parts:
        if p.startswith("seed"):
            try:
                return int(p[4:])
            except ValueError:
                return None
    return None


def load_lambda_and_penalty_from_cfg(cfg_path: Path) -> Tuple[float, float]:
    """
    从 cfg.json 读取 buffer_cost_weight 和 deadlock_penalty。
    """
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ua = cfg.get("upper_agent_cfg", {})
    lam = float(ua.get("buffer_cost_weight", 0.5))
    pen = float(ua.get("deadlock_penalty", 1000.0))
    return lam, pen


def load_best_offline_metrics(offline_path: Path) -> Tuple[float, float, float]:
    """
    从 offline_buffer_eval_val.csv 中取“最好”的那一行，并返回：
        (avg_makespan, avg_total_buffer, deadlock_rate)

    策略：
        1) 若有 deadlock_rate == 0 的行：
            在这些行中选 avg_makespan 最小的一行。
        2) 否则，在全表中选 avg_makespan 最小的一行。
    """
    df = pd.read_csv(offline_path)

    if "avg_makespan" not in df.columns:
        raise ValueError(f"{offline_path} 中没有 'avg_makespan' 列。")

    if "avg_total_buffer" not in df.columns:
        df["avg_total_buffer"] = np.nan
    if "deadlock_rate" not in df.columns:
        df["deadlock_rate"] = np.nan

    safe = df[df["deadlock_rate"] == 0]
    if not safe.empty:
        best = safe.loc[safe["avg_makespan"].idxmin()]
    else:
        best = df.loc[df["avg_makespan"].idxmin()]

    ms = float(best["avg_makespan"])
    buf = float(best["avg_total_buffer"])
    dead = float(best["deadlock_rate"])
    return ms, buf, dead


def collect_all_runs(root: Path) -> List[RunMetrics]:
    """
    扫描 ROOT_DIR 下所有子目录，收集 RunMetrics。
    """
    runs: List[RunMetrics] = []

    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {root}")

    for d in root.iterdir():
        if not d.is_dir():
            continue

        seed = parse_seed_from_dirname(d.name)
        if seed is None:
            # 非实验目录，跳过
            continue

        cfg_path = d / "cfg.json"
        offline_path = d / "offline_buffer_eval_val.csv"

        if not offline_path.exists():
            print(f"[WARN] {d} 缺少 offline_buffer_eval_val.csv，跳过。")
            continue
        if not cfg_path.exists():
            print(f"[WARN] {d} 缺少 cfg.json，跳过。")
            continue

        try:
            lam, pen = load_lambda_and_penalty_from_cfg(cfg_path)
            ms, buf, dead = load_best_offline_metrics(offline_path)
        except Exception as e:
            print(f"[WARN] 解析 {d} 失败: {e}，跳过。")
            continue

        runs.append(
            RunMetrics(
                lambda_B=lam,
                deadlock_penalty=pen,
                seed=seed,
                avg_makespan=ms,
                avg_total_buffer=buf,
                deadlock_rate=dead,
            )
        )

    return runs


def summarize_by_reward(runs: List[RunMetrics]) -> List[RewardSummary]:
    """
    按 (lambda_B, deadlock_penalty) 聚合所有 runs。
    """
    # 分组 key: (lambda_B, deadlock_penalty)
    groups: Dict[Tuple[float, float], List[RunMetrics]] = {}
    for r in runs:
        key = (r.lambda_B, r.deadlock_penalty)
        groups.setdefault(key, []).append(r)

    summaries: List[RewardSummary] = []

    for (lam, pen), rs in sorted(groups.items()):
        ms_arr = np.array([r.avg_makespan for r in rs], dtype=np.float32)
        buf_arr = np.array([r.avg_total_buffer for r in rs], dtype=np.float32)
        dead_arr = np.array([r.deadlock_rate for r in rs], dtype=np.float32)

        summaries.append(
            RewardSummary(
                lambda_B=float(lam),
                deadlock_penalty=float(pen),
                num_seeds=len(rs),
                ms_mean=float(ms_arr.mean()),
                ms_std=float(ms_arr.std(ddof=1)) if len(ms_arr) > 1 else 0.0,
                buf_mean=float(buf_arr.mean()),
                buf_std=float(buf_arr.std(ddof=1)) if len(buf_arr) > 1 else 0.0,
                dead_mean=float(dead_arr.mean()),
                dead_std=float(dead_arr.std(ddof=1)) if len(dead_arr) > 1 else 0.0,
            )
        )

    return summaries


def main():
    print(f"[INFO] ROOT_DIR = {ROOT_DIR}")

    runs = collect_all_runs(ROOT_DIR)
    if not runs:
        print("[WARN] 未找到任何有效 run（含 cfg.json + offline_buffer_eval_val.csv）。")
        return

    summaries = summarize_by_reward(runs)

    df = pd.DataFrame([asdict(s) for s in summaries])
    # 按 dead_mean -> ms_mean -> buf_mean 排序
    df = df.sort_values(["dead_mean", "ms_mean", "buf_mean"]).reset_index(drop=True)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] summary saved to: {SUMMARY_CSV}")

    print("\n====== Top reward configs (sorted by dead_mean -> ms_mean) ======")
    for _, row in df.head(20).iterrows():
        print(
            f"lambda={row['lambda_B']:.3f}, pen={row['deadlock_penalty']:.1f}, "
            f"seeds={int(row['num_seeds'])}, "
            f"dead_mean={row['dead_mean']:.4f}, dead_std={row['dead_std']:.4f}, "
            f"ms_mean={row['ms_mean']:.3f}, ms_std={row['ms_std']:.3f}, "
            f"buf_mean={row['buf_mean']:.3f}, buf_std={row['buf_std']:.3f}"
        )


if __name__ == "__main__":
    main()

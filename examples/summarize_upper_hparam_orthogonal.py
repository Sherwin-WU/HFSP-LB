# examples/summarize_upper_hparam_orthogonal.py
"""
第二组实验（超参 L27 正交试验）结果汇总脚本。

目录结构假定为：
    results/j50s3m3/hparam_orthogonal/
        design_01_row1_seed0/
            cfg.json
            offline_buffer_eval_val.csv
            ...
        design_01_row1_seed1/
        ...
        design_27_row27_seed9/

功能：
    - 汇总每个 design(row) 在 10 个 seed 上的表现：
        * 对每个 run：
            - 从 offline_buffer_eval_val.csv 中选 "最好" 的 buffer：
                · 优先选 deadlock_rate == 0 中 avg_makespan 最小的一行；
                · 如无无死锁方案，则选全表 avg_makespan 最小的一行。
            - 记录该行的 avg_makespan / avg_total_buffer / deadlock_rate。
        * 对每个 row 聚合所有 seed：
            - 计算 ms_mean/std, buf_mean/std, dead_mean/std。
        * 同时从 cfg.json 读取该 row 的 9 个超参值，保证一目了然。

    - 输出 summary CSV：
        results/j50s3m3/hparam_orthogonal/summary_hparam_orthogonal.csv

    - 在终端打印一个按 "deadlock_mean → makespan_mean" 排序的简要排名。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


ROOT_DIR = Path("results/j50s3m3/hparam_orthogonal")
SUMMARY_CSV = ROOT_DIR / "summary_hparam_orthogonal.csv"


@dataclass
class RunMetrics:
    row_id: int
    seed: int
    avg_makespan: float
    avg_total_buffer: float
    deadlock_rate: float


@dataclass
class DesignSummary:
    row_id: int
    num_seeds: int

    # 9 个超参
    gamma: float
    lr: float
    batch_size: int
    buffer_capacity: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_rate: float
    hidden_dim: int

    # 聚合指标
    ms_mean: float
    ms_std: float
    buf_mean: float
    buf_std: float
    dead_mean: float
    dead_std: float


def parse_row_seed_from_dirname(name: str) -> Tuple[int | None, int | None]:
    """
    从目录名解析 row_id 和 seed。
    约定：目录名形如 'design_01_row1_seed0'。
    """
    if not name.startswith("design_"):
        return None, None
    parts = name.split("_")
    row_id = None
    seed = None
    for p in parts:
        if p.startswith("row"):
            try:
                row_id = int(p[3:])
            except ValueError:
                pass
        if p.startswith("seed"):
            try:
                seed = int(p[4:])
            except ValueError:
                pass
    return row_id, seed


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

    # 补全可能缺失的列
    if "avg_total_buffer" not in df.columns:
        df["avg_total_buffer"] = np.nan
    if "deadlock_rate" not in df.columns:
        df["deadlock_rate"] = np.nan

    # 先找 deadlock_rate == 0 的子集
    safe = df[df["deadlock_rate"] == 0]
    if not safe.empty:
        best = safe.loc[safe["avg_makespan"].idxmin()]
    else:
        best = df.loc[df["avg_makespan"].idxmin()]

    ms = float(best["avg_makespan"])
    buf = float(best["avg_total_buffer"])
    dead = float(best["deadlock_rate"])

    return ms, buf, dead


def load_hparams_from_cfg(cfg_path: Path) -> Dict[str, Any]:
    """
    从 cfg.json 读取 upper_agent_cfg 下的 9 个超参值。
    """
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ua = cfg.get("upper_agent_cfg", {})

    return dict(
        gamma=float(ua.get("gamma", 0.99)),
        lr=float(ua.get("lr", 1e-4)),
        batch_size=int(ua.get("batch_size", 64)),
        buffer_capacity=int(ua.get("buffer_capacity", 50000)),
        target_update_interval=int(ua.get("target_update_interval", 200)),
        epsilon_start=float(ua.get("epsilon_start", 0.5)),
        epsilon_end=float(ua.get("epsilon_end", 0.01)),
        epsilon_decay_rate=float(ua.get("epsilon_decay_rate", 0.995)),
        hidden_dim=int(ua.get("hidden_dim", 128)),
    )


def collect_all_runs(root: Path) -> Tuple[Dict[int, List[RunMetrics]], Dict[int, Dict[str, Any]]]:
    """
    扫描 ROOT_DIR 下所有 design_* 子目录，收集：
        - 每个 row_id 的所有 RunMetrics
        - 每个 row_id 对应的一份 hyperparams（从第一个 cfg.json 中取）
    """
    row_to_runs: Dict[int, List[RunMetrics]] = {}
    row_to_hparams: Dict[int, Dict[str, Any]] = {}

    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {root}")

    for d in root.iterdir():
        if not d.is_dir():
            continue

        row_id, seed = parse_row_seed_from_dirname(d.name)
        if row_id is None or seed is None:
            continue

        cfg_path = d / "cfg.json"
        offline_path = d / "offline_buffer_eval_val.csv"

        if not offline_path.exists():
            print(f"[WARN] {d} 缺少 offline_buffer_eval_val.csv，跳过该 run。")
            continue

        # 记录 best offline metrics
        try:
            ms, buf, dead = load_best_offline_metrics(offline_path)
        except Exception as e:
            print(f"[WARN] 解析 {offline_path} 失败: {e}，跳过该 run。")
            continue

        metric = RunMetrics(
            row_id=row_id,
            seed=seed,
            avg_makespan=ms,
            avg_total_buffer=buf,
            deadlock_rate=dead,
        )
        row_to_runs.setdefault(row_id, []).append(metric)

        # 记录该 row 的超参（只取一次）
        if row_id not in row_to_hparams and cfg_path.exists():
            hparams = load_hparams_from_cfg(cfg_path)
            row_to_hparams[row_id] = hparams

    return row_to_runs, row_to_hparams


def summarize_designs(
    row_to_runs: Dict[int, List[RunMetrics]],
    row_to_hparams: Dict[int, Dict[str, Any]],
) -> List[DesignSummary]:
    """
    对每个 row_id 聚合所有 RunMetrics，生成 DesignSummary。
    """
    summaries: List[DesignSummary] = []

    for row_id, runs in sorted(row_to_runs.items()):
        ms_list = np.array([r.avg_makespan for r in runs], dtype=np.float32)
        buf_list = np.array([r.avg_total_buffer for r in runs], dtype=np.float32)
        dead_list = np.array([r.deadlock_rate for r in runs], dtype=np.float32)

        h = row_to_hparams.get(row_id, {})

        summary = DesignSummary(
            row_id=row_id,
            num_seeds=len(runs),
            gamma=float(h.get("gamma", np.nan)),
            lr=float(h.get("lr", np.nan)),
            batch_size=int(h.get("batch_size", -1)),
            buffer_capacity=int(h.get("buffer_capacity", -1)),
            target_update_interval=int(h.get("target_update_interval", -1)),
            epsilon_start=float(h.get("epsilon_start", np.nan)),
            epsilon_end=float(h.get("epsilon_end", np.nan)),
            epsilon_decay_rate=float(h.get("epsilon_decay_rate", np.nan)),
            hidden_dim=int(h.get("hidden_dim", -1)),
            ms_mean=float(ms_list.mean()),
            ms_std=float(ms_list.std(ddof=1)) if len(ms_list) > 1 else 0.0,
            buf_mean=float(buf_list.mean()),
            buf_std=float(buf_list.std(ddof=1)) if len(buf_list) > 1 else 0.0,
            dead_mean=float(dead_list.mean()),
            dead_std=float(dead_list.std(ddof=1)) if len(dead_list) > 1 else 0.0,
        )
        summaries.append(summary)

    return summaries


def main():
    print(f"[INFO] ROOT_DIR = {ROOT_DIR}")
    row_to_runs, row_to_hparams = collect_all_runs(ROOT_DIR)

    if not row_to_runs:
        print("[WARN] 没有找到任何有效的 design_* 子目录（含 offline_buffer_eval_val.csv）。")
        return

    summaries = summarize_designs(row_to_runs, row_to_hparams)

    # 转成 DataFrame 保存
    df = pd.DataFrame([asdict(s) for s in summaries])
    df = df.sort_values(["dead_mean", "ms_mean", "buf_mean"]).reset_index(drop=True)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] summary saved to: {SUMMARY_CSV}")

    # 在终端打印前若干个“最好”的 design，方便快速查看
    print("\n====== Top designs (sorted by dead_mean -> ms_mean) ======")
    for _, row in df.head(10).iterrows():
        print(
            f"row={int(row['row_id'])}, seeds={int(row['num_seeds'])}, "
            f"dead_mean={row['dead_mean']:.3f}, ms_mean={row['ms_mean']:.3f}, "
            f"buf_mean={row['buf_mean']:.3f}, "
            f"gamma={row['gamma']}, lr={row['lr']}, batch={int(row['batch_size'])}, "
            f"tu={int(row['target_update_interval'])}, "
            f"eps=({row['epsilon_start']}, {row['epsilon_end']}, {row['epsilon_decay_rate']}), "
            f"hidden_dim={int(row['hidden_dim'])}"
        )


if __name__ == "__main__":
    main()

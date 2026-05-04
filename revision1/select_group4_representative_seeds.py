# revision1/select_group4_representative_seeds.py
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# 配置
# ============================================================

TARGET_EXPERIMENTS = ["j200s3", "j200s4", "j200s5"]


# ============================================================
# dataclass
# ============================================================

@dataclass
class SeedRunRecord:
    experiment: str
    seed: int
    run_dir: str
    summary_csv: str
    cfg_json: Optional[str]
    avg_makespan: float
    deadlock_rate: float
    has_upper_ckpt: bool
    has_lower_ckpt: bool


@dataclass
class SelectedSeedResult:
    experiment: str
    selected_seed: int
    selected_run_dir: str
    selected_avg_makespan: float
    selected_deadlock_rate: float
    median_rank_position_1based: int
    num_candidates: int
    note: str


# ============================================================
# 基础工具
# ============================================================

def _safe_float(x: Any, default: float = math.inf) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_summary_csv(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Empty summary csv: {summary_path}")

    row = rows[0]
    avg_makespan = _safe_float(row.get("avg_makespan"), math.inf)
    deadlock_rate = _safe_float(row.get("deadlock_rate"), math.inf)

    return {
        "avg_makespan": avg_makespan,
        "deadlock_rate": deadlock_rate,
        "raw_row": row,
    }


def _read_cfg_seed(cfg_json_path: Path) -> int:
    with cfg_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return _safe_int(obj.get("random_seed"), default=-1)


def _parse_seed_from_dirname(run_dir: Path) -> int:
    # 目录格式通常为：seed{seed}_YYYYMMDD_HHMMSS
    name = run_dir.name
    if not name.startswith("seed"):
        return -1
    i = 4
    digits = []
    while i < len(name) and name[i].isdigit():
        digits.append(name[i])
        i += 1
    if not digits:
        return -1
    return int("".join(digits))


def _find_run_dirs_for_experiment(
    results_root: Path,
    experiment: str,
    method_name: str,
) -> List[Path]:
    exp_dir = results_root / experiment / method_name
    if not exp_dir.exists():
        return []
    return sorted([p for p in exp_dir.iterdir() if p.is_dir()])


def collect_seed_run_records(
    results_root: Path,
    experiment: str,
    method_name: str,
) -> List[SeedRunRecord]:
    records: List[SeedRunRecord] = []

    for run_dir in _find_run_dirs_for_experiment(results_root, experiment, method_name):
        summary_csv = run_dir / "eval_test_summary_detail.csv"
        cfg_json = run_dir / "cfg.json"
        upper_ckpt = run_dir / "upper_q_last.pth"
        lower_ckpt = run_dir / "lower_q_last.pth"

        if not summary_csv.exists():
            continue

        try:
            summ = _read_summary_csv(summary_csv)
        except Exception as e:
            print(f"[WARN] Skip bad summary: {summary_csv} ({e})")
            continue

        seed = -1
        if cfg_json.exists():
            try:
                seed = _read_cfg_seed(cfg_json)
            except Exception:
                seed = _parse_seed_from_dirname(run_dir)
        else:
            seed = _parse_seed_from_dirname(run_dir)

        rec = SeedRunRecord(
            experiment=experiment,
            seed=seed,
            run_dir=str(run_dir),
            summary_csv=str(summary_csv),
            cfg_json=str(cfg_json) if cfg_json.exists() else None,
            avg_makespan=float(summ["avg_makespan"]),
            deadlock_rate=float(summ["deadlock_rate"]),
            has_upper_ckpt=upper_ckpt.exists(),
            has_lower_ckpt=lower_ckpt.exists(),
        )
        records.append(rec)

    return records


def deduplicate_by_seed_keep_best(records: Sequence[SeedRunRecord]) -> List[SeedRunRecord]:
    """
    若同一 seed 有多次运行，仅保留：
      1) deadlock_rate 更低
      2) avg_makespan 更低
      3) run_dir 字典序更靠后（通常时间更晚）
    """
    best_by_seed: Dict[int, SeedRunRecord] = {}
    for rec in records:
        prev = best_by_seed.get(rec.seed)
        if prev is None:
            best_by_seed[rec.seed] = rec
            continue

        prev_key = (prev.deadlock_rate, prev.avg_makespan, prev.run_dir)
        curr_key = (rec.deadlock_rate, rec.avg_makespan, rec.run_dir)
        if curr_key < prev_key:
            best_by_seed[rec.seed] = rec

    out = list(best_by_seed.values())
    out.sort(key=lambda r: (r.deadlock_rate, r.avg_makespan, r.seed, r.run_dir))
    return out


def select_median_seed(records: Sequence[SeedRunRecord]) -> SelectedSeedResult:
    if not records:
        raise RuntimeError("No candidate records.")

    # 先按死锁率、makespan、seed 排序
    ordered = sorted(records, key=lambda r: (r.deadlock_rate, r.avg_makespan, r.seed, r.run_dir))
    n = len(ordered)

    # 中位位置：n=10 时取第 5 个（1-based），即索引 4；如需第 6 个可改这里
    median_idx = (n - 1) // 2
    picked = ordered[median_idx]

    note_parts = []
    if not picked.has_upper_ckpt:
        note_parts.append("missing upper_q_last.pth")
    if not picked.has_lower_ckpt:
        note_parts.append("missing lower_q_last.pth")
    note = "; ".join(note_parts) if note_parts else "ok"

    return SelectedSeedResult(
        experiment=picked.experiment,
        selected_seed=picked.seed,
        selected_run_dir=picked.run_dir,
        selected_avg_makespan=picked.avg_makespan,
        selected_deadlock_rate=picked.deadlock_rate,
        median_rank_position_1based=median_idx + 1,
        num_candidates=n,
        note=note,
    )


def write_candidates_csv(out_path: Path, records: Sequence[SeedRunRecord]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "seed",
            "run_dir",
            "avg_makespan",
            "deadlock_rate",
            "has_upper_ckpt",
            "has_lower_ckpt",
            "summary_csv",
            "cfg_json",
        ])
        for r in records:
            writer.writerow([
                r.experiment,
                r.seed,
                r.run_dir,
                r.avg_makespan,
                r.deadlock_rate,
                int(r.has_upper_ckpt),
                int(r.has_lower_ckpt),
                r.summary_csv,
                r.cfg_json or "",
            ])


def write_selection_csv(out_path: Path, selections: Sequence[SelectedSeedResult]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "selected_seed",
            "selected_run_dir",
            "selected_avg_makespan",
            "selected_deadlock_rate",
            "median_rank_position_1based",
            "num_candidates",
            "note",
        ])
        for s in selections:
            writer.writerow([
                s.experiment,
                s.selected_seed,
                s.selected_run_dir,
                s.selected_avg_makespan,
                s.selected_deadlock_rate,
                s.median_rank_position_1based,
                s.num_candidates,
                s.note,
            ])


def write_selection_json(out_path: Path, selections: Sequence[SelectedSeedResult]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in selections]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="扫描 group4 已有结果，为 j80s3/j80s4/j80s5 选取代表 seed（中位代表 seed）。"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results-initial",
        help="项目 results-initial 根目录，例如 results/",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="group4_two_level",
        help="group4 方法目录名。",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=TARGET_EXPERIMENTS,
        help="要扫描的实验名，默认 j80s3 j80s4 j80s5。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/revision1_j200_selection",
        help="输出目录。",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_selections: List[SelectedSeedResult] = []

    for exp_name in args.experiments:
        raw_records = collect_seed_run_records(
            results_root=results_root,
            experiment=exp_name,
            method_name=args.method_name,
        )
        records = deduplicate_by_seed_keep_best(raw_records)

        if not records:
            print(f"[WARN] No valid records found: experiment={exp_name}")
            continue

        cand_csv = out_dir / f"{exp_name}_seed_candidates.csv"
        write_candidates_csv(cand_csv, records)

        sel = select_median_seed(records)
        all_selections.append(sel)

        print(
            f"[SELECT] {exp_name}: seed={sel.selected_seed}, "
            f"deadlock_rate={sel.selected_deadlock_rate:.6f}, "
            f"avg_makespan={sel.selected_avg_makespan:.6f}, "
            f"rank={sel.median_rank_position_1based}/{sel.num_candidates}, "
            f"note={sel.note}"
        )

    summary_csv = out_dir / "selected_j200_stage_models.csv"
    summary_json = out_dir / "selected_j200_stage_models.json"
    write_selection_csv(summary_csv, all_selections)
    write_selection_json(summary_json, all_selections)

    print(f"[SAVE] selection csv -> {summary_csv}")
    print(f"[SAVE] selection json -> {summary_json}")


if __name__ == "__main__":
    main()

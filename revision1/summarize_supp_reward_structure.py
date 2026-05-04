# revision1/summarize_supp_reward_structure.py
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# 路径与配置
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "revision1"

# 只扫这轮 revision1 reward structure 相关的方法
DEFAULT_METHOD_ALLOWLIST = {
    "group4_two_level_dense_epi",
    "group4_two_level_terminal_only",
    "group4_two_level_dense_only",
}

# 若 cfg.json 中已有 lower_reward_scheme，则优先使用 cfg；
# 若没有，再从 method_name 推断
DEFAULT_REWARD_SCHEME_BY_METHOD = {
    "group4_two_level_dense_epi": "dense_epi",
    "group4_two_level_terminal_only": "terminal_only",
    "group4_two_level_dense_only": "dense_only",
}

# 训练日志最后 K 行统计，用来粗看训练末段状态
TRAIN_LAST_K = 50


# ============================================================
# 基础工具
# ============================================================

def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def mean_or_nan(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    if not vals:
        return math.nan
    return float(mean(vals))


def std_or_nan(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    if not vals:
        return math.nan
    if len(vals) == 1:
        return 0.0
    return float(stdev(vals))


def parse_seed_and_runid(dirname: str) -> Tuple[Optional[int], str]:
    """
    支持：
      seed0_20260309_153000
      seed3
    """
    if not dirname.startswith("seed"):
        return None, ""
    rest = dirname[4:]
    if "_" in rest:
        seed_str, run_id = rest.split("_", 1)
    else:
        seed_str, run_id = rest, ""
    try:
        seed = int(seed_str)
    except ValueError:
        return None, ""
    return seed, run_id


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def find_summary_csv(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "eval_test_summary_detail.csv",
        run_dir / "eval_test_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_detail_csv(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "eval_test_detail.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def infer_reward_scheme(method_name: str, cfg: Dict[str, Any]) -> str:
    # 1) 先从 cfg 顶层取
    if "lower_reward_scheme" in cfg:
        s = str(cfg.get("lower_reward_scheme", "")).strip()
        if s:
            return s

    # 2) 再从 cfg 的 method_name 推
    if method_name in DEFAULT_REWARD_SCHEME_BY_METHOD:
        return DEFAULT_REWARD_SCHEME_BY_METHOD[method_name]

    # 3) 再根据名字兜底
    name = method_name.lower()
    if "terminal" in name:
        return "terminal_only"
    if "dense_only" in name or name.endswith("_dense"):
        return "dense_only"
    return "dense_epi"


def parse_buffers_to_total_buffer(buf_str: Any) -> float:
    """
    兼容：
      "2 3 1"
      "[2, 3, 1]"
      "(2,3,1)"
      "2,3,1"
    """
    if buf_str is None:
        return math.nan

    s = str(buf_str).strip()
    if s == "":
        return math.nan

    nums = re.findall(r"-?\d+", s)
    if not nums:
        return math.nan

    vals = [int(x) for x in nums]
    return float(sum(vals))


# ============================================================
# 读取单个 run 的各类指标
# ============================================================

def load_eval_summary_metrics(summary_path: Path) -> Dict[str, float]:
    rows = load_csv_dicts(summary_path)
    if not rows:
        return {
            "num_eval_episodes": math.nan,
            "avg_makespan": math.nan,
            "deadlock_rate": math.nan,
            "num_deadlocks": math.nan,
        }

    row = rows[0]
    if "split" in row:
        test_rows = [
            r for r in rows
            if str(r.get("split", "")).strip().lower() == "test"
        ]
        if test_rows:
            row = test_rows[0]

    num_eval_episodes = safe_float(
        row.get("num_eval_episodes", row.get("episodes", math.nan))
    )
    avg_makespan = safe_float(row.get("avg_makespan", math.nan))
    deadlock_rate = safe_float(row.get("deadlock_rate", math.nan))

    # 某些 summary 没有 num_deadlocks
    num_deadlocks = safe_float(row.get("num_deadlocks", math.nan))
    if math.isnan(num_deadlocks) and not math.isnan(num_eval_episodes) and not math.isnan(deadlock_rate):
        num_deadlocks = float(num_eval_episodes * deadlock_rate)

    return {
        "num_eval_episodes": num_eval_episodes,
        "avg_makespan": avg_makespan,
        "deadlock_rate": deadlock_rate,
        "num_deadlocks": num_deadlocks,
    }


def load_eval_detail_metrics(detail_path: Path) -> Dict[str, float]:
    rows = load_csv_dicts(detail_path)
    if not rows:
        return {
            "avg_total_buffer": math.nan,
            "std_total_buffer": math.nan,
            "num_deadlocks_from_detail": math.nan,
        }

    total_buffers: List[float] = []
    deadlocks: List[float] = []

    for row in rows:
        tb = parse_buffers_to_total_buffer(row.get("buffers"))
        if not math.isnan(tb):
            total_buffers.append(tb)

        if "deadlock" in row:
            deadlocks.append(safe_float(row.get("deadlock", 0.0), 0.0))

    avg_total_buffer = mean_or_nan(total_buffers)
    std_total_buffer = std_or_nan(total_buffers)
    num_deadlocks_from_detail = float(sum(deadlocks)) if deadlocks else math.nan

    return {
        "avg_total_buffer": avg_total_buffer,
        "std_total_buffer": std_total_buffer,
        "num_deadlocks_from_detail": num_deadlocks_from_detail,
    }


def load_train_log_metrics(train_log_path: Path, last_k: int = TRAIN_LAST_K) -> Dict[str, float]:
    rows = load_csv_dicts(train_log_path)
    if not rows:
        return {
            "train_rows": 0,
            "train_lastk_makespan": math.nan,
            "train_lastk_deadlock_rate": math.nan,
            "train_lastk_total_buffer": math.nan,
        }

    tail = rows[-last_k:] if len(rows) > last_k else rows

    makespans = [safe_float(r.get("final_makespan", math.nan)) for r in tail]
    deadlocks = [safe_float(r.get("deadlock", math.nan)) for r in tail]
    total_buffers = [safe_float(r.get("total_buffer", math.nan)) for r in tail]

    return {
        "train_rows": float(len(rows)),
        "train_lastk_makespan": mean_or_nan(makespans),
        "train_lastk_deadlock_rate": mean_or_nan(deadlocks),
        "train_lastk_total_buffer": mean_or_nan(total_buffers),
    }


def load_runtime_metrics(run_dir: Path) -> Dict[str, float]:
    """
    兼容后续 runtime 统计。
    若尚未生成 runtime_train.json / runtime_eval.csv，则返回 NaN。
    """
    train_wall_clock_sec = math.nan
    eval_wall_clock_sec = math.nan

    runtime_train_json = run_dir / "runtime_train.json"
    if runtime_train_json.exists():
        try:
            data = read_json(runtime_train_json)
            for key in [
                "train_wall_clock_sec",
                "wall_clock_sec",
                "total_train_wall_clock_sec",
            ]:
                if key in data:
                    train_wall_clock_sec = safe_float(data.get(key, math.nan))
                    break
        except Exception:
            pass

    runtime_eval_csv = run_dir / "runtime_eval.csv"
    if runtime_eval_csv.exists():
        try:
            rows = load_csv_dicts(runtime_eval_csv)
            vals: List[float] = []
            for row in rows:
                for key in [
                    "eval_wall_clock_sec",
                    "wall_clock_sec",
                    "total_eval_wall_clock_sec",
                    "instance_wall_clock_sec",
                ]:
                    if key in row:
                        vals.append(safe_float(row.get(key, math.nan)))
                        break
            eval_wall_clock_sec = mean_or_nan(vals)
        except Exception:
            pass

    return {
        "train_wall_clock_sec": train_wall_clock_sec,
        "eval_wall_clock_sec": eval_wall_clock_sec,
    }


# ============================================================
# 扫描与汇总
# ============================================================

def collect_runs(
    results_dir: Path,
    method_allowlist: Sequence[str] = tuple(DEFAULT_METHOD_ALLOWLIST),
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []

    if not results_dir.exists():
        print(f"[ERROR] results dir not found: {results_dir}")
        return all_rows

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name == "revision1":
            continue

        experiment = exp_dir.name

        for method_dir in sorted(exp_dir.iterdir()):
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            if method_allowlist and method_name not in method_allowlist:
                continue

            for run_dir in sorted(method_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                seed, run_id = parse_seed_and_runid(run_dir.name)
                if seed is None:
                    continue

                cfg_path = run_dir / "cfg.json"
                summary_path = find_summary_csv(run_dir)
                detail_path = find_detail_csv(run_dir)
                train_log_path = run_dir / "train_log.csv"

                if not cfg_path.exists():
                    print(f"[WARN] skip (no cfg.json): {run_dir}")
                    continue
                if summary_path is None:
                    print(f"[WARN] skip (no eval summary csv): {run_dir}")
                    continue

                try:
                    cfg = read_json(cfg_path)
                except Exception as e:
                    print(f"[WARN] bad cfg.json -> {run_dir}: {e}")
                    continue

                reward_scheme = infer_reward_scheme(method_name, cfg)
                lower_epi_reward_weight = safe_float(
                    cfg.get("lower_epi_reward_weight", math.nan)
                )

                summary_metrics = load_eval_summary_metrics(summary_path)

                detail_metrics = {
                    "avg_total_buffer": math.nan,
                    "std_total_buffer": math.nan,
                    "num_deadlocks_from_detail": math.nan,
                }
                if detail_path is not None and detail_path.exists():
                    detail_metrics = load_eval_detail_metrics(detail_path)

                train_metrics = {
                    "train_rows": math.nan,
                    "train_lastk_makespan": math.nan,
                    "train_lastk_deadlock_rate": math.nan,
                    "train_lastk_total_buffer": math.nan,
                }
                if train_log_path.exists():
                    train_metrics = load_train_log_metrics(train_log_path, TRAIN_LAST_K)

                runtime_metrics = load_runtime_metrics(run_dir)

                num_deadlocks = summary_metrics["num_deadlocks"]
                if math.isnan(num_deadlocks):
                    num_deadlocks = detail_metrics["num_deadlocks_from_detail"]

                row = {
                    "experiment": experiment,
                    "method_name": method_name,
                    "reward_scheme": reward_scheme,
                    "lower_epi_reward_weight": lower_epi_reward_weight,
                    "seed": seed,
                    "run_id": run_id,
                    "num_eval_episodes": summary_metrics["num_eval_episodes"],
                    "avg_makespan": summary_metrics["avg_makespan"],
                    "deadlock_rate": summary_metrics["deadlock_rate"],
                    "num_deadlocks": num_deadlocks,
                    "avg_total_buffer": detail_metrics["avg_total_buffer"],
                    "std_total_buffer": detail_metrics["std_total_buffer"],
                    "train_rows": train_metrics["train_rows"],
                    "train_lastk_makespan": train_metrics["train_lastk_makespan"],
                    "train_lastk_deadlock_rate": train_metrics["train_lastk_deadlock_rate"],
                    "train_lastk_total_buffer": train_metrics["train_lastk_total_buffer"],
                    "train_wall_clock_sec": runtime_metrics["train_wall_clock_sec"],
                    "eval_wall_clock_sec": runtime_metrics["eval_wall_clock_sec"],
                    "run_dir": str(run_dir),
                    "summary_path": str(summary_path),
                    "detail_path": str(detail_path) if detail_path is not None else "",
                    "train_log_path": str(train_log_path) if train_log_path.exists() else "",
                }
                all_rows.append(row)

    return all_rows


def aggregate_rows(by_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]] = {}

    for row in by_seed_rows:
        key = (
            str(row["experiment"]),
            str(row["reward_scheme"]),
            str(row["method_name"]),
            safe_float(row["lower_epi_reward_weight"], math.nan),
        )
        groups.setdefault(key, []).append(row)

    out_rows: List[Dict[str, Any]] = []

    for (experiment, reward_scheme, method_name, epi_weight), rows in sorted(groups.items()):
        avg_makespan_vals = [safe_float(r["avg_makespan"]) for r in rows]
        deadlock_rate_vals = [safe_float(r["deadlock_rate"]) for r in rows]
        num_deadlocks_vals = [safe_float(r["num_deadlocks"]) for r in rows]
        avg_total_buffer_vals = [safe_float(r["avg_total_buffer"]) for r in rows]

        train_lastk_makespan_vals = [safe_float(r["train_lastk_makespan"]) for r in rows]
        train_lastk_deadlock_rate_vals = [safe_float(r["train_lastk_deadlock_rate"]) for r in rows]
        train_lastk_total_buffer_vals = [safe_float(r["train_lastk_total_buffer"]) for r in rows]

        train_wall_clock_vals = [safe_float(r["train_wall_clock_sec"]) for r in rows]
        eval_wall_clock_vals = [safe_float(r["eval_wall_clock_sec"]) for r in rows]

        seeds = sorted({safe_int(r["seed"], -1) for r in rows if safe_int(r["seed"], -1) >= 0})

        out_rows.append({
            "experiment": experiment,
            "reward_scheme": reward_scheme,
            "method_name": method_name,
            "lower_epi_reward_weight": epi_weight,
            "num_runs": len(rows),
            "num_seeds": len(seeds),
            "avg_makespan_mean": mean_or_nan(avg_makespan_vals),
            "avg_makespan_std": std_or_nan(avg_makespan_vals),
            "deadlock_rate_mean": mean_or_nan(deadlock_rate_vals),
            "deadlock_rate_std": std_or_nan(deadlock_rate_vals),
            "num_deadlocks_mean": mean_or_nan(num_deadlocks_vals),
            "num_deadlocks_std": std_or_nan(num_deadlocks_vals),
            "avg_total_buffer_mean": mean_or_nan(avg_total_buffer_vals),
            "avg_total_buffer_std": std_or_nan(avg_total_buffer_vals),
            "train_lastk_makespan_mean": mean_or_nan(train_lastk_makespan_vals),
            "train_lastk_makespan_std": std_or_nan(train_lastk_makespan_vals),
            "train_lastk_deadlock_rate_mean": mean_or_nan(train_lastk_deadlock_rate_vals),
            "train_lastk_deadlock_rate_std": std_or_nan(train_lastk_deadlock_rate_vals),
            "train_lastk_total_buffer_mean": mean_or_nan(train_lastk_total_buffer_vals),
            "train_lastk_total_buffer_std": std_or_nan(train_lastk_total_buffer_vals),
            "train_wall_clock_sec_mean": mean_or_nan(train_wall_clock_vals),
            "eval_wall_clock_sec_mean": mean_or_nan(eval_wall_clock_vals),
        })

    # 先按 experiment，再按 reward_scheme 固定顺序
    order = {"dense_epi": 0, "terminal_only": 1, "dense_only": 2}
    out_rows.sort(key=lambda r: (
        str(r["experiment"]),
        order.get(str(r["reward_scheme"]), 99),
        str(r["method_name"]),
    ))
    return out_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# main
# ============================================================

def main() -> None:
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] RESULTS_DIR   = {RESULTS_DIR}")

    by_seed_rows = collect_runs(RESULTS_DIR, tuple(DEFAULT_METHOD_ALLOWLIST))
    if not by_seed_rows:
        print("[WARN] No revision1 reward-structure runs found.")
        print("[WARN] Expected method dirs such as:")
        for m in sorted(DEFAULT_METHOD_ALLOWLIST):
            print(f"       results/<experiment>/{m}/seed*_*/")
        return

    summary_rows = aggregate_rows(by_seed_rows)

    by_seed_fieldnames = [
        "experiment",
        "method_name",
        "reward_scheme",
        "lower_epi_reward_weight",
        "seed",
        "run_id",
        "num_eval_episodes",
        "avg_makespan",
        "deadlock_rate",
        "num_deadlocks",
        "avg_total_buffer",
        "std_total_buffer",
        "train_rows",
        "train_lastk_makespan",
        "train_lastk_deadlock_rate",
        "train_lastk_total_buffer",
        "train_wall_clock_sec",
        "eval_wall_clock_sec",
        "run_dir",
        "summary_path",
        "detail_path",
        "train_log_path",
    ]

    summary_fieldnames = [
        "experiment",
        "reward_scheme",
        "method_name",
        "lower_epi_reward_weight",
        "num_runs",
        "num_seeds",
        "avg_makespan_mean",
        "avg_makespan_std",
        "deadlock_rate_mean",
        "deadlock_rate_std",
        "num_deadlocks_mean",
        "num_deadlocks_std",
        "avg_total_buffer_mean",
        "avg_total_buffer_std",
        "train_lastk_makespan_mean",
        "train_lastk_makespan_std",
        "train_lastk_deadlock_rate_mean",
        "train_lastk_deadlock_rate_std",
        "train_lastk_total_buffer_mean",
        "train_lastk_total_buffer_std",
        "train_wall_clock_sec_mean",
        "eval_wall_clock_sec_mean",
    ]

    by_seed_csv = OUT_DIR / "supp_reward_structure_by_seed.csv"
    summary_csv = OUT_DIR / "supp_reward_structure_summary.csv"

    write_csv(by_seed_csv, by_seed_rows, by_seed_fieldnames)
    write_csv(summary_csv, summary_rows, summary_fieldnames)

    print(f"[SAVE] by-seed csv -> {by_seed_csv}")
    print(f"[SAVE] summary csv -> {summary_csv}")

    print("\n[INFO] Summary preview:")
    for row in summary_rows:
        print(
            f"  exp={row['experiment']}, scheme={row['reward_scheme']}, "
            f"runs={row['num_runs']}, seeds={row['num_seeds']}, "
            f"avg_makespan_mean={row['avg_makespan_mean']:.4f}, "
            f"deadlock_rate_mean={row['deadlock_rate_mean']:.4f}, "
            f"avg_total_buffer_mean={row['avg_total_buffer_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
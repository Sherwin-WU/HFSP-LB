# revision1/run_runtime_flat_drl.py
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ============================================================
# 路径
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
REVISION1_DIR = ROOT_DIR / "revision1"

for p in [ROOT_DIR, REVISION1_DIR]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)

# ============================================================
# 本地导入
# ============================================================

from revision1.runtime_config import (
    EXPERIMENTS,
    SEED,
    DEVICE_STR,
    TRAIN_EPISODES,
    INFER_REPEATS,
    NUM_EVAL_EPISODES,
    RESULT_ROOT_NAME,
    USE_CUDA_SYNC,
)

from revision1.flat_drl_baseline import (
    build_flat_drl_config,
    train_and_eval_one_seed,
    evaluate_flat_drl_on_test,
)

# ============================================================
# 路径与输出
# ============================================================

RESULT_ROOT = ROOT_DIR / RESULT_ROOT_NAME
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_RAW_CSV = RESULT_ROOT / "train_runtime_raw.csv"
INFER_RAW_CSV = RESULT_ROOT / "infer_runtime_raw.csv"

TRAIN_RAW_HEADER = [
    "experiment",
    "method",
    "family",
    "seed",
    "train_episodes",
    "train_wall_clock_sec",
    "train_sec_per_episode",
    "device",
    "out_dir",
    "note",
]

INFER_RAW_HEADER = [
    "experiment",
    "method",
    "family",
    "seed",
    "repeat_id",
    "num_eval_episodes",
    "infer_total_wall_clock_sec",
    "infer_avg_ms_per_episode",
    "uba_forward_calls",
    "uba_forward_total_ms",
    "uba_forward_avg_ms",
    "lda_forward_calls",
    "lda_forward_total_ms",
    "lda_forward_avg_ms",
    "device",
    "source_model_dir",
    "note",
]


# ============================================================
# 工具
# ============================================================

def ensure_csv_file(csv_path: Path, header: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())


def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def rewrite_csv_rows(csv_path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _same_key(row: Dict[str, Any], experiment: str, method: str, seed: int) -> bool:
    return (
        str(row.get("experiment", "")).strip() == str(experiment)
        and str(row.get("method", "")).strip() == str(method)
        and str(row.get("seed", "")).strip() == str(seed)
    )


def get_train_rows_for(experiment: str, method: str, seed: int) -> List[Dict[str, Any]]:
    rows = read_csv_rows(TRAIN_RAW_CSV)
    return [r for r in rows if _same_key(r, experiment, method, seed)]


def get_infer_rows_for(experiment: str, method: str, seed: int) -> List[Dict[str, Any]]:
    rows = read_csv_rows(INFER_RAW_CSV)
    return [r for r in rows if _same_key(r, experiment, method, seed)]


def get_finished_repeat_ids(experiment: str, method: str, seed: int) -> set[int]:
    rows = get_infer_rows_for(experiment, method, seed)
    out: set[int] = set()
    for r in rows:
        try:
            out.add(int(float(str(r.get("repeat_id", "")).strip())))
        except Exception:
            pass
    return out


def purge_method_records(
    experiment: str,
    method: str,
    seed: int,
    remove_train: bool,
    remove_infer: bool,
) -> None:
    if remove_train:
        old_rows = read_csv_rows(TRAIN_RAW_CSV)
        new_rows = [r for r in old_rows if not _same_key(r, experiment, method, seed)]
        rewrite_csv_rows(TRAIN_RAW_CSV, TRAIN_RAW_HEADER, new_rows)

    if remove_infer:
        old_rows = read_csv_rows(INFER_RAW_CSV)
        new_rows = [r for r in old_rows if not _same_key(r, experiment, method, seed)]
        rewrite_csv_rows(INFER_RAW_CSV, INFER_RAW_HEADER, new_rows)


def prepare_method_block(
    experiment: str,
    method: str,
    seed: int,
    require_train: bool,
) -> bool:
    train_rows = get_train_rows_for(experiment, method, seed)
    infer_rows = get_infer_rows_for(experiment, method, seed)
    finished_repeats = get_finished_repeat_ids(experiment, method, seed)

    train_done = (not require_train) or (len(train_rows) >= 1)
    infer_done = all(rep in finished_repeats for rep in range(1, INFER_REPEATS + 1))

    if train_done and infer_done:
        print(f"[SKIP] already done: exp={experiment}, method={method}, seed={seed}")
        return False

    if train_rows or infer_rows:
        print(
            f"[RESUME] partial block detected -> reset and rerun this block only: "
            f"exp={experiment}, method={method}, seed={seed}"
        )
        purge_method_records(
            experiment=experiment,
            method=method,
            seed=seed,
            remove_train=require_train,
            remove_infer=True,
        )

    return True


def cuda_sync_if_needed(device_str: str) -> None:
    if USE_CUDA_SYNC and "cuda" in device_str.lower() and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    cuda_sync_if_needed(DEVICE_STR)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    cuda_sync_if_needed(DEVICE_STR)
    t1 = time.perf_counter()
    return out, (t1 - t0)


# ============================================================
# Flat-DRL runtime
# ============================================================

METHOD_NAME = "Flat-DRL"


def run_flat_drl_train_and_infer(experiment_name: str) -> None:
    if not prepare_method_block(
        experiment=experiment_name,
        method=METHOD_NAME,
        seed=SEED,
        require_train=True,
    ):
        return

    cfg = build_flat_drl_config(
        experiment_name=experiment_name,
        method_name="flat_drl_single_agent",
        seed=SEED,
        num_outer_episodes=TRAIN_EPISODES,
        device_str=DEVICE_STR,
        dispatch_reward_scheme="dense_epi",
        dispatch_epi_reward_weight=1.0,
    )

    base_out_dir = ROOT_DIR / "results" / experiment_name / "flat_drl_single_agent"

    payload, train_dt = timed_call(
        train_and_eval_one_seed,
        cfg,
        base_out_dir,
        True,   # skip_final_test_eval
        True,   # return_trained_objects
    )

    if payload is None:
        print(f"[ERROR] Failed to train Flat-DRL for experiment {experiment_name}")
        return

    out_dir = str(payload["out_dir"])

    train_row = dict(
        experiment=experiment_name,
        method=METHOD_NAME,
        family="revision1_flat_drl",
        seed=SEED,
        train_episodes=TRAIN_EPISODES,
        train_wall_clock_sec=train_dt,
        train_sec_per_episode=train_dt / float(TRAIN_EPISODES),
        device=DEVICE_STR,
        out_dir=out_dir,
        note="train only, final test eval skipped",
    )
    append_csv_row(TRAIN_RAW_CSV, TRAIN_RAW_HEADER, train_row)
    print(
        f"[TRAIN][Flat-DRL] exp={experiment_name}, method={METHOD_NAME}, sec={train_dt:.6f}"
    )

    # warm-up
    warm_dir = Path(out_dir) / "_runtime_warmup"
    warm_dir.mkdir(parents=True, exist_ok=True)

    evaluate_flat_drl_on_test(
        cfg=payload["cfg"],
        agent=payload["agent"],
        test_instances=payload["test_instances"],
        device=payload["device"],
        out_dir=warm_dir,
    )

    for repeat_id in range(1, INFER_REPEATS + 1):
        rep_dir = Path(out_dir) / f"_runtime_repeat_{repeat_id:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        _, infer_dt = timed_call(
            evaluate_flat_drl_on_test,
            cfg=payload["cfg"],
            agent=payload["agent"],
            test_instances=payload["test_instances"],
            device=payload["device"],
            out_dir=rep_dir,
        )

        infer_row = dict(
            experiment=experiment_name,
            method=METHOD_NAME,
            family="revision1_flat_drl",
            seed=SEED,
            repeat_id=repeat_id,
            num_eval_episodes=NUM_EVAL_EPISODES,
            infer_total_wall_clock_sec=infer_dt,
            infer_avg_ms_per_episode=(infer_dt / float(NUM_EVAL_EPISODES)) * 1000.0,
            uba_forward_calls=np.nan,
            uba_forward_total_ms=np.nan,
            uba_forward_avg_ms=np.nan,
            lda_forward_calls=np.nan,
            lda_forward_total_ms=np.nan,
            lda_forward_avg_ms=np.nan,
            device=DEVICE_STR,
            source_model_dir=out_dir,
            note="end-to-end eval runtime only",
        )
        append_csv_row(INFER_RAW_CSV, INFER_RAW_HEADER, infer_row)
        print(
            f"[INFER][Flat-DRL] exp={experiment_name}, method={METHOD_NAME}, "
            f"repeat={repeat_id}, sec={infer_dt:.6f}"
        )


def main() -> None:
    ensure_csv_file(TRAIN_RAW_CSV, TRAIN_RAW_HEADER)
    ensure_csv_file(INFER_RAW_CSV, INFER_RAW_HEADER)

    print("=" * 80)
    print("[RUNTIME] Flat-DRL runtime")
    print(f"[RUNTIME] experiments={EXPERIMENTS}")
    print(f"[RUNTIME] seed={SEED}, device={DEVICE_STR}")
    print(f"[RUNTIME] train_episodes={TRAIN_EPISODES}, infer_repeats={INFER_REPEATS}")
    print(f"[RUNTIME] train csv -> {TRAIN_RAW_CSV}")
    print(f"[RUNTIME] infer csv -> {INFER_RAW_CSV}")
    print("=" * 80)

    for experiment_name in EXPERIMENTS:
        print("\n" + "#" * 80)
        print(f"[EXP][Flat-DRL] {experiment_name}")
        print("#" * 80)
        run_flat_drl_train_and_infer(experiment_name)

    print("\n[DONE] Flat-DRL runtime finished.")
    print(f"  - {TRAIN_RAW_CSV}")
    print(f"  - {INFER_RAW_CSV}")


if __name__ == "__main__":
    main()
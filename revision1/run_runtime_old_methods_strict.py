# revision1/run_runtime_old_methods_strict.py
from __future__ import annotations

import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ============================================================
# 路径
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT_DIR / "examples"
SRC_DIR = ROOT_DIR / "src"

for p in [ROOT_DIR, EXAMPLES_DIR, SRC_DIR]:
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
    GROUP2_RULES,
    GROUP3_FIXED_BUFFERS,
    RESULT_ROOT_NAME,
    USE_CUDA_SYNC,
)
from revision1.runtime_rulebank import load_rule_baseline_buffers

# Group1
from examples.group1_rule_baseline import _evaluate_one_buffer_for_rule
from examples.group2_two_level_rules import (
    build_group2_config,
    load_instances_for_experiment as load_instances_g2,
    train_and_eval_one_seed_for_rule,
)
from examples.group4_two_level import (
    build_group4_config,
    load_instances_for_experiment as load_instances_g4,
    train_and_eval_one_seed,
)
from examples.train_dispatch_d3qn_fixedbuf import (
    DispatchAgentConfig,
    DispatchTrainConfig,
    load_instances_for_experiment as load_instances_g3,
    run_greedy_eval_episodes,
    train_dispatch_d3qn_fixedbuf,
)
from envs.reward import ShopRewardConfig

# ============================================================
# 工具
# ============================================================

RESULT_ROOT = ROOT_DIR / RESULT_ROOT_NAME
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cuda_sync_if_needed(device_str: str) -> None:
    if USE_CUDA_SYNC and "cuda" in device_str.lower() and torch.cuda.is_available():
        torch.cuda.synchronize()


def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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
    """
    返回 True 表示需要执行该方法块；
    返回 False 表示该方法块已经完整完成，可直接跳过。

    规则：
    - 若该块 train + infer 都完整，则跳过；
    - 若该块已有部分记录但不完整，则清理该块 raw 记录，重跑该块；
    - 若该块完全没有记录，则正常执行。
    """
    train_rows = get_train_rows_for(experiment, method, seed)
    infer_rows = get_infer_rows_for(experiment, method, seed)
    finished_repeats = get_finished_repeat_ids(experiment, method, seed)

    train_done = (not require_train) or (len(train_rows) >= 1)
    infer_done = all(rep in finished_repeats for rep in range(1, INFER_REPEATS + 1))

    if train_done and infer_done:
        print(f"[SKIP] already done: exp={experiment}, method={method}, seed={seed}")
        return False

    # 只要存在不完整记录，就清空这个块，避免重复行和脏数据
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

TRAIN_RAW_CSV = RESULT_ROOT / "train_runtime_raw.csv"
INFER_RAW_CSV = RESULT_ROOT / "infer_runtime_raw.csv"


def timed_call(fn, *args, **kwargs):
    cuda_sync_if_needed(DEVICE_STR)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    cuda_sync_if_needed(DEVICE_STR)
    t1 = time.perf_counter()
    return out, (t1 - t0)


# ============================================================
# Group1: FB-* 推理
# ============================================================

def run_fb_inference_runtime(
    experiment_name: str,
    fb_buffers_bank: Dict[str, Dict[str, List[int]]],
) -> None:
    _, _, test_instances = load_instances_g2(experiment_name)

    method_to_rule = {
        "FB-FIFO": "fifo",
        "FB-LPT": "lpt",
        "FB-SPT": "spt",
        "FB-SRPT": "srpt",
    }

    if experiment_name not in fb_buffers_bank:
        raise RuntimeError(f"No FB buffer bank found for experiment={experiment_name}")

    exp_bank = fb_buffers_bank[experiment_name]

    for method_name, rule_name in method_to_rule.items():
        if not prepare_method_block(
            experiment=experiment_name,
            method=method_name,
            seed=SEED,
            require_train=False,
        ):
            continue

        buffers = exp_bank[method_name]

        # warm-up
        _ = _evaluate_one_buffer_for_rule(
            rule=rule_name,
            buffers=buffers,
            instances=test_instances,
            max_steps=2000,
            seed=SEED,
        )

        for repeat_id in range(1, INFER_REPEATS + 1):
            _, dt = timed_call(
                _evaluate_one_buffer_for_rule,
                rule_name,
                buffers,
                test_instances,
                2000,
                SEED + repeat_id,
            )

            row = dict(
                experiment=experiment_name,
                method=method_name,
                family="group1",
                seed=SEED,
                repeat_id=repeat_id,
                num_eval_episodes=len(test_instances),
                infer_total_wall_clock_sec=dt,
                infer_avg_ms_per_episode=(dt / max(1, len(test_instances))) * 1000.0,
                uba_forward_calls=np.nan,
                uba_forward_total_ms=np.nan,
                uba_forward_avg_ms=np.nan,
                lda_forward_calls=np.nan,
                lda_forward_total_ms=np.nan,
                lda_forward_avg_ms=np.nan,
                device=DEVICE_STR,
                source_model_dir="N/A",
                note=str(buffers),
            )
            append_csv_row(INFER_RAW_CSV, INFER_RAW_HEADER, row)
            print(
                f"[INFER][FB] exp={experiment_name}, method={method_name}, "
                f"repeat={repeat_id}, sec={dt:.6f}"
            )


# ============================================================
# Group2: UB-* 训练 + 推理
# ============================================================

def run_group2_train_and_infer(experiment_name: str) -> None:
    train_instances, _, test_instances = load_instances_g2(experiment_name)

    for rule_name in GROUP2_RULES:
        method_name = f"UB-{rule_name.upper()}"

        if not prepare_method_block(
            experiment=experiment_name,
            method=method_name,
            seed=SEED,
            require_train=True,
        ):
            continue

        cfg = build_group2_config(
            experiment_name=experiment_name,
            rule_name=rule_name,
            seed=SEED,
            num_outer_episodes=TRAIN_EPISODES,
            device_str=DEVICE_STR,
        )

        base_out_dir = ROOT_DIR / "results" / experiment_name / f"group2_two_level_{rule_name}"

        payload, train_dt = timed_call(
            train_and_eval_one_seed_for_rule,
            cfg,
            train_instances,
            test_instances,
            base_out_dir,
            True,   # skip_final_test_eval
            True,   # return_trained_objects
        )

        out_dir = str(payload["out_dir"])

        train_row = dict(
            experiment=experiment_name,
            method=method_name,
            family="group2",
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
            f"[TRAIN][G2] exp={experiment_name}, method={method_name}, "
            f"sec={train_dt:.6f}"
        )

        # 说明：第一版先不做更细粒度 UBA 前向计时，仅统计端到端 eval
        from examples.group2_two_level_rules import evaluate_group2_on_test

        # warm-up
        warm_dir = Path(out_dir) / "_runtime_warmup"
        warm_dir.mkdir(parents=True, exist_ok=True)
        evaluate_group2_on_test(
            cfg=payload["cfg"],
            upper_agent=payload["upper_agent"],
            test_instances=test_instances,
            out_dir=warm_dir,
            device=payload["device"],
            num_eval_episodes=NUM_EVAL_EPISODES,
        )

        for repeat_id in range(1, INFER_REPEATS + 1):
            rep_dir = Path(out_dir) / f"_runtime_repeat_{repeat_id:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            _, infer_dt = timed_call(
                evaluate_group2_on_test,
                cfg=payload["cfg"],
                upper_agent=payload["upper_agent"],
                test_instances=test_instances,
                out_dir=rep_dir,
                device=payload["device"],
                num_eval_episodes=NUM_EVAL_EPISODES,
            )

            infer_row = dict(
                experiment=experiment_name,
                method=method_name,
                family="group2",
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
                f"[INFER][G2] exp={experiment_name}, method={method_name}, "
                f"repeat={repeat_id}, sec={infer_dt:.6f}"
            )


# ============================================================
# Group3: LD-Agent 训练 + 推理
# ============================================================

def run_group3_train_and_infer(experiment_name: str) -> None:
    train_instances, val_instances, test_instances = load_instances_g3(experiment_name)

    if not prepare_method_block(
        experiment=experiment_name,
        method="LD-Agent",
        seed=SEED,
        require_train=True,
    ):
        return

    fixed_buffers = GROUP3_FIXED_BUFFERS[experiment_name]

    reward_cfg = ShopRewardConfig(
        mode="progress",
        time_weight=1.0,
        per_operation_reward=0.05,
        per_job_reward=0.1,
        blocking_penalty=0.2,
        terminal_bonus=0.5,
        invalid_action_weight=0.2,
        makespan_weight=0.0,
    )

    agent_cfg = DispatchAgentConfig(obs_dim=0, action_dim=0)

    cfg = DispatchTrainConfig(
        experiment_name=experiment_name,
        device=DEVICE_STR,
        random_seed=SEED,
        fixed_buffers=fixed_buffers,
        reward_cfg=reward_cfg,
        num_episodes=TRAIN_EPISODES,
        max_steps_per_episode=2000,
        eval_interval=2000,
        log_interval=100,
        agent_cfg=agent_cfg,
    )

    out_dir = ROOT_DIR / "results" / experiment_name / "dispatch_d3qn_fixedbuf" / f"seed{SEED}_{now_ts()}"

    payload, train_dt = timed_call(
        train_dispatch_d3qn_fixedbuf,
        cfg,
        train_instances,
        val_instances,
        test_instances,
        str(out_dir),
        True,   # skip_final_test_eval
        True,   # return_trained_objects
    )

    train_row = dict(
        experiment=experiment_name,
        method="LD-Agent",
        family="group3",
        seed=SEED,
        train_episodes=TRAIN_EPISODES,
        train_wall_clock_sec=train_dt,
        train_sec_per_episode=train_dt / float(TRAIN_EPISODES),
        device=DEVICE_STR,
        out_dir=str(payload["out_dir"]),
        note=f"fixed_buffers={fixed_buffers}",
    )
    append_csv_row(TRAIN_RAW_CSV, TRAIN_RAW_HEADER, train_row)
    print(f"[TRAIN][G3] exp={experiment_name}, method=LD-Agent, sec={train_dt:.6f}")

    # warm-up
    _ = run_greedy_eval_episodes(
        agent=payload["agent"],
        instances=test_instances,
        buffers=fixed_buffers,
        reward_cfg=reward_cfg,
        device=payload["device"],
        max_steps=cfg.max_steps_per_episode,
        num_eval_episodes=NUM_EVAL_EPISODES,
    )

    for repeat_id in range(1, INFER_REPEATS + 1):
        _, infer_dt = timed_call(
            run_greedy_eval_episodes,
            agent=payload["agent"],
            instances=test_instances,
            buffers=fixed_buffers,
            reward_cfg=reward_cfg,
            device=payload["device"],
            max_steps=cfg.max_steps_per_episode,
            num_eval_episodes=NUM_EVAL_EPISODES,
        )

        infer_row = dict(
            experiment=experiment_name,
            method="LD-Agent",
            family="group3",
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
            source_model_dir=str(payload["out_dir"]),
            note=f"fixed_buffers={fixed_buffers}",
        )
        append_csv_row(INFER_RAW_CSV, INFER_RAW_HEADER, infer_row)
        print(
            f"[INFER][G3] exp={experiment_name}, method=LD-Agent, "
            f"repeat={repeat_id}, sec={infer_dt:.6f}"
        )


# ============================================================
# Group4: UB+LD 训练 + 推理
# ============================================================

def run_group4_train_and_infer(experiment_name: str) -> None:
    train_instances, _, test_instances = load_instances_g4(experiment_name)

    if not prepare_method_block(
        experiment=experiment_name,
        method="UB+LD",
        seed=SEED,
        require_train=True,
    ):
        return

    cfg = build_group4_config(
        experiment_name=experiment_name,
        method_name="group4_two_level",
        seed=SEED,
        num_outer_episodes=TRAIN_EPISODES,
        device_str=DEVICE_STR,
    )

    base_out_dir = ROOT_DIR / "results" / experiment_name / "group4_two_level"

    payload, train_dt = timed_call(
        train_and_eval_one_seed,
        cfg,
        train_instances,
        test_instances,
        base_out_dir,
        True,   # skip_final_test_eval
        True,   # return_trained_objects
    )

    out_dir = str(payload["out_dir"])

    train_row = dict(
        experiment=experiment_name,
        method="UB+LD",
        family="group4",
        seed=SEED,
        train_episodes=TRAIN_EPISODES,
        train_wall_clock_sec=train_dt,
        train_sec_per_episode=train_dt / float(TRAIN_EPISODES),
        device=DEVICE_STR,
        out_dir=out_dir,
        note="train only, final test eval skipped",
    )
    append_csv_row(TRAIN_RAW_CSV, TRAIN_RAW_HEADER, train_row)
    print(f"[TRAIN][G4] exp={experiment_name}, method=UB+LD, sec={train_dt:.6f}")

    from examples.group4_two_level import evaluate_group4_on_test

    # warm-up
    warm_dir = Path(out_dir) / "_runtime_warmup"
    warm_dir.mkdir(parents=True, exist_ok=True)
    evaluate_group4_on_test(
        upper_agent=payload["upper_agent"],
        lower_agent=payload["lower_agent"],
        cfg=payload["cfg"],
        test_instances=test_instances,
        out_dir=warm_dir,
        device=payload["device"],
        num_eval_episodes=NUM_EVAL_EPISODES,
    )

    for repeat_id in range(1, INFER_REPEATS + 1):
        rep_dir = Path(out_dir) / f"_runtime_repeat_{repeat_id:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        _, infer_dt = timed_call(
            evaluate_group4_on_test,
            upper_agent=payload["upper_agent"],
            lower_agent=payload["lower_agent"],
            cfg=payload["cfg"],
            test_instances=test_instances,
            out_dir=rep_dir,
            device=payload["device"],
            num_eval_episodes=NUM_EVAL_EPISODES,
        )

        infer_row = dict(
            experiment=experiment_name,
            method="UB+LD",
            family="group4",
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
            f"[INFER][G4] exp={experiment_name}, method=UB+LD, "
            f"repeat={repeat_id}, sec={infer_dt:.6f}"
        )


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    print("=" * 80)
    print("[RUNTIME] strict runtime for old methods")
    print(f"[RUNTIME] experiments={EXPERIMENTS}")
    print(f"[RUNTIME] seed={SEED}, device={DEVICE_STR}")
    print(f"[RUNTIME] train_episodes={TRAIN_EPISODES}, infer_repeats={INFER_REPEATS}")
    print("=" * 80)

    fb_buffers_bank = load_rule_baseline_buffers()

    for experiment_name in EXPERIMENTS:
        print("\n" + "#" * 80)
        print(f"[EXP] {experiment_name}")
        print("#" * 80)

        # Group1: inference only
        run_fb_inference_runtime(experiment_name, fb_buffers_bank)

        # Group2: train + inference
        run_group2_train_and_infer(experiment_name)

        # Group3: train + inference
        run_group3_train_and_infer(experiment_name)

        # Group4: train + inference
        run_group4_train_and_infer(experiment_name)

    print("\n[DONE] runtime raw csv generated:")
    print(f"  - {TRAIN_RAW_CSV}")
    print(f"  - {INFER_RAW_CSV}")


if __name__ == "__main__":
    main()
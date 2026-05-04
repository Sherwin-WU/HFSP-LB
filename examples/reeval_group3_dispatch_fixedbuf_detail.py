# examples/reeval_group3_dispatch_fixedbuf_detail.py
"""
Group3 重新推理脚本：只用已训练好的模型，在 test 集上重新做 100 次 greedy 推理，
并生成：
    - eval_test_detail.csv
    - eval_test_summary_detail.csv

不重新训练，只重跑测试。
"""

from __future__ import annotations

import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch

# -------------------------------------------------------------
# 路径设置
# -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

# -------------------------------------------------------------
# 导入项目内模块（和训练脚本保持一致）
# -------------------------------------------------------------
from envs.reward import ShopRewardConfig  # type: ignore

from train_dispatch_d3qn_fixedbuf import (  # type: ignore
    DispatchAgentConfig,
    create_dispatch_agent,
    load_instances_for_experiment,
    run_greedy_eval_episodes,
)

RESULTS_DIR = Path(ROOT_DIR) / "results"


# =============================================================
# 工具函数
# =============================================================

def _scan_experiments() -> List[str]:
    """扫描 results/ 下有哪些 experiment（一级子目录名）。"""
    if not RESULTS_DIR.exists():
        return []
    return sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()])


def _parse_seed_and_runid_from_dirname(name: str) -> Tuple[int | None, str]:
    """
    从 seed 目录名 seed<seed>_<run_id> 中解析出 (seed, run_id)。

    例如：
      'seed3_20251202_184927' -> (3, '20251202_184927')
    """
    if not name.startswith("seed"):
        return None, ""
    rest = name[4:]
    if "_" in rest:
        seed_str, run_id = rest.split("_", 1)
    else:
        seed_str, run_id = rest, ""
    try:
        seed = int(seed_str)
    except ValueError:
        seed = None
    return seed, run_id


# =============================================================
# 核心：对单个 seed 重新推理
# =============================================================

def reeval_one_seed(exp_name: str, seed_dir: Path) -> None:
    print(f"\n[REEVAL] experiment={exp_name}, seed_dir={seed_dir.name}")

    cfg_path = seed_dir / "cfg.json"
    if not cfg_path.exists():
        print(f"[WARN] cfg.json not found, skip: {cfg_path}")
        return

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    # --------- 从 cfg.json 还原必要配置 ---------
    # experiment_name 也可以从 exp_name 推出，这里用 cfg_json 里的为准（做个 sanity check）
    exp_name_cfg = cfg_json.get("experiment_name", exp_name)
    if exp_name_cfg != exp_name:
        print(f"[WARN] experiment mismatch: cfg={exp_name_cfg}, dir={exp_name}")

    fixed_buffers = cfg_json.get("fixed_buffers", [])
    if not isinstance(fixed_buffers, list):
        fixed_buffers = list(fixed_buffers)

    reward_cfg_dict = cfg_json.get("reward_cfg", {})
    reward_cfg = ShopRewardConfig(**reward_cfg_dict)

    agent_cfg_dict = cfg_json.get("agent_cfg", {})
    agent_cfg = DispatchAgentConfig(**agent_cfg_dict)

    max_steps = int(cfg_json.get("max_steps_per_episode", 2000))

    device_str = cfg_json.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --------- 构造 agent 并载入 checkpoint ---------
    agent = create_dispatch_agent(agent_cfg, device)

    ckpt_path = seed_dir / "dispatch_q_last.pth"
    if not ckpt_path.exists():
        print(f"[WARN] checkpoint not found, skip: {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    agent["q_net"].load_state_dict(state_dict)
    agent["target_q_net"].load_state_dict(state_dict)

    # --------- 读取 test 实例 ---------
    train_insts, val_insts, test_insts = load_instances_for_experiment(exp_name_cfg)
    if not test_insts:
        print(f"[WARN] No test instances for {exp_name_cfg}, skip.")
        return

    # --------- 调用训练脚本里的 greedy 评估函数 ---------
    print(f"[REEVAL] Running greedy eval on test set ({len(test_insts)} instances)...")
    metrics = run_greedy_eval_episodes(
        agent=agent,
        instances=test_insts,
        buffers=fixed_buffers,
        reward_cfg=reward_cfg,
        device=device,
        max_steps=max_steps,
        num_episodes=100,
    )

    avg_ms = float(metrics.get("avg_makespan", 0.0))
    dead_rate = float(metrics.get("deadlock_rate", 0.0))
    episodes = int(metrics.get("episodes", 0))
    details = metrics.get("details", [])

    # 统计 deadlock 次数（details 里每条都有 deadlock 标记）
    num_deadlocks = 0
    for row in details:
        try:
            d = int(row.get("deadlock", 0))
        except Exception:
            d = 1 if bool(row.get("deadlock", False)) else 0
        num_deadlocks += d

    # =========================================================
    # 写明细文件：eval_test_detail.csv
    # =========================================================
    detail_path = seed_dir / "eval_test_detail.csv"
    with detail_path.open("w", newline="") as f_det:
        writer = csv.writer(f_det)
        # 多加一列 buffers（Group3 每个 seed 固定相同的 buffers）
        writer.writerow(["episode", "instance_idx", "buffers", "makespan", "deadlock", "steps"])
        buf_str = " ".join(str(int(b)) for b in fixed_buffers)
        for row in details:
            writer.writerow(
                [
                    row.get("episode", ""),
                    row.get("instance_idx", ""),
                    buf_str,
                    row.get("makespan", ""),
                    row.get("deadlock", ""),
                    row.get("steps", ""),
                ]
            )

    # =========================================================
    # 写 summary 文件：eval_test_summary_detail.csv
    # =========================================================
    summary_path = seed_dir / "eval_test_summary_detail.csv"
    with summary_path.open("w", newline="") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            ["split", "num_eval_episodes", "avg_makespan", "deadlock_rate", "num_deadlocks"]
        )
        writer.writerow(["test", episodes, avg_ms, dead_rate, num_deadlocks])

    print(
        f"[DONE] detail -> {detail_path}, "
        f"summary -> {summary_path}, "
        f"avg_makespan={avg_ms:.3f}, deadlock_rate={dead_rate:.3f}"
    )


# =============================================================
# 主函数：遍历所有 Group3 结果
# =============================================================

def main():
    exps = _scan_experiments()
    if not exps:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Start Group3 re-evaluation...")
    for exp in exps:
        exp_dir = RESULTS_DIR / exp
        method_dir = exp_dir / "dispatch_d3qn_fixedbuf"
        if not method_dir.exists():
            continue

        print(f"\n[SCAN] Experiment = {exp}")
        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed, run_id = _parse_seed_and_runid_from_dirname(seed_dir.name)
            if seed is None:
                continue
            reeval_one_seed(exp, seed_dir)

    print("\n[FINISHED] Group3 re-evaluation for all available seeds.")


if __name__ == "__main__":
    main()

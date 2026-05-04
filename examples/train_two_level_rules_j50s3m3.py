# examples/train_two_level_rules_j50s3m3.py
"""
在 j50s3m3 上批量跑组 2 实验：缓存 Agent + 规则调度（下层不训练）。

对多条规则 (fifo/spt/lpt/srpt) × 多个随机种子，调用 train_two_level：
  - 上层：BufferDesignEnv + UpperAgent（D3QN）
  - 下层：lower_train_mode = "rule"，通过 simulate_instance_with_job_rule 调度

用法示例（在项目根目录）：

  # 默认跑 4 条规则、seed=0
  python examples/train_two_level_rules_j50s3m3.py

  # 指定规则 + 多个种子
  python examples/train_two_level_rules_j50s3m3.py --rules fifo,spt --seeds 0,1,2

  # 使用 CPU / 修改 outer episodes
  python examples/train_two_level_rules_j50s3m3.py --device cpu --num_outer_episodes 300
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List

import argparse
import itertools

# ---- 路径设置：把 src/ 加进 sys.path（train_two_level 里也会做一次，但这里不妨再写一遍） ----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# 从同目录下的 train_two_level.py 中导入训练主流程和配置构造函数
from train_two_level import (  # type: ignore
    train_two_level,
    load_j50s3m3_instances,
    build_standard_two_level_config_for_j50s3m3,
)


# ============================================================
# 1. CLI 解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-level buffer agent + rule-based dispatching on j50s3m3."
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="fifo,spt,lpt,srpt",
        help="Comma-separated list of job rules to run. "
             "Options include: fifo,spt,lpt,srpt. (default: fifo,spt,lpt,srpt)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated list of random seeds. (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device for training: "cuda" or "cpu". (default: cuda)',
    )
    parser.add_argument(
        "--num_outer_episodes",
        type=int,
        default=400,
        help="Number of outer episodes for upper-level training. (default: 400)",
    )
    parser.add_argument(
        "--buffer_cost_weight",
        type=float,
        default=1.0,
        help="Upper-level buffer_cost_weight in reward. (default: 1.0)",
    )
    parser.add_argument(
        "--deadlock_penalty",
        type=float,
        default=2000.0,
        help="Upper-level deadlock_penalty in reward. (default: 2000.0)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="j50s3m3",
        help="Experiment name (default: j50s3m3)",
    )
    return parser.parse_args()


def parse_comma_list_int(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_comma_list_str(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip() != ""]


# ============================================================
# 2. 主逻辑：循环 (rule, seed) 调用 train_two_level
# ============================================================

def main():
    args = parse_args()

    experiment_name = args.experiment_name
    rules: List[str] = parse_comma_list_str(args.rules)
    seeds: List[int] = parse_comma_list_int(args.seeds)

    if not rules:
        print("[ERROR] No rules specified. Abort.")
        return
    if not seeds:
        print("[ERROR] No seeds specified. Abort.")
        return

    print(f"[INFO] Experiment name: {experiment_name}")
    print(f"[INFO] Rules to run: {rules}")
    print(f"[INFO] Seeds to run: {seeds}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] num_outer_episodes: {args.num_outer_episodes}")
    print(f"[INFO] buffer_cost_weight: {args.buffer_cost_weight}")
    print(f"[INFO] deadlock_penalty: {args.deadlock_penalty}")

    # ---- 1. 加载 j50s3m3 的 train/val/test 实例 ----
    train_instances, val_instances, test_instances = load_j50s3m3_instances()

    # ---- 2. 组合 (rule, seed) 逐个跑 ----
    for rule_name, seed in itertools.product(rules, seeds):
        rule_name = rule_name.lower()

        # 构造标准 two-level 配置（静态上层）
        cfg = build_standard_two_level_config_for_j50s3m3(
            seed=seed,
            num_outer_episodes=args.num_outer_episodes,
            device=args.device,
            buffer_cost_weight=args.buffer_cost_weight,
            deadlock_penalty=args.deadlock_penalty,
        )

        # 下层改成：完全规则调度，不训练 DQN
        cfg.lower_train_mode = "rule"      # 关键：用 rule 模式
        cfg.lower_job_rule = rule_name     # 使用指定规则：fifo/spt/lpt/srpt

        # 为了区分不同设置，更新一下实验/算法名字
        cfg.experiment_name = experiment_name
        cfg.algo_name = f"ba_rule_{rule_name}_seed{seed}"

        # 输出目录：results/<exp>/ba_rule/<rule>/seed<seed>_<timestamp>/
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            ROOT_DIR,
            "results",
            experiment_name,
            "ba_rule",
            rule_name,
            f"seed{seed}_{run_id}",
        )
        os.makedirs(out_dir, exist_ok=True)

        print(
            f"\n[RUN] rule={rule_name}, seed={seed}, "
            f"out_dir={out_dir}"
        )

        # 调用两层训练主循环：
        # 上层：BufferDesignEnv + UpperAgent（DQN）
        # 下层：simulate_instance_with_job_rule（不训练）
        train_two_level(
            cfg=cfg,
            train_instances=train_instances,
            val_instances=val_instances,
            test_instances=test_instances,
            out_dir=out_dir,
        )

    print("\n[ALL DONE] All (rule, seed) runs finished.")


if __name__ == "__main__":
    main()

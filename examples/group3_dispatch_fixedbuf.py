# examples/group3_dispatch_fixedbuf.py
"""
Group3：固定缓存 + 下层 D3QN 派工（函数版）

对外接口（给总控脚本调用）：

    run_group3_for_experiment(
        experiment_name: str,
        fixed_buffers: List[int],
        seeds: List[int],
        device_str: str = "cuda",
        num_episodes: int = 400,
        max_steps_per_episode: int = 2_000,
        eval_interval: int = 2_000,
        log_interval: int = 100,
    )

行为：
  - 从 experiments/raw/<experiment_name>/{train,val,test} 加载实例；
  - 对 seeds 中每个 seed：
      * 固定 buffers = fixed_buffers
      * 在 train 上训练下层 D3QN 派工（进度奖励）
      * 定期在 val 上评估（逻辑在原 train_dispatch_d3qn_fixedbuf 里）
      * 训练结束后在 test 上评估
  - 每个 seed 的输出目录：
      results/<experiment_name>/dispatch_d3qn_fixedbuf/seed{seed}_YYYYMMDD_HHMMSS/

注意：本文件只是一个“壳”，真正的训练逻辑完全复用
      examples/train_dispatch_d3qn_fixedbuf.py 中已经实现的函数。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List

# ---------- 路径设置：把 src/ 和 examples/ 加入 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

# ---------- 复用 Group3 现有训练代码 ----------
from train_dispatch_d3qn_fixedbuf import (  # type: ignore
    DispatchTrainConfig,
    DispatchAgentConfig,
    load_instances_for_experiment,
    train_dispatch_d3qn_fixedbuf,
)
from envs.reward import ShopRewardConfig


def run_group3_for_experiment(
    experiment_name: str,
    fixed_buffers: List[int],
    seeds: List[int],
    device_str: str = "cuda",
    num_episodes: int = 400,
    max_steps_per_episode: int = 2_000,
    eval_interval: int = 2_000,
    log_interval: int = 100,
) -> None:
    """
    在给定 experiment_name 上，针对一组固定缓存 fixed_buffers，
    对 seeds 中的每个随机种子训练一个 “固定缓存 + D3QN 派工” agent。

    结果目录：
        results/<experiment_name>/dispatch_d3qn_fixedbuf/seed{seed}_YYYYMMDD_HHMMSS/
    """

    # --- 固定缓存检查 ---
    fixed_buffers = list(fixed_buffers)
    print(f"[GROUP3] experiment_name={experiment_name}")
    print(f"[GROUP3] fixed_buffers={fixed_buffers}")
    print(f"[GROUP3] seeds={seeds}")
    print(f"[GROUP3] device={device_str}")
    print(f"[GROUP3] num_episodes={num_episodes}, max_steps_per_episode={max_steps_per_episode}")
    print(f"[GROUP3] eval_interval={eval_interval}, log_interval={log_interval}")

    # --- 加载实例 ---
    train_instances, val_instances, test_instances = load_instances_for_experiment(
        experiment_name
    )
    if not train_instances:
        print(
            f"[GROUP3][ERROR] No train instances loaded for {experiment_name}. "
            f"Please check experiments/raw/{experiment_name}/train/"
        )
        return

    # --- 下层奖励配置（与原脚本保持一致的 progress reward） ---
    base_reward_cfg = ShopRewardConfig(
        mode="progress",
        time_weight=1.0,
        per_operation_reward=0.05,
        per_job_reward=0.1,
        blocking_penalty=0.2,
        terminal_bonus=0.5,
        invalid_action_weight=0.2,
        makespan_weight=0.0,
    )

    # agent_cfg 先给占位，obs_dim/action_dim 会在 train_dispatch_d3qn_fixedbuf 里根据 dummy env 设置
    base_agent_cfg = DispatchAgentConfig(obs_dim=0, action_dim=0)

    # --- 每个 seed 单独跑一遍 ---
    for seed in seeds:
        cfg = DispatchTrainConfig(
            experiment_name=experiment_name,
            device=device_str,
            random_seed=seed,
            fixed_buffers=fixed_buffers,
            reward_cfg=base_reward_cfg,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            eval_interval=eval_interval,
            log_interval=log_interval,
            agent_cfg=base_agent_cfg,
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            ROOT_DIR,
            "results",
            experiment_name,
            "dispatch_d3qn_fixedbuf",
            f"seed{seed}_{run_id}",
        )

        print(
            f"\n[GROUP3][RUN] experiment={experiment_name}, seed={seed}, "
            f"out_dir={out_dir}"
        )

        train_dispatch_d3qn_fixedbuf(
            cfg=cfg,
            train_instances=train_instances,
            val_instances=val_instances,
            test_instances=test_instances,
            out_dir=out_dir,
        )

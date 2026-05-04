# examples/eval_dispatch_d3qn_fixedbuf_detail_all.py
"""
一键版：批量对 dispatch_d3qn_fixedbuf 的所有种子做 greedy 推理 + detail 记录

使用方式：
    直接运行本脚本即可（双击 / IDE 运行 / python examples/... 都行），
    不需要也不支持任何命令行参数。

默认假设目录结构为：

    ROOT_DIR/
      src/
      examples/
      results/
        j50s3m3/
          dispatch_d3qn_fixedbuf/
            seed0_.../
              cfg.json
              dispatch_q_last.pth
            seed1_.../
              cfg.json
              dispatch_q_last.pth
            ...

脚本行为：
    - 固定 experiment_name = "j50s3m3"
    - parent_dir = ROOT_DIR/results/j50s3m3/dispatch_d3qn_fixedbuf
    - 匹配子目录 pattern = "seed*_*"
    - 每个 run 用 dispatch_q_last.pth
    - 在 test 集上 greedy 跑 NUM_EPISODES = 100 次
    - 为每个 run 写：
        - eval_test_summary_detail.csv
        - eval_test_detail.csv
"""

from __future__ import annotations

import os
import sys
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ----------------- 可调常量（需要的时候手动改这里） -----------------
EXPERIMENT_NAME = "j50s3m3"          # 如需换算例，改这个
PATTERN = "seed*_*"                  # 运行目录命名模式
CKPT_TAG = "last"                    # 使用 dispatch_q_last.pth
NUM_EPISODES = 100                   # 每个 run 推理多少次
DEVICE = "cuda"                      # "cuda" / "cpu"
# -------------------------------------------------------------------

# ---------- 路径设置：把 src/ 加进 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from instances.io import load_instances_from_dir
from instances.types import InstanceData

from envs.ffs_core_env import FlowShopCoreEnv
from envs.shop_env import ShopEnv
from envs.observations.shop_obs_dense import build_shop_obs
from envs.reward import ShopRewardConfig, make_shop_reward_fn

from models.q_networks import DQNNet
from utils.replay_buffer import ReplayBuffer


# ============================================================
# 工具：环境与 agent 构造（只用于推理）
# ============================================================

def make_shop_env(
    instance: InstanceData,
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    max_steps: int,
) -> ShopEnv:
    core_env = FlowShopCoreEnv(
        instance=instance,
        buffers=buffers,
    )
    obs_builder = build_shop_obs
    reward_fn = make_shop_reward_fn(reward_cfg)
    env = ShopEnv(
        core_env=core_env,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        max_steps=max_steps,
    )
    return env


def create_dispatch_agent_for_eval(
    obs_dim: int,
    action_dim: int,
    lr: float,
    buffer_capacity: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    只为加载权重 / 推理用的 agent：
      - DQNNet 结构与训练时一致
      - 虽然也创建了 optimizer / replay_buffer，但推理阶段不会用到
    """
    q_net = DQNNet(obs_dim, action_dim).to(device)
    target_q_net = DQNNet(obs_dim, action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity, obs_dim)

    agent: Dict[str, Any] = dict(
        q_net=q_net,
        target_q_net=target_q_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        cfg=dict(obs_dim=obs_dim, action_dim=action_dim),
        global_step=0,
    )
    return agent


def select_action_greedy(
    env: ShopEnv,
    obs: np.ndarray,
    agent: Dict[str, Any],
    device: torch.device,
) -> int:
    q_net: nn.Module = agent["q_net"]
    action_dim: int = agent["cfg"]["action_dim"]

    legal_actions = env._core_env.get_legal_actions()
    if not legal_actions:
        return int(np.random.randint(0, action_dim))

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        q_values = q_net(obs_tensor)[0].cpu().numpy()

    mask = np.ones(action_dim, dtype=bool)
    mask[legal_actions] = False
    q_values[mask] = -1e9
    return int(q_values.argmax())


# ============================================================
# 仅推理：在 test 集上跑若干 greedy episodes（带明细）
# ============================================================

def run_greedy_eval_episodes(
    agent: Dict[str, Any],
    instances: List[InstanceData],
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
    num_episodes: int = 100,
) -> Dict[str, Any]:
    """
    从 instances 中随机采样，每次用 greedy 策略跑一集，总共 num_episodes 次。
    返回：
      - avg_makespan, deadlock_rate, episodes
      - details: List[dict]，每一局的细节（episode_id, instance_idx, makespan, deadlock, steps）
    """
    details: List[Dict[str, Any]] = []

    if (not instances) or num_episodes <= 0:
        return dict(
            avg_makespan=float("inf"),
            deadlock_rate=0.0,
            episodes=0,
            details=details,
        )

    total_makespan = 0.0
    total_deadlock = 0
    episodes = 0

    for ep in range(num_episodes):
        inst_idx = int(np.random.randint(len(instances)))
        inst = instances[inst_idx]

        env = make_shop_env(
            instance=inst,
            buffers=buffers,
            reward_cfg=reward_cfg,
            max_steps=max_steps,
        )
        obs = env.reset()
        done = False
        steps = 0
        last_info: Dict[str, Any] = {}

        while (not done) and (steps < max_steps):
            action = select_action_greedy(env, obs, agent, device)
            obs, reward, done, info = env.step(action)
            steps += 1
            last_info = info

        makespan = float(last_info.get("makespan", max_steps))
        deadlock = bool(last_info.get("deadlock", False))
        if (not last_info) or (steps >= max_steps and "makespan" not in last_info):
            deadlock = True

        total_makespan += makespan
        total_deadlock += 1 if deadlock else 0
        episodes += 1

        details.append(
            dict(
                episode=ep + 1,
                instance_idx=inst_idx,
                makespan=makespan,
                deadlock=int(deadlock),
                steps=steps,
            )
        )

    avg_makespan = total_makespan / episodes
    deadlock_rate = total_deadlock / episodes

    return dict(
        avg_makespan=float(avg_makespan),
        deadlock_rate=float(deadlock_rate),
        episodes=int(episodes),
        details=details,
    )


# ============================================================
# CFG / 实例加载
# ============================================================

def load_cfg_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "cfg.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg.json not found in run_dir: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def load_instances_for_experiment(experiment_name: str) -> Tuple[List[InstanceData], List[InstanceData], List[InstanceData]]:
    data_root = os.path.join(ROOT_DIR, "experiments", "raw", experiment_name)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_instances = load_instances_from_dir(train_dir)
    val_instances = load_instances_from_dir(val_dir) if os.path.isdir(val_dir) else []
    test_instances = load_instances_from_dir(test_dir) if os.path.isdir(test_dir) else []

    print(
        f"[INFO] Loaded instances ({experiment_name}): "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )
    return train_instances, val_instances, test_instances


# ============================================================
# 单个 run_dir 的 eval 逻辑
# ============================================================

def eval_one_run_dir(
    run_dir: Path,
    device: torch.device,
    num_episodes: int,
    ckpt_tag: str,
) -> None:
    print(f"\n[RUN_EVAL] run_dir = {run_dir}")

    # 1) 读取 cfg.json
    cfg_json = load_cfg_from_run_dir(run_dir)
    experiment_name = cfg_json.get("experiment_name", EXPERIMENT_NAME)
    fixed_buffers = cfg_json.get("fixed_buffers", None)
    max_steps = int(cfg_json.get("max_steps_per_episode", 2000))

    if fixed_buffers is None:
        raise ValueError(f"fixed_buffers not found in cfg.json of run_dir={run_dir}")

    print(f"[INFO] experiment_name = {experiment_name}")
    print(f"[INFO] fixed_buffers   = {fixed_buffers}")
    print(f"[INFO] max_steps       = {max_steps}")

    # 2) 还原 reward_cfg
    reward_cfg_dict = cfg_json.get("reward_cfg", None)
    if reward_cfg_dict is None:
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
        print("[WARN] reward_cfg not found in cfg.json, using default progress config.")
    else:
        reward_cfg = ShopRewardConfig(**reward_cfg_dict)

    # 3) 还原 agent 维度（obs_dim / action_dim）
    agent_cfg_dict = cfg_json.get("agent_cfg", None)
    if agent_cfg_dict is None:
        raise ValueError(f"agent_cfg not found in cfg.json of run_dir={run_dir}")
    obs_dim = int(agent_cfg_dict.get("obs_dim"))
    action_dim = int(agent_cfg_dict.get("action_dim"))
    lr = float(agent_cfg_dict.get("lr", 1e-4))
    buffer_capacity = int(agent_cfg_dict.get("buffer_capacity", 100_000))

    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}")

    # 4) 加载 test 实例
    _, _, test_instances = load_instances_for_experiment(experiment_name)
    if not test_instances:
        print(f"[WARN] No test instances for experiment={experiment_name}, skip run_dir={run_dir}")
        return

    # 5) 构造 agent，并加载 checkpoint
    agent = create_dispatch_agent_for_eval(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=lr,
        buffer_capacity=buffer_capacity,
        device=device,
    )

    if ckpt_tag == "last":
        ckpt_path = run_dir / "dispatch_q_last.pth"
    else:
        ckpt_path = run_dir / ckpt_tag

    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found: {ckpt_path}, skip this run.")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    agent["q_net"].load_state_dict(state_dict)
    print(f"[INFO] Loaded checkpoint from {ckpt_path}")

    # 6) 在 test 上跑 num_episodes 次 greedy，并记录 detail
    metrics = run_greedy_eval_episodes(
        agent=agent,
        instances=test_instances,
        buffers=list(fixed_buffers),
        reward_cfg=reward_cfg,
        device=device,
        max_steps=max_steps,
        num_episodes=num_episodes,
    )

    print(
        f"[Eval][TEST] episodes={metrics['episodes']}, "
        f"avg_makespan={metrics['avg_makespan']:.3f}, "
        f"deadlock_rate={metrics['deadlock_rate']:.3f}"
    )

    # 7) 写 summary
    summary_csv = run_dir / "eval_test_summary_detail.csv"
    with summary_csv.open("w", newline="") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            ["split", "num_eval_episodes", "avg_makespan", "deadlock_rate", "ckpt"]
        )
        writer.writerow(
            [
                "test",
                int(metrics["episodes"]),
                float(metrics["avg_makespan"]),
                float(metrics["deadlock_rate"]),
                str(ckpt_path.name),
            ]
        )
    print(f"[SAVE] Summary saved to {summary_csv}")

    # 8) 写明细
    detail_csv = run_dir / "eval_test_detail.csv"
    with detail_csv.open("w", newline="") as f_det:
        writer = csv.writer(f_det)
        writer.writerow(
            ["episode", "instance_idx", "makespan", "deadlock", "steps"]
        )
        for row in metrics["details"]:
            writer.writerow(
                [
                    int(row["episode"]),
                    int(row["instance_idx"]),
                    float(row["makespan"]),
                    int(row["deadlock"]),
                    int(row["steps"]),
                ]
            )
    print(f"[SAVE] Detail saved to {detail_csv}")


# ============================================================
# main：一键批量
# ============================================================

def main():
    # 设备
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 自动推断 parent_dir
    parent_dir = Path(ROOT_DIR) / "results" / EXPERIMENT_NAME / "dispatch_d3qn_fixedbuf"
    parent_dir = parent_dir.resolve()
    print(f"[INFO] parent_dir = {parent_dir}")
    print(f"[INFO] pattern    = {PATTERN}")
    print(f"[INFO] num_episodes = {NUM_EPISODES}")
    print(f"[INFO] ckpt_tag   = {CKPT_TAG}")

    if not parent_dir.exists():
        print(f"[ERROR] parent_dir not found: {parent_dir}")
        return

    run_dirs = sorted(parent_dir.glob(PATTERN))
    if not run_dirs:
        print(f"[WARN] No run dirs found under {parent_dir} with pattern {PATTERN}")
        return

    print(f"[INFO] Found {len(run_dirs)} run dirs to evaluate.")

    for rd in run_dirs:
        if not rd.is_dir():
            continue
        try:
            eval_one_run_dir(
                run_dir=rd,
                device=device,
                num_episodes=NUM_EPISODES,
                ckpt_tag=CKPT_TAG,
            )
        except Exception as e:
            print(f"[ERROR] Failed to eval run_dir={rd}: {e}")


if __name__ == "__main__":
    main()

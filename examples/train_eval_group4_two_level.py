# examples/train_eval_group4_two_level.py
"""
组4：两层端到端（上层 Buffer Agent + 下层 D3QN 派工，BA+DA）

功能：
    - 对多个 seed 进行训练（上层缓冲 + 下层调度联合训练）
    - 训练结束后，在 test 集上做 greedy 推理若干次
    - 为每个 seed 生成：
        * train_log.csv        —— 训练日志（按 outer episode）
        * eval_test_summary_detail.csv  —— test 汇总
        * eval_test_detail.csv          —— test 每局明细

使用方式：
    直接运行本脚本（不需要命令行参数）：
        python examples/train_eval_group4_two_level.py

如需更换算例，只需修改顶部常量：
    EXPERIMENT_NAME
"""

from __future__ import annotations

import os
import sys
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------- 可调常量（后面只改这里即可） -----------------
EXPERIMENT_NAME = "j50s3m3"
METHOD_NAME = "group4_two_level"   # 结果目录名
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

NUM_OUTER_EPISODES = 400           # 上层 episode 数
NUM_INNER_EPISODES = 5             # 每次上层缓冲评估前，下层 inner 训练轮数
MAX_LOWER_STEPS = 2000             # 下层 max_steps_per_episode
NUM_EVAL_EPISODES = 100            # 每个 seed 的 test 推理次数

DEVICE_STR = "cuda"                # 默认用 cuda
BUFFER_COST_WEIGHT = 1.0
DEADLOCK_PENALTY = 2000.0

# 下层调度 D3QN 超参（参考组3）
LOWER_LR = 1e-3
LOWER_GAMMA = 0.99
LOWER_BATCH_SIZE = 64
LOWER_BUFFER_CAPACITY = 100_000
LOWER_TARGET_UPDATE_INTERVAL = 500

# 上层 D3QN 超参（参考你标准 upper 配置）
UPPER_LR = 1e-4
UPPER_GAMMA = 0.99
UPPER_BATCH_SIZE = 128
UPPER_BUFFER_CAPACITY = 10_000
UPPER_TARGET_UPDATE_INTERVAL = 100

# epsilon：统一按 episode 衰减（和你上层 / 组3 对齐）
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY_RATE = 0.992
# -----------------------------------------------------------


# ---------- 路径设置：把 src/ 加到 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ---------- 导入项目内模块 ----------
from instances.io import load_instances_from_dir
from instances.types import InstanceData
from envs.ffs_core_env import FlowShopCoreEnv
from envs.shop_env import ShopEnv
from envs.observations.shop_obs_dense import build_shop_obs
from envs.reward import ShopRewardConfig, make_shop_reward_fn
from envs.buffer_design_env import (
    BufferDesignEnv,
    BufferDesignEnvConfig,
    compute_buffer_upper_bounds,
)
from models.q_networks import DQNNet
from utils.replay_buffer import ReplayBuffer
from policies.dqn_common import dqn_update_step
from policies.upper_buffer_agent import UpperAgentConfig, UpperAgent, create_upper_agent


# ============================================================
# dataclass：下层配置 + 组4 TwoLevelConfig
# ============================================================

@dataclass
class LowerAgentConfig:
    obs_dim: int
    action_dim: int
    gamma: float = LOWER_GAMMA
    lr: float = LOWER_LR
    batch_size: int = LOWER_BATCH_SIZE
    buffer_capacity: int = LOWER_BUFFER_CAPACITY
    target_update_interval: int = LOWER_TARGET_UPDATE_INTERVAL
    # epsilon 按 episode 衰减：
    epsilon_start: float = EPSILON_START
    epsilon_end: float = EPSILON_END
    epsilon_decay_rate: float = EPSILON_DECAY_RATE
    max_steps_per_episode: int = MAX_LOWER_STEPS
    num_inner_episodes: int = NUM_INNER_EPISODES


@dataclass
class Group4TwoLevelConfig:
    experiment_name: str
    method_name: str
    num_outer_episodes: int
    device: str
    random_seed: int

    lower_agent_cfg: LowerAgentConfig
    upper_agent_cfg: UpperAgentConfig
    lower_reward_cfg: ShopRewardConfig

    buffer_cost_weight: float = BUFFER_COST_WEIGHT
    deadlock_penalty: float = DEADLOCK_PENALTY

    enable_train_log: bool = True


# ============================================================
# 实例加载
# ============================================================

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
# 下层 env / agent / epsilon / action
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


def create_lower_agent(cfg: LowerAgentConfig, device: torch.device) -> Dict[str, Any]:
    q_net = DQNNet(cfg.obs_dim, cfg.action_dim).to(device)
    target_q_net = DQNNet(cfg.obs_dim, cfg.action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay_buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

    agent = dict(
        q_net=q_net,
        target_q_net=target_q_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        cfg=cfg,
        global_step=0,
        global_episode=0,
    )
    return agent


def compute_lower_epsilon(agent: Dict[str, Any]) -> float:
    cfg: LowerAgentConfig = agent["cfg"]
    ep = agent.get("global_episode", 0)
    eps = cfg.epsilon_start * (cfg.epsilon_decay_rate ** ep)
    if eps < cfg.epsilon_end:
        eps = cfg.epsilon_end
    return float(eps)


def select_lower_action_epsilon_greedy(
    env: ShopEnv,
    obs: np.ndarray,
    agent: Dict[str, Any],
    device: torch.device,
) -> int:
    cfg: LowerAgentConfig = agent["cfg"]
    q_net: DQNNet = agent["q_net"]
    epsilon = compute_lower_epsilon(agent)
    action_dim = cfg.action_dim

    legal_actions = env._core_env.get_legal_actions()

    if (not legal_actions) or (np.random.rand() < epsilon):
        if legal_actions:
            return int(np.random.choice(legal_actions))
        return int(np.random.randint(0, action_dim))

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        q_values = q_net(obs_tensor)[0].cpu().numpy()

    mask = np.ones(action_dim, dtype=bool)
    mask[legal_actions] = False
    q_values[mask] = -1e9

    return int(q_values.argmax())


def select_lower_action_greedy(
    env: ShopEnv,
    obs: np.ndarray,
    agent: Dict[str, Any],
    device: torch.device,
) -> int:
    cfg: LowerAgentConfig = agent["cfg"]
    q_net: DQNNet = agent["q_net"]
    action_dim = cfg.action_dim

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


def run_lower_episode_and_learn(
    instance: InstanceData,
    buffers: List[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
) -> Dict[str, Any]:
    cfg: LowerAgentConfig = agent["cfg"]

    env = make_shop_env(
        instance=instance,
        buffers=buffers,
        reward_cfg=reward_cfg,
        max_steps=cfg.max_steps_per_episode,
    )
    obs = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while (not done) and (steps < cfg.max_steps_per_episode):
        action = select_lower_action_epsilon_greedy(env, obs, agent, device)
        next_obs, reward, done, info = env.step(action)

        agent["replay_buffer"].add(obs, action, reward, next_obs, done)
        agent["global_step"] += 1
        obs = next_obs
        steps += 1
        ep_reward += reward
        last_info = info

        loss = dqn_update_step(
            agent["q_net"],
            agent["target_q_net"],
            agent["optimizer"],
            agent["replay_buffer"],
            cfg.batch_size,
            cfg.gamma,
            device,
        )

        if agent["global_step"] % cfg.target_update_interval == 0:
            agent["target_q_net"].load_state_dict(agent["q_net"].state_dict())

    makespan = float(last_info.get("makespan", cfg.max_steps_per_episode))
    deadlock_flag = bool(last_info.get("deadlock", False))

    return dict(
        ep_reward=ep_reward,
        steps=steps,
        makespan=makespan,
        deadlock=deadlock_flag,
    )


def evaluate_lower_greedy(
    instance: InstanceData,
    buffers: List[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    env = make_shop_env(
        instance=instance,
        buffers=buffers,
        reward_cfg=reward_cfg,
        max_steps=max_steps,
    )
    obs = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while (not done) and (steps < max_steps):
        action = select_lower_action_greedy(env, obs, agent, device)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        steps += 1
        last_info = info

    makespan = float(last_info.get("makespan", max_steps))
    deadlock_flag = bool(last_info.get("deadlock", False))
    if (not last_info) or (steps >= max_steps and "makespan" not in last_info):
        deadlock_flag = True

    return dict(
        ep_reward=ep_reward,
        steps=steps,
        makespan=makespan,
        deadlock=deadlock_flag,
    )


# ============================================================
# 上层 evaluate_fn for BufferDesignEnv
# ============================================================

def make_evaluate_fn_for_upper(
    lower_agent: Dict[str, Any],
    lower_reward_cfg: ShopRewardConfig,
    device: torch.device,
    num_inner_episodes: int,
    max_lower_steps: int,
    lower_train_mode: str = "da",
) -> Any:
    """
    组4训练用：
        - lower_train_mode="da" 时：先做 num_inner_episodes 次 inner 训练，再评估 greedy。
        - lower_train_mode="rd" 时：只评估 greedy（不更新下层），用于 test eval。
    """

    mode = lower_train_mode.lower()

    def evaluate_fn(instance: InstanceData, buffers: List[int]) -> Dict[str, float]:
        if mode == "da":
            for _ in range(num_inner_episodes):
                run_lower_episode_and_learn(
                    instance=instance,
                    buffers=buffers,
                    agent=lower_agent,
                    reward_cfg=lower_reward_cfg,
                    device=device,
                )

        eval_metrics = evaluate_lower_greedy(
            instance=instance,
            buffers=buffers,
            agent=lower_agent,
            reward_cfg=lower_reward_cfg,
            device=device,
            max_steps=max_lower_steps,
        )
        makespan = float(eval_metrics["makespan"])
        deadlock = bool(eval_metrics.get("deadlock", False))
        return {
            "makespan": makespan,
            "deadlock": deadlock,
        }

    return evaluate_fn


def select_upper_action_greedy(
    obs: np.ndarray,
    agent: UpperAgent,
    device: torch.device,
) -> int:
    # UpperAgent 自带 greedy 接口
    return int(agent.select_greedy_action(obs))


# ============================================================
# test 集 eval：给定训练好的 upper_agent + lower_agent
# ============================================================

def evaluate_group4_on_test(
    cfg: Group4TwoLevelConfig,
    upper_agent: UpperAgent,
    lower_agent: Dict[str, Any],
    test_instances: List[InstanceData],
    device: torch.device,
    out_dir: Path,
    num_eval_episodes: int = NUM_EVAL_EPISODES,
) -> None:
    if not test_instances:
        print("[TEST] No test instances, skip.")
        return

    out_dir = Path(out_dir)
    summary_path = out_dir / "eval_test_summary_detail.csv"
    detail_path = out_dir / "eval_test_detail.csv"

    # eval 时：下层不再训练，使用 "rd" 模式
    eval_fn = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=0,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode="rd",
    )

    # BufferDesignEnvConfig：不随机实例，每个 episode 只给一个实例
    eval_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=False,
        max_total_buffer=None,
    )

    rng = np.random.RandomState(cfg.random_seed + 999)

    total_ms = 0.0
    total_dead = 0
    episodes = 0
    details: List[Dict[str, Any]] = []

    for ep in range(num_eval_episodes):
        inst_idx = int(rng.randint(len(test_instances)))
        inst = test_instances[inst_idx]

        eval_env = BufferDesignEnv(
            instances=[inst],
            evaluate_fn=eval_fn,
            cfg=eval_env_cfg,
            obs_cfg=None,
            seed=cfg.random_seed + ep,
            custom_reward_fn=None,
        )

        obs_U = eval_env.reset()
        done_U = False

        while not done_U:
            action_U = select_upper_action_greedy(obs_U, upper_agent, device)
            obs_U, r_U, done_U, info_U = eval_env.step(action_U)

        metrics = info_U.get("metrics", {})
        ms = float(metrics.get("makespan", math.inf))
        bufs = metrics.get("buffers", [])
        dl = bool(metrics.get("deadlock", False))

        total_ms += ms
        total_dead += 1 if dl else 0
        episodes += 1

        if isinstance(bufs, (list, tuple)):
            buf_str = " ".join(str(b) for b in bufs)
        else:
            buf_str = str(bufs)

        details.append(
            dict(
                episode=ep + 1,
                instance_idx=inst_idx,
                buffers=buf_str,
                makespan=ms,
                deadlock=int(dl),
            )
        )

    if episodes == 0:
        print("[TEST] No episodes evaluated on test, skip writing csv.")
        return

    avg_ms = total_ms / episodes
    dead_rate = total_dead / episodes

    print(
        f"[Eval][TEST] episodes={episodes}, "
        f"avg_makespan={avg_ms:.3f}, deadlock_rate={dead_rate:.3f}"
    )

    # 写 summary
    with summary_path.open("w", newline="") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            ["split", "num_eval_episodes", "avg_makespan", "deadlock_rate", "ckpt"]
        )
        writer.writerow(
            [
                "test",
                int(episodes),
                float(avg_ms),
                float(dead_rate),
                "upper_q_last.pth",
            ]
        )
    print(f"[SAVE] Test summary saved to {summary_path}")

    # 写 detail
    with detail_path.open("w", newline="") as f_det:
        writer = csv.writer(f_det)
        writer.writerow(
            ["episode", "instance_idx", "buffers", "makespan", "deadlock"]
        )
        for row in details:
            writer.writerow(
                [
                    int(row["episode"]),
                    int(row["instance_idx"]),
                    row["buffers"],
                    float(row["makespan"]),
                    int(row["deadlock"]),
                ]
            )
    print(f"[SAVE] Test detail saved to {detail_path}")


# ============================================================
# 单个 seed 的训练 + eval
# ============================================================

def build_group4_config(
    experiment_name: str,
    seed: int,
    dummy_instance: InstanceData,
    device_str: str,
) -> Group4TwoLevelConfig:
    # 下层 reward：progress，与组3一致
    lower_reward_cfg = ShopRewardConfig(
        mode="progress",
        time_weight=1.0,
        per_operation_reward=0.05,
        per_job_reward=0.1,
        blocking_penalty=0.2,
        terminal_bonus=0.5,
        invalid_action_weight=0.2,
        makespan_weight=0.0,
    )

    # 下层 obs_dim / action_dim 后面再根据 dummy env 写回去
    lower_agent_cfg = LowerAgentConfig(
        obs_dim=0,
        action_dim=0,
        num_inner_episodes=NUM_INNER_EPISODES,
        max_steps_per_episode=MAX_LOWER_STEPS,
    )

    # 上层 D3QN 配置
    upper_agent_cfg = UpperAgentConfig(
        obs_dim=0,
        action_dim=0,
        gamma=UPPER_GAMMA,
        lr=UPPER_LR,
        batch_size=UPPER_BATCH_SIZE,
        buffer_capacity=UPPER_BUFFER_CAPACITY,
        target_update_interval=UPPER_TARGET_UPDATE_INTERVAL,
        buffer_cost_weight=BUFFER_COST_WEIGHT,
        algo_type="d3qn",
        replay_type="uniform",
    )
    # epsilon 在 UpperAgent 内部用自己的参数，你如果想精调，也可以在 UpperAgentConfig 里设

    cfg = Group4TwoLevelConfig(
        experiment_name=experiment_name,
        method_name=METHOD_NAME,
        num_outer_episodes=NUM_OUTER_EPISODES,
        device=device_str,
        random_seed=seed,
        lower_agent_cfg=lower_agent_cfg,
        upper_agent_cfg=upper_agent_cfg,
        lower_reward_cfg=lower_reward_cfg,
        buffer_cost_weight=BUFFER_COST_WEIGHT,
        deadlock_penalty=DEADLOCK_PENALTY,
        enable_train_log=True,
    )
    return cfg


def train_and_eval_one_seed(
    seed: int,
    train_instances: List[InstanceData],
    test_instances: List[InstanceData],
    base_out_dir: Path,
) -> None:
    device = torch.device(DEVICE_STR if torch.cuda.is_available() else "cpu")

    dummy_instance = train_instances[0]
    machines_per_stage = dummy_instance.machines_per_stage
    num_stages = len(machines_per_stage)
    num_buffers = num_stages - 1

    # 构造 config
    cfg = build_group4_config(
        experiment_name=EXPERIMENT_NAME,
        seed=seed,
        dummy_instance=dummy_instance,
        device_str=DEVICE_STR,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"seed{seed}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] experiment={EXPERIMENT_NAME}, method={METHOD_NAME}, seed={seed}, out_dir={out_dir}")
    print(f"[INFO] Using device: {device}")

    # 下层 dummy env -> lower_obs_dim
    tmp_env = make_shop_env(
        instance=dummy_instance,
        buffers=[0] * num_buffers,
        reward_cfg=cfg.lower_reward_cfg,
        max_steps=cfg.lower_agent_cfg.max_steps_per_episode,
    )
    tmp_obs = tmp_env.reset()
    lower_obs_dim = tmp_obs.shape[0]

    # 上层：用 BufferDesignEnv dummy env 拿 obs_dim / action_dim
    def dummy_eval_fn(inst, bufs):
        return {"makespan": 0.0, "deadlock": False}

    dummy_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=False,
        max_total_buffer=None,
    )
    dummy_env = BufferDesignEnv(
        instances=[dummy_instance],
        evaluate_fn=dummy_eval_fn,
        cfg=dummy_env_cfg,
        obs_cfg=None,
        seed=seed,
        custom_reward_fn=None,
    )
    dummy_obs_U = dummy_env.reset()
    upper_obs_dim = dummy_obs_U.shape[0]

    # 动作维度：a_k ∈ {0,...,max_a}
    max_a = 0
    for inst in train_instances:
        bounds = compute_buffer_upper_bounds(inst)
        if bounds:
            max_a = max(max_a, max(bounds))
    if max_a <= 0:
        max_a = 1
    upper_action_dim = max_a + 1

    num_jobs = len(dummy_instance.jobs)
    lower_action_dim = num_jobs

    cfg.lower_agent_cfg.obs_dim = lower_obs_dim
    cfg.lower_agent_cfg.action_dim = lower_action_dim
    cfg.upper_agent_cfg.obs_dim = upper_obs_dim
    cfg.upper_agent_cfg.action_dim = upper_action_dim

    print(
        f"[INFO] lower_obs_dim={lower_obs_dim}, lower_action_dim={lower_action_dim}, "
        f"upper_obs_dim={upper_obs_dim}, upper_action_dim={upper_action_dim}"
    )

    # 构造上下层 agent
    lower_agent = create_lower_agent(cfg.lower_agent_cfg, device)
    upper_agent: UpperAgent = create_upper_agent(cfg.upper_agent_cfg, device)

    # 训练用 evaluate_fn（BA+DA 联合）
    train_eval_fn = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=cfg.lower_agent_cfg.num_inner_episodes,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode="da",
    )

    upper_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=True,
        max_total_buffer=None,
    )
    upper_env = BufferDesignEnv(
        instances=train_instances,
        evaluate_fn=train_eval_fn,
        cfg=upper_env_cfg,
        obs_cfg=None,
        seed=seed,
        custom_reward_fn=None,
    )

    # 保存 cfg.json
    cfg_json_path = out_dir / "cfg.json"
    try:
        with cfg_json_path.open("w", encoding="utf-8") as f_cfg:
            json.dump(
                dict(
                    experiment_name=cfg.experiment_name,
                    method_name=cfg.method_name,
                    random_seed=cfg.random_seed,
                    num_outer_episodes=cfg.num_outer_episodes,
                    buffer_cost_weight=cfg.buffer_cost_weight,
                    deadlock_penalty=cfg.deadlock_penalty,
                    lower_reward_cfg=cfg.lower_reward_cfg.__dict__,
                    lower_agent_cfg=cfg.lower_agent_cfg.__dict__,
                    upper_agent_cfg={
                        k: getattr(cfg.upper_agent_cfg, k)
                        for k in cfg.upper_agent_cfg.__dict__.keys()
                    },
                ),
                f_cfg,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] Saved cfg.json to {cfg_json_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cfg.json: {e}")

    # 打开 train_log.csv
    log_file = None
    log_writer = None
    if cfg.enable_train_log:
        log_path = out_dir / "train_log.csv"
        log_file = log_path.open("w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "outer_ep",
                "instance_idx",
                "upper_ep_reward",
                "final_makespan",
                "deadlock",
                "total_buffer",
                "epsilon_U",
                "buffers",
            ]
        )
        log_file.flush()

    # ------------ 训练主循环 ------------
    try:
        for outer_ep in range(cfg.num_outer_episodes):
            upper_agent.global_episode = outer_ep
            epsilon_U = upper_agent.compute_epsilon()

            # sync 下层 episode index（用于 epsilon）
            lower_agent["global_episode"] = outer_ep

            obs_U = upper_env.reset()
            done_U = False
            ep_reward_U = 0.0
            traj: List[Tuple[np.ndarray, int, float, np.ndarray]] = []

            while not done_U:
                action_U = upper_agent.select_action(obs_U, epsilon_U)
                next_obs_U, reward_U, done_U, info_U = upper_env.step(action_U)
                traj.append((obs_U, action_U, float(reward_U), next_obs_U))
                obs_U = next_obs_U
                ep_reward_U += float(reward_U)

            metrics_U = info_U.get("metrics", {})
            final_makespan = float(metrics_U.get("makespan", 0.0))
            deadlock_flag = bool(metrics_U.get("deadlock", False))
            bufs = metrics_U.get("buffers", [])
            total_buffer = float(sum(bufs)) if isinstance(bufs, (list, tuple)) else 0.0
            instance_idx = info_U.get("instance_idx", -1)

            # 回放到 upper_agent 的 replay buffer
            for i, (s, a, r, s_next) in enumerate(traj):
                done_flag = (i == len(traj) - 1)
                upper_agent.add_transition(s, a, r, s_next, done_flag)

            # 上层更新
            upper_agent.update_one_step()
            if (outer_ep + 1) % cfg.upper_agent_cfg.target_update_interval == 0:
                upper_agent.update_target()

            # 训练 log
            if isinstance(bufs, (list, tuple)):
                buffer_str = " ".join(str(b) for b in bufs)
            else:
                buffer_str = str(bufs)

            if log_writer is not None:
                log_writer.writerow(
                    [
                        outer_ep,
                        instance_idx,
                        float(ep_reward_U),
                        float(final_makespan),
                        int(deadlock_flag),
                        total_buffer,
                        float(epsilon_U),
                        buffer_str,
                    ]
                )
                if (outer_ep + 1) % 100 == 0:
                    log_file.flush()

            if (outer_ep + 1) % 100 == 0 or outer_ep == 0:
                print(
                    f"[OuterEP {outer_ep+1:04d}] reward={ep_reward_U:.3f}, "
                    f"makespan={final_makespan:.1f}, deadlock={int(deadlock_flag)}, "
                    f"epsilon_U={epsilon_U:.3f}"
                )

        # 保存 upper_agent 的 last checkpoint
        ckpt_path = out_dir / "upper_q_last.pth"
        torch.save(upper_agent.q_net.state_dict(), ckpt_path)
        print(f"[CKPT] Saved upper_agent q_net to {ckpt_path}")

    finally:
        if log_file is not None:
            log_file.close()

    # ------------ 训练结束，做 test eval ------------
    evaluate_group4_on_test(
        cfg=cfg,
        upper_agent=upper_agent,
        lower_agent=lower_agent,
        test_instances=test_instances,
        device=device,
        out_dir=out_dir,
        num_eval_episodes=NUM_EVAL_EPISODES,
    )


# ============================================================
# main：多 seed 训练 + eval
# ============================================================

def main():
    print(f"[INFO] ROOT_DIR   = {ROOT_DIR}")
    print(f"[INFO] experiment = {EXPERIMENT_NAME}")
    print(f"[INFO] method     = {METHOD_NAME}")
    print(f"[INFO] seeds      = {SEEDS}")

    train_instances, val_instances, test_instances = load_instances_for_experiment(EXPERIMENT_NAME)
    if not train_instances:
        print("[ERROR] No train instances, abort.")
        return
    if not test_instances:
        print("[WARN] No test instances, test eval will be skipped.")

    base_out_dir = Path(ROOT_DIR) / "results" / EXPERIMENT_NAME / METHOD_NAME
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        try:
            train_and_eval_one_seed(
                seed=seed,
                train_instances=train_instances,
                test_instances=test_instances,
                base_out_dir=base_out_dir,
            )
        except Exception as e:
            print(f"[ERROR] Failed on seed={seed}: {e}")


if __name__ == "__main__":
    main()

# examples/train_two_level.py

import os
import sys

from pathlib import Path
import csv
import math
import itertools  # 用于枚举所有 buffer 向量

import json
from datetime import datetime

# 先把 src/ 加进 sys.path，确保能找到 instances, envs 等包
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dataclasses import dataclass
from typing import Any, Dict, Callable, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.q_networks import DQNNet
from utils.replay_buffer import ReplayBuffer
from policies.dqn_common import dqn_update_step
from policies.upper_buffer_agent import UpperAgentConfig, UpperAgent, create_upper_agent

from instances.io import load_instances_from_dir
from instances.types import InstanceData

from instances.generators import FlowShopGeneratorConfig
from envs.buffer_design_env import (
    BufferDesignEnv,
    BufferDesignEnvConfig,
    compute_buffer_upper_bounds,
)
from envs.shop_env import ShopEnv
from envs.ffs_core_env import FlowShopCoreEnv, simulate_instance_with_job_rule
from envs.observations.shop_obs_dense import build_shop_obs
from envs.reward import ShopRewardConfig, make_shop_reward_fn


# ======================
# 1. 配置 dataclass
# ======================

@dataclass
class LowerAgentConfig:
    # 下层 DQN 训练配置
    obs_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 100_000
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    max_steps_per_episode: int = 2000

    # 两时间尺度：每个上层 episode 内，下层训练的 inner episodes 数
    num_inner_episodes: int = 5


@dataclass
class TwoLevelConfig:
    # 算例生成配置
    instance_cfg: FlowShopGeneratorConfig

    # 缓冲上界列表 a_k
    # 若为 None，则上层动作空间的上界完全由实例决定：
    #   a_k = max{ m_k, m_{k+1} } （见 compute_buffer_upper_bounds）。
    # 若不为 None，则可以用于强行截断上界（做消融实验时用）。
    buffer_upper_bounds: Optional[List[int]] = None

    # 训练轮数（外层 episode）
    num_outer_episodes: int = 1000

    # device: "cpu" / "cuda"
    device: str = "cuda"

    # 嵌套 agent 配置（下层 / 上层）
    lower_agent_cfg: LowerAgentConfig = None
    upper_agent_cfg: UpperAgentConfig = None

    # 下层 reward 配置（用 progress 模式）
    lower_reward_cfg: ShopRewardConfig = None

    # 下层训练模式：
    #   "da" = 下层做 DQN 调度；
    #   "rd" = 下层只做贪婪评估，不训练；
    #   "rule" = 下层完全不用 DQN，走规则调度。
    lower_train_mode: str = "da"

    # 随机种子（整个脚本里多处会用到）
    random_seed: int = 0


# =====================================================
# 2. DQN 网络与 ReplayBuffer —— 下层环境构造
# =====================================================

def make_shop_env(
    instance,
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    max_steps: int,
):
    """
    根据当前算例 instance 和缓冲配置 buffers 构造一个 ShopEnv。

    对应你现在的 ShopEnv 定义：
        ShopEnv(core_env, obs_builder, reward_fn, max_steps=None)

    其中：
      - core_env = FlowShopCoreEnv(instance, buffers)
      - obs_builder = build_shop_obs  （函数）
      - reward_fn = make_shop_reward_fn(reward_cfg)
    """
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


# ============================================================
# [VAL] 下层：在给定 instance + buffers 下，用当前 lower_agent 做一次贪婪调度评估
# ============================================================

def simulate_instance_with_greedy_lower(
    cfg,
    instance: InstanceData,
    buffers: List[int],
    lower_agent,
    device,
) -> Tuple[float, bool]:
    """
    在固定缓冲配置 buffers 下，对单个 instance 跑一次下层调度评估，
    复用 evaluate_lower_greedy，返回 (makespan, deadlock_flag)。
    """
    max_steps = cfg.lower_agent_cfg.max_steps_per_episode

    metrics = evaluate_lower_greedy(
        instance=instance,
        buffers=buffers,
        agent=lower_agent,
        reward_cfg=cfg.lower_reward_cfg,
        device=device,
        max_steps=max_steps,
    )

    makespan = float(metrics.get("makespan", math.inf))
    deadlock = bool(metrics.get("deadlock", False))

    return makespan, deadlock


def select_upper_action_greedy(
    obs: np.ndarray,
    agent,
    device: torch.device,
) -> int:
    """
    上层纯贪心动作选择，用于 validation 评估（不探索）。

    - 如果 agent 是 UpperAgent，就直接用 agent.select_greedy_action(...)
    - 否则退化到旧版：使用 agent["q_net"] / agent["cfg"]
    """
    if isinstance(agent, UpperAgent) or hasattr(agent, "select_greedy_action"):
        return int(agent.select_greedy_action(obs))

    cfg: UpperAgentConfig = agent["cfg"]
    q_net: DQNNet = agent["q_net"]
    action_dim = cfg.action_dim

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        q_values = q_net(obs_tensor)[0].cpu().numpy()
    action = int(q_values.argmax())

    if action < 0:
        action = 0
    elif action >= action_dim:
        action = action_dim - 1

    return action


# ============================================================
# [VAL] 在 validation set 上评估当前上下层策略（静态 BufferDesignEnv）
# ============================================================

def evaluate_on_validation(
    cfg,
    upper_agent,
    lower_agent,
    val_instances: List[InstanceData],
    device,
    out_dir: Path,
    outer_ep: int,
    eval_num_instances: int = 20,
    eval_runs_per_instance: int = 1,
) -> Optional[Dict[str, float]]:
    """
    使用当前的 upper_agent + lower_agent，在 val_instances 上做评估。

    - 上层：epsilon_U = 0.0，纯贪婪；
    - 下层：simulate_instance_with_greedy_lower，epsilon_L = 0.0；
    - 不做任何参数更新；
    - 这里只打印统计量，不再写 val_log.csv。
    """
    if not val_instances:
        print("[VAL] no validation instances, skip.")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_total = len(val_instances)
    num_use = min(eval_num_instances, num_total)

    base_seed = getattr(cfg, "random_seed", 0)
    rng = np.random.RandomState(base_seed + outer_ep)

    indices = rng.choice(num_total, size=num_use, replace=False)

    total_makespan = 0.0
    total_buffer = 0.0
    total_deadlock = 0
    total_episodes = 0

    for idx in indices:
        inst = val_instances[idx]

        for _ in range(eval_runs_per_instance):
            def eval_fn_for_this_inst(instance_, buffers_):
                mk, dl = simulate_instance_with_greedy_lower(
                    cfg, instance_, buffers_, lower_agent, device
                )
                metrics = {
                    "makespan": mk,
                    "buffers": list(buffers_),
                    "deadlock": dl,
                }
                return metrics

            upper_env_cfg = BufferDesignEnvConfig(
                buffer_cost_weight=cfg.upper_agent_cfg.buffer_cost_weight,
                deadlock_penalty=getattr(cfg.upper_agent_cfg, "deadlock_penalty", 1000.0),
                randomize_instances=False,
                max_total_buffer=None,
            )
            upper_env = BufferDesignEnv(
                instances=[inst],
                evaluate_fn=eval_fn_for_this_inst,
                cfg=upper_env_cfg,
                obs_cfg=None,
                seed=base_seed + outer_ep,
            )

            obs_U = upper_env.reset()
            done_U = False
            ep_reward_U = 0.0

            while not done_U:
                action_U = select_upper_action_greedy(obs_U, upper_agent, device)
                obs_U, r_U, done_U, info_U = upper_env.step(action_U)
                ep_reward_U += float(r_U)

            metrics = info_U.get("metrics", {})
            mk = float(metrics.get("makespan", math.inf))
            bufs = metrics.get("buffers", [])
            dl = bool(metrics.get("deadlock", False))

            total_makespan += mk
            total_buffer += float(sum(bufs))
            total_deadlock += 1 if dl else 0
            total_episodes += 1

    if total_episodes == 0:
        print("[VAL] no episodes evaluated, skip logging.")
        return None

    avg_makespan = total_makespan / total_episodes
    avg_total_buffer = total_buffer / total_episodes
    deadlock_rate = total_deadlock / total_episodes

    print(
        f"[VAL][outer_ep={outer_ep}] "
        f"avg_makespan={avg_makespan:.3f}, "
        f"avg_total_buffer={avg_total_buffer:.3f}, "
        f"deadlock_rate={deadlock_rate:.3f}, "
        f"episodes={total_episodes}"
    )

    return {
        "avg_makespan": float(avg_makespan),
        "avg_total_buffer": float(avg_total_buffer),
        "deadlock_rate": float(deadlock_rate),
        "episodes": int(total_episodes),
    }


# ============================================================
# [OFFLINE] 在 val 集上对所有缓冲向量做一次离线评估（静态）
# ============================================================

def _compute_global_buffer_upper_bounds(
    instances: List[InstanceData],
) -> List[int]:
    """
    对给定实例集合，按 compute_buffer_upper_bounds 计算每个缓冲段的全局上界。
    """
    if not instances:
        return []

    base_bounds = compute_buffer_upper_bounds(instances[0])
    if not base_bounds:
        return []

    global_bounds = list(base_bounds)
    for inst in instances[1:]:
        bounds = compute_buffer_upper_bounds(inst)
        if not bounds:
            continue
        n = min(len(global_bounds), len(bounds))
        for k in range(n):
            global_bounds[k] = max(global_bounds[k], bounds[k])

    return global_bounds


def _enumerate_candidate_buffers(upper_bounds: List[int]) -> List[List[int]]:
    """
    给定每个缓冲段的上界，例如 [3, 3]，枚举所有 0..a_k 的组合：
      -> [[0,0], [0,1], ..., [3,3]]
    """
    if not upper_bounds:
        return []
    ranges = [range(b + 1) for b in upper_bounds]
    return [list(buf) for buf in itertools.product(*ranges)]


def build_buffer_candidates_for_upper(
    instances: List[InstanceData],
) -> List[List[int]]:
    """
    根据训练集实例，构造上层可选的 buffer 向量集合。
    当前实现：先算全局上界，再做 0..a_k 的全枚举。
    （本文件当前训练流程未用到，仅保留以备后续分析/可视化。）
    """
    upper_bounds = _compute_global_buffer_upper_bounds(instances)
    if not upper_bounds:
        return []
    candidates = _enumerate_candidate_buffers(upper_bounds)
    return candidates


def offline_evaluate_all_buffers(
    cfg: TwoLevelConfig,
    lower_agent: Dict[str, Any],
    val_instances: List[InstanceData],
    device: torch.device,
    out_dir: str | Path,
) -> None:
    """
    使用当前 lower_agent（贪婪策略），在 val_instances 上对
    所有候选缓冲向量做一次离线评估。

    结果按 avg_makespan 升序排序后：
      1) 打印到控制台
      2) 写入 out_dir / 'offline_buffer_eval_val.csv'
    """
    if not val_instances:
        print("[OFFLINE_EVAL] no val instances, skip.")
        return

    per_buffer_upper = _compute_global_buffer_upper_bounds(val_instances)
    if not per_buffer_upper:
        print("[OFFLINE_EVAL] failed to compute buffer upper bounds, skip.")
        return

    candidate_buffers = _enumerate_candidate_buffers(per_buffer_upper)
    if not candidate_buffers:
        print("[OFFLINE_EVAL] no candidate buffers generated, skip.")
        return

    print(
        f"[OFFLINE_EVAL] buffer_upper_bounds={per_buffer_upper}, "
        f"num_candidates={len(candidate_buffers)}, "
        f"num_val_instances={len(val_instances)}"
    )

    results = []

    for buf_idx, buffers in enumerate(candidate_buffers):
        sum_makespan = 0.0
        sum_total_buffer = 0.0
        deadlocks = 0
        episodes = 0

        for inst in val_instances:
            makespan, deadlock = simulate_instance_with_greedy_lower(
                cfg=cfg,
                instance=inst,
                buffers=buffers,
                lower_agent=lower_agent,
                device=device,
            )
            sum_makespan += makespan
            sum_total_buffer += float(sum(buffers))
            deadlocks += 1 if deadlock else 0
            episodes += 1

        if episodes == 0:
            continue

        avg_makespan = sum_makespan / episodes
        avg_total_buffer = sum_total_buffer / episodes
        deadlock_rate = deadlocks / episodes

        results.append(
            dict(
                buffers=list(buffers),
                avg_makespan=avg_makespan,
                avg_total_buffer=avg_total_buffer,
                deadlock_rate=deadlock_rate,
                episodes=episodes,
            )
        )

    if not results:
        print("[OFFLINE_EVAL] no results computed, skip csv.")
        return

    results.sort(key=lambda x: x["avg_makespan"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "offline_buffer_eval_val.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "buffers", "avg_makespan", "avg_total_buffer", "deadlock_rate", "episodes"]
        )
        for rank, row in enumerate(results, start=1):
            buffer_str = " ".join(str(b) for b in row["buffers"])
            writer.writerow(
                [
                    rank,
                    buffer_str,
                    row["avg_makespan"],
                    row["avg_total_buffer"],
                    row["deadlock_rate"],
                    row["episodes"],
                ]
            )

    print("=== Offline evaluation of buffer vectors on VAL set ===")
    for rank, row in enumerate(results, start=1):
        bufs = " ".join(str(b) for b in row["buffers"])
        print(
            f"{rank:3d}: [{bufs}]  "
            f"avg_makespan={row['avg_makespan']:.3f}, "
            f"avg_total_buffer={row['avg_total_buffer']:.3f}, "
            f"deadlock_rate={row['deadlock_rate']:.3f}, "
            f"episodes={row['episodes']}"
        )
    print(f"[OFFLINE_EVAL] saved csv to: {csv_path}")


# ==================================
# 3. Agent 构造（上下两层）
# ==================================

def create_lower_agent(cfg: LowerAgentConfig, device: torch.device):
    """构造下层调度 DQN agent：网络 + target + optimizer + replay buffer。"""
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
    )
    return agent


def save_upper_agent_for_eval(upper_agent, out_dir: Path, tag: str = "best_val") -> None:
    """
    轻量级保存函数：只把用于推理的 q_net 权重存到 out_dir/upper_q_{tag}.pth。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"upper_q_{tag}.pth"

    if isinstance(upper_agent, dict):
        q_net: nn.Module = upper_agent["q_net"]
    else:
        q_net: nn.Module = upper_agent.q_net

    torch.save(q_net.state_dict(), ckpt_path)
    print(f"[CKPT] saved upper_agent q_net to {ckpt_path}")


def load_upper_agent_for_eval(upper_agent, out_dir: Path, tag: str = "best_val") -> None:
    """
    从 out_dir/upper_q_{tag}.pth 加载 q_net 权重到给定的 upper_agent 中。
    """
    out_dir = Path(out_dir)
    ckpt_path = out_dir / f"upper_q_{tag}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"upper-agent checkpoint not found: {ckpt_path}")

    if isinstance(upper_agent, dict):
        q_net: nn.Module = upper_agent["q_net"]
    else:
        q_net: nn.Module = upper_agent.q_net

    state_dict = torch.load(ckpt_path, map_location=next(q_net.parameters()).device)
    q_net.load_state_dict(state_dict)
    print(f"[CKPT] loaded upper_agent q_net from {ckpt_path}")


# =========================================
# 4. 下层 fast timescale：训练 + 贪心评估
# =========================================

def compute_lower_epsilon(agent: Dict[str, Any]) -> float:
    cfg: LowerAgentConfig = agent["cfg"]
    t = agent["global_step"]
    frac = min(1.0, t / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_lower_action_epsilon_greedy(
    env,
    obs: np.ndarray,
    agent: Dict[str, Any],
    device: torch.device,
) -> int:
    """带合法动作掩码的 epsilon-greedy：给下层调度 env 用。"""
    cfg: LowerAgentConfig = agent["cfg"]
    q_net: DQNNet = agent["q_net"]
    epsilon = compute_lower_epsilon(agent)

    legal_actions = env._core_env.get_legal_actions()
    action_dim = cfg.action_dim

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


def select_lower_action_greedy(env, obs: np.ndarray, agent: Dict[str, Any], device: torch.device) -> int:
    """纯贪心策略，用于对当前下层策略做评估。"""
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
    instance,
    buffers: List[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    在给定 instance + buffers 下跑一局调度，
    使用 epsilon-greedy，下层 DQN 做 fast timescale 更新。
    """
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

        dqn_update_step(
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
    metrics = dict(
        ep_reward=ep_reward,
        steps=steps,
        makespan=makespan,
    )
    return metrics


def evaluate_lower_greedy(
    instance,
    buffers: List[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    """
    用当前下层 DQN 的贪心策略，对给定 instance + buffers 做一次评估。
    """
    env = make_shop_env(
        instance=instance,
        buffers=buffers,
        reward_cfg=reward_cfg,
        max_steps=max_steps,
    )
    obs = env.reset()
    done = False
    steps = 0
    ep_reward = 0.0
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

    return {
        "ep_reward": ep_reward,
        "steps": steps,
        "makespan": makespan,
        "deadlock": deadlock_flag,
    }


# =================================================
# 5. 上层 evaluate_fn：静态 BufferDesignEnv 使用
# =================================================

def make_evaluate_fn_for_upper(
    lower_agent: Dict[str, Any],
    lower_reward_cfg: ShopRewardConfig,
    device: torch.device,
    num_inner_episodes: int,
    max_lower_steps: int,
    lower_train_mode: str = "da",
    job_rule: str | None = None,
) -> Callable[[Any, List[int]], Dict[str, float]]:
    """
    返回给 BufferDesignEnv 使用的 evaluate_fn(instance, buffers)。

    lower_train_mode:
        - "da": 下层做真正 DQN 调度（每次调用前先 inner 训练 num_inner_episodes）。
        - "rd": 下层只做 DQN 贪婪评估，不做任何参数更新。
        - "rule": 不用下层 DQN，直接用规则调度（simulate_instance_with_job_rule）。
    """
    mode = lower_train_mode.lower()
    job_rule = (job_rule or "spt").lower()

    def evaluate_fn(instance, buffers: List[int]) -> Dict[str, float]:
        # 1) fast timescale: 下层 inner 训练（仅在 "da" 模式）
        if mode == "da":
            for _ in range(num_inner_episodes):
                run_lower_episode_and_learn(
                    instance=instance,
                    buffers=buffers,
                    agent=lower_agent,
                    reward_cfg=lower_reward_cfg,
                    device=device,
                )

        # 2) 评估：
        if mode == "rule":
            # 规则调度：直接调用 FlowShopCoreEnv 的规则仿真
            metrics = simulate_instance_with_job_rule(
                instance=instance,
                buffers=buffers,
                job_rule=job_rule,
                machine_rule="min_index",
                max_steps=max_lower_steps,
                seed=None,
            )
        else:
            # "da" / "rd"：用下层 DQN 的贪婪策略评估
            metrics = evaluate_lower_greedy(
                instance=instance,
                buffers=buffers,
                agent=lower_agent,
                reward_cfg=lower_reward_cfg,
                device=device,
                max_steps=max_lower_steps,
            )

        makespan = float(metrics.get("makespan", math.inf))
        deadlock = bool(metrics.get("deadlock", False))
        return {
            "makespan": makespan,
            "deadlock": deadlock,
        }

    return evaluate_fn


# ===========================================
# 6. 上层 DQN 训练主循环（slow timescale, 静态上层）
# ===========================================

def compute_upper_epsilon(agent) -> float:
    """
    根据 agent 内部的 global_episode 计算当前 epsilon。
    """
    if isinstance(agent, UpperAgent):
        cfg: UpperAgentConfig = agent.cfg
        ep = getattr(agent, "global_episode", 0)
    else:
        cfg: UpperAgentConfig = agent["cfg"]
        ep = agent.get("global_episode", 0)

    frac = min(1.0, ep / float(cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def train_two_level(
    cfg: TwoLevelConfig,
    train_instances,
    val_instances,
    test_instances,
    out_dir: str,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # === C1: dummy 实例 & 一些结构信息 ===
    dummy_instance = train_instances[0]

    first_inst = train_instances[0]
    print(
        "[DEBUG] machines_per_stage =",
        list(first_inst.machines_per_stage),
        "buffer_upper_bounds =",
        compute_buffer_upper_bounds(first_inst),
    )

    machines_per_stage = dummy_instance.machines_per_stage
    num_stages = len(machines_per_stage)
    num_buffers = num_stages - 1

    # 下层 dummy env，用来拿 lower_obs_dim
    tmp_env = make_shop_env(
        instance=dummy_instance,
        buffers=[0] * num_buffers,
        reward_cfg=cfg.lower_reward_cfg,
        max_steps=cfg.lower_agent_cfg.max_steps_per_episode,
    )
    tmp_obs = tmp_env.reset()
    lower_obs_dim = tmp_obs.shape[0]

    # === 上层 obs_dim / action_dim：仅静态 BufferDesignEnv ===
    dummy_eval_fn = lambda inst, bufs: {"makespan": 0.0, "deadlock": False}
    dummy_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=0.0,
        randomize_instances=False,
        max_total_buffer=None,
    )
    dummy_env = BufferDesignEnv(
        instances=[dummy_instance],
        evaluate_fn=dummy_eval_fn,
        cfg=dummy_env_cfg,
        obs_cfg=None,
        seed=0,
    )
    dummy_obs = dummy_env.reset()
    upper_obs_dim = dummy_obs.shape[0]

    # 上层动作维度：a_k ∈ {0,...,max_a}
    max_a = 0
    for inst in train_instances:
        bounds = compute_buffer_upper_bounds(inst)
        if bounds:
            max_a = max(max_a, max(bounds))
    if max_a <= 0:
        max_a = 1
    upper_action_dim = max_a + 1
    print(f"[DEBUG] [STATIC] upper_action_dim={upper_action_dim}, max_a={max_a}")

    # 下层动作空间 = job 数量（动作：选 job）
    num_jobs = len(dummy_instance.jobs)
    lower_action_dim = num_jobs

    # === 把 obs_dim / action_dim 写回到 agent config 中 ===
    upper_agent_cfg = cfg.upper_agent_cfg
    upper_agent_cfg.obs_dim = upper_obs_dim
    upper_agent_cfg.action_dim = upper_action_dim

    lower_agent_cfg = cfg.lower_agent_cfg
    lower_agent_cfg.obs_dim = lower_obs_dim
    lower_agent_cfg.action_dim = lower_action_dim

    # === 构造上下两层 agent ===
    lower_agent = create_lower_agent(cfg.lower_agent_cfg, device)
    upper_agent = create_upper_agent(cfg.upper_agent_cfg, device)

    # === 静态上层 evaluate_fn ===
    evaluate_fn = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=cfg.lower_agent_cfg.num_inner_episodes,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode=getattr(cfg, "lower_train_mode", "da"),
        job_rule=getattr(cfg, "lower_job_rule", "spt"),   #"spt" / "fifo" / "lpt" / "srpt"
    )

    # === 构造上层 env，用于训练（静态 BufferDesignEnv） ===
    upper_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.upper_agent_cfg.buffer_cost_weight,
        deadlock_penalty=getattr(cfg.upper_agent_cfg, "deadlock_penalty", 1000.0),
        randomize_instances=True,
        max_total_buffer=None,
    )
    upper_env = BufferDesignEnv(
        instances=train_instances,
        evaluate_fn=evaluate_fn,  # instance, buffers -> metrics
        cfg=upper_env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )

    # === 训练日志 ===
    os.makedirs(out_dir, exist_ok=True)

    enable_train_log = getattr(cfg, "enable_train_log", True)
    log_file = None
    log_writer = None
    if enable_train_log:
        log_path = os.path.join(out_dir, "train_log.csv")
        log_file = open(log_path, "w", newline="")
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

    # 保存 cfg.json
    try:
        cfg_path = os.path.join(out_dir, "cfg.json")
        import dataclasses
        with open(cfg_path, "w", encoding="utf-8") as f_cfg:
            json.dump(dataclasses.asdict(cfg), f_cfg, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 保存 cfg.json 失败: {e}")

    best_val_tuple = None
    best_val_outer_ep = None

    try:
        for outer_ep in range(cfg.num_outer_episodes):
            # === 计算上层 epsilon ===
            if isinstance(upper_agent, UpperAgent):
                upper_agent.global_episode = outer_ep
                epsilon_U = upper_agent.compute_epsilon()
            else:
                upper_agent["global_episode"] = outer_ep
                epsilon_U = compute_upper_epsilon(upper_agent)

            # === 单个上层 episode ===
            obs_U = upper_env.reset()
            done_U = False
            ep_reward_U = 0.0
            traj: List[Tuple[np.ndarray, int, float, np.ndarray]] = []

            while not done_U:
                # 上层 epsilon-greedy
                if isinstance(upper_agent, UpperAgent):
                    action_U = upper_agent.select_action(obs_U, epsilon_U)
                else:
                    cfg_U: UpperAgentConfig = upper_agent["cfg"]
                    q_net_U: DQNNet = upper_agent["q_net"]
                    action_dim_U = cfg_U.action_dim
                    if np.random.rand() < epsilon_U:
                        action_U = int(np.random.randint(0, action_dim_U))
                    else:
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs_U).float().unsqueeze(0).to(device)
                            q_values = q_net_U(obs_tensor)[0].cpu().numpy()
                        action_U = int(q_values.argmax())

                next_obs_U, reward_U, done_U, info_U = upper_env.step(action_U)
                traj.append((obs_U, action_U, reward_U, next_obs_U))
                obs_U = next_obs_U
                ep_reward_U += reward_U

            metrics_U = info_U.get("metrics", {})
            final_makespan = float(metrics_U.get("makespan", 0.0))
            deadlock_flag = bool(metrics_U.get("deadlock", False))

            # === 回放上层轨迹，做 DQN 更新 ===
            for i, (s, a, r, s_next) in enumerate(traj):
                done_flag = (i == len(traj) - 1)
                if isinstance(upper_agent, UpperAgent):
                    upper_agent.add_transition(s, a, r, s_next, done_flag)
                else:
                    upper_agent["replay_buffer"].add(s, a, r, s_next, done_flag)

            num_upper_updates = 1
            for _ in range(num_upper_updates):
                if isinstance(upper_agent, UpperAgent):
                    upper_agent.update_one_step()
                else:
                    dqn_update_step(
                        upper_agent["q_net"],
                        upper_agent["target_q_net"],
                        upper_agent["optimizer"],
                        upper_agent["replay_buffer"],
                        cfg.upper_agent_cfg.batch_size,
                        cfg.upper_agent_cfg.gamma,
                        device,
                    )

            if outer_ep % cfg.upper_agent_cfg.target_update_interval == 0:
                if isinstance(upper_agent, UpperAgent):
                    upper_agent.update_target()
                else:
                    upper_agent["target_q_net"].load_state_dict(upper_agent["q_net"].state_dict())

            buffers = info_U.get("buffers", [])
            if buffers is None:
                total_buffer = 0.0
            else:
                try:
                    total_buffer = float(sum(buffers))
                except TypeError:
                    total_buffer = 0.0

            instance_idx = info_U.get("instance_idx", -1)

            if isinstance(buffers, (list, tuple)):
                buffer_str = " ".join(str(b) for b in buffers)
            elif buffers is not None:
                buffer_str = str(buffers)
            else:
                buffer_str = ""

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
                log_file.flush()

            print(
                f"[OuterEP {outer_ep:04d}] upper_ep_reward={ep_reward_U:.3f}, "
                f"final_makespan={final_makespan:.1f}, epsilon_U={epsilon_U:.3f}"
            )

            # === validation 评估（静态上层） ===
            eval_interval = getattr(cfg, "eval_interval", 0)
            if (eval_interval is not None) and (eval_interval > 0):
                if (outer_ep + 1) % eval_interval == 0:
                    val_metrics = evaluate_on_validation(
                        cfg=cfg,
                        upper_agent=upper_agent,
                        lower_agent=lower_agent,
                        val_instances=val_instances,
                        device=device,
                        out_dir=out_dir,
                        outer_ep=outer_ep + 1,
                        eval_num_instances=20,
                        eval_runs_per_instance=1,
                    )

                    if val_metrics is not None:
                        dead = float(val_metrics["deadlock_rate"])
                        avg_ms = float(val_metrics["avg_makespan"])
                        avg_buf = float(val_metrics["avg_total_buffer"])
                        candidate = (dead, avg_ms, avg_buf)

                        if (best_val_tuple is None) or (candidate < best_val_tuple):
                            best_val_tuple = candidate
                            best_val_outer_ep = outer_ep + 1
                            save_upper_agent_for_eval(upper_agent, Path(out_dir), tag="best_val")
                            print(
                                f"[CKPT] new best val at outer_ep={outer_ep + 1}, "
                                f"avg_ms={avg_ms:.3f}, deadlock={dead:.3f}, buf={avg_buf:.3f}"
                            )

        save_upper_agent_for_eval(upper_agent, Path(out_dir), tag="last")
        if best_val_tuple is not None and best_val_outer_ep is not None:
            print(
                f"[INFO] best val checkpoint at outer_ep={best_val_outer_ep}, "
                f"(deadlock_rate, avg_ms, avg_buf)={best_val_tuple}"
            )

        print("[INFO] Running offline buffer evaluation on VAL set...")
        offline_evaluate_all_buffers(
            cfg=cfg,
            lower_agent=lower_agent,
            val_instances=val_instances,
            device=device,
            out_dir=out_dir,
        )

    finally:
        if log_file is not None:
            log_file.close()


# ============================================================
# 0. Helper：加载固定算例 & 构造 j50s3m3 专用的 TwoLevelConfig
# ============================================================

def load_j50s3m3_instances() -> Tuple[List[InstanceData], List[InstanceData], List[InstanceData]]:
    experiment_name = "j50s3m3"
    data_root = os.path.join(ROOT_DIR, "experiments", "raw", experiment_name)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_instances = load_instances_from_dir(train_dir)
    val_instances = load_instances_from_dir(val_dir)
    test_instances = load_instances_from_dir(test_dir)

    print(
        f"[INFO] Loaded instances (j50s3m3): "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )
    return train_instances, val_instances, test_instances


def build_two_level_config_for_j50s3m3(
    algo_type: str,
    replay_type: str,
    seed: int,
    num_outer_episodes: int = 400,
    device: str = "cuda",
) -> TwoLevelConfig:
    """
    为 j50s3m3 的架构消融实验构造一个 TwoLevelConfig。
    """
    inst_cfg = FlowShopGeneratorConfig(
        num_jobs=50,
        num_stages=3,
        machines_per_stage=[3, 3, 3],
        proc_time_low=1,
        proc_time_high=30,
        same_proc_time_across_machines=True,
        seed=seed,
    )

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

    lower_agent_cfg = LowerAgentConfig(
        obs_dim=0,
        action_dim=0,
        num_inner_episodes=5,
    )

    upper_agent_cfg = UpperAgentConfig(
        obs_dim=0,
        action_dim=0,
        buffer_cost_weight=0.5,
        algo_type=algo_type,
        replay_type=replay_type,
    )

    cfg = TwoLevelConfig(
        instance_cfg=inst_cfg,
        buffer_upper_bounds=None,
        num_outer_episodes=num_outer_episodes,
        device=device,
        lower_agent_cfg=lower_agent_cfg,
        upper_agent_cfg=upper_agent_cfg,
        lower_reward_cfg=lower_reward_cfg,
        random_seed=seed,
    )

    cfg.experiment_name = "j50s3m3"
    cfg.algo_name = f"upper_{algo_type}_buffer_{replay_type}"

    cfg.eval_interval = 0
    cfg.enable_train_log = True

    # 默认下层做 DQN 调度（BA + DA）
    cfg.lower_train_mode = "da"

    return cfg


def build_standard_two_level_config_for_j50s3m3(
    seed: int,
    num_outer_episodes: int = 400,
    device: str = "cuda",
    buffer_cost_weight: float = 1.0,
    deadlock_penalty: float = 2000.0,
) -> TwoLevelConfig:
    """
    统一的“Standard-Upper-Agent + Standard-Hyperparams + Standard-Reward-Static” 配置。
    """
    cfg = build_two_level_config_for_j50s3m3(
        algo_type="d3qn",
        replay_type="uniform",
        seed=seed,
        num_outer_episodes=num_outer_episodes,
        device=device,
    )

    ua = cfg.upper_agent_cfg

    ua.gamma = 0.99
    ua.lr = 1e-4
    ua.batch_size = 128
    ua.buffer_capacity = 10_000
    ua.target_update_interval = 100
    ua.epsilon_start = 0.3
    ua.epsilon_end = 0.05
    ua.epsilon_decay_rate = 0.992
    if hasattr(ua, "hidden_dim"):
        ua.hidden_dim = 256

    ua.buffer_cost_weight = float(buffer_cost_weight)
    if hasattr(ua, "deadlock_penalty"):
        ua.deadlock_penalty = float(deadlock_penalty)

    cfg.num_outer_episodes = num_outer_episodes
    cfg.device = device
    cfg.experiment_name = "j50s3m3"
    cfg.algo_name = "upper_d3qn_uniform_standard"
    cfg.random_seed = seed

    cfg.eval_interval = 20
    cfg.enable_train_log = True

    # 标准配置默认也是 BA + DA
    cfg.lower_train_mode = "da"

    return cfg


if __name__ == "__main__":
    experiment_name = "j50s3m3"
    algo_name = "two_level_dqn"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT_DIR, "results", experiment_name, algo_name, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] results will be saved to: {out_dir}")

    data_root = os.path.join(ROOT_DIR, "experiments", "raw", experiment_name)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_instances = load_instances_from_dir(train_dir)
    val_instances = load_instances_from_dir(val_dir)
    test_instances = load_instances_from_dir(test_dir)

    print(
        f"[INFO] Loaded instances: "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )

    inst_cfg = FlowShopGeneratorConfig(
        num_jobs=5,
        num_stages=4,
        proc_time_low=1,
        proc_time_high=30,
        seed=0,
    )

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

    lower_agent_cfg = LowerAgentConfig(
        obs_dim=0,
        action_dim=0,
        num_inner_episodes=5,
    )

    upper_agent_cfg = UpperAgentConfig(
        obs_dim=0,
        action_dim=0,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_capacity=50_000,
        target_update_interval=200,
        buffer_cost_weight=0.5,
        algo_type="d3qn",
        replay_type="uniform",
    )

    cfg = TwoLevelConfig(
        instance_cfg=inst_cfg,
        num_outer_episodes=400,
        device="cuda",
        lower_agent_cfg=lower_agent_cfg,
        upper_agent_cfg=upper_agent_cfg,
        lower_reward_cfg=lower_reward_cfg,
        lower_train_mode="rule",   # 或 "rd" / "rule"
        random_seed=0,
    )

    train_two_level(cfg, train_instances, val_instances, test_instances, out_dir)

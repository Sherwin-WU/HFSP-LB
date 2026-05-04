# examples/group4_two_level.py
"""
Group4: Two-level BA+DA (Buffer Agent + D3QN Dispatch) 函数版

对外接口：

    run_group4_for_experiment(
        experiment_name: str,
        seeds: List[int],
        device_str: str = "cuda",
        method_name: str = "group4_two_level",
        num_outer_episodes: int = 400,
    )

功能：
  - 从 experiments/raw/<experiment_name>/{train,val,test} 加载实例
  - 对 seeds 中的每个 seed 做：
      * 上层缓冲 D3QN + 下层派工 D3QN 联合训练（two-level）
      * 训练结束后在 test 上 greedy 推理若干次
  - 每个 seed 的结果目录：
      results/<experiment_name>/<method_name>/seed{seed}_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import os
import sys
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- 路径设置：把 src/ 加进 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ---------- 导入项目内模块 ----------
from instances.types import InstanceData
from instances.io import load_instances_from_dir
from envs.ffs_core_env import FlowShopCoreEnv
from envs.shop_env import ShopEnv
from envs.observations.shop_obs_dense import build_shop_obs
from envs.reward import (
    ShopRewardConfig,
    make_shop_reward_fn,
    compute_shared_epi_reward,
)
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
# 默认超参数（组4用）
# ============================================================

# 训练轮数 & inner / eval 配置
DEFAULT_NUM_OUTER_EPISODES = 400
DEFAULT_NUM_INNER_EPISODES = 5
DEFAULT_MAX_LOWER_STEPS = 1000
DEFAULT_NUM_EVAL_EPISODES = 100

# reward / cost
DEFAULT_BUFFER_COST_WEIGHT = 1.0
DEFAULT_DEADLOCK_PENALTY = 2000.0

DEFAULT_LOWER_REWARD_SCHEME = "dense_epi"   # dense_epi / terminal_only / dense_only
DEFAULT_LOWER_EPI_REWARD_WEIGHT = 1.0

# 下层 D3QN
DEFAULT_LOWER_LR = 1e-3
DEFAULT_LOWER_GAMMA = 0.99
DEFAULT_LOWER_BATCH_SIZE = 128
DEFAULT_LOWER_BUFFER_CAPACITY = 10_000
DEFAULT_LOWER_TARGET_UPDATE_INTERVAL = 100

# 上层 D3QN
DEFAULT_UPPER_LR = 1e-4
DEFAULT_UPPER_GAMMA = 0.99
DEFAULT_UPPER_BATCH_SIZE = 128
DEFAULT_UPPER_BUFFER_CAPACITY = 10_000
DEFAULT_UPPER_TARGET_UPDATE_INTERVAL = 100

# epsilon 统一按 episode 衰减
DEFAULT_EPSILON_START = 0.3
DEFAULT_EPSILON_END = 0.05
DEFAULT_EPSILON_DECAY_RATE = 0.992


# ============================================================
# dataclass：下层配置 + 组4 TwoLevelConfig
# ============================================================

@dataclass
class LowerAgentConfig:
    obs_dim: int
    action_dim: int
    gamma: float = DEFAULT_LOWER_GAMMA
    lr: float = DEFAULT_LOWER_LR
    batch_size: int = DEFAULT_LOWER_BATCH_SIZE
    buffer_capacity: int = DEFAULT_LOWER_BUFFER_CAPACITY
    target_update_interval: int = DEFAULT_LOWER_TARGET_UPDATE_INTERVAL
    epsilon_start: float = DEFAULT_EPSILON_START
    epsilon_end: float = DEFAULT_EPSILON_END
    epsilon_decay_rate: float = DEFAULT_EPSILON_DECAY_RATE
    max_steps_per_episode: int = DEFAULT_MAX_LOWER_STEPS
    num_inner_episodes: int = DEFAULT_NUM_INNER_EPISODES


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

    buffer_cost_weight: float = DEFAULT_BUFFER_COST_WEIGHT
    deadlock_penalty: float = DEFAULT_DEADLOCK_PENALTY

    # ===== Phase A：LDA reward 结构开关 =====
    lower_reward_scheme: str = DEFAULT_LOWER_REWARD_SCHEME
    lower_epi_reward_weight: float = DEFAULT_LOWER_EPI_REWARD_WEIGHT

    # ===== 后续 runtime 统计会用到，先一并放进去 =====
    enable_train_log: bool = True
    enable_runtime_log: bool = True
    runtime_cuda_sync: bool = True


# ============================================================
# 加载算例
# ============================================================

def load_instances_for_experiment(
    experiment_name: str,
) -> Tuple[List[InstanceData], List[InstanceData], List[InstanceData]]:
    data_root = Path(ROOT_DIR) / "experiments" / "raw" / experiment_name
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_instances = load_instances_from_dir(str(train_dir))
    val_instances = load_instances_from_dir(str(val_dir)) if val_dir.exists() else []
    test_instances = load_instances_from_dir(str(test_dir)) if test_dir.exists() else []

    print(
        f"[DATA] Loaded instances ({experiment_name}): "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )
    return train_instances, val_instances, test_instances


# ============================================================
# 工具：计算某个实例原始下层 obs 维度（未 padding）
# ============================================================

def compute_raw_lower_obs_dim_for_instance(inst: InstanceData) -> int:
    """
    使用 FlowShopCoreEnv + build_shop_obs 计算该实例的原始观测维度。
    注意：必须先调用 core_env.reset()，否则 machines 等还没初始化。
    """
    num_buffers = max(0, len(inst.machines_per_stage) - 1)
    buffers = [0] * num_buffers
    core_env = FlowShopCoreEnv(instance=inst, buffers=buffers)
    core_env.reset()                       # ✅ 关键补这一句
    obs = build_shop_obs(core_env)
    return int(obs.shape[0])



# ============================================================
# 下层 env / agent / epsilon / action
# ============================================================

def make_shop_env(
    instance: InstanceData,
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    max_steps: int,
    obs_dim_target: Optional[int] = None,
) -> ShopEnv:
    """
    下层调度环境。若 obs_dim_target 不为 None，则对 build_shop_obs 的输出做
    padding / 截断到固定长度 obs_dim_target，以保证所有实例的 obs 维度一致。
    """
    core_env = FlowShopCoreEnv(
        instance=instance,
        buffers=buffers,
    )
    reward_fn = make_shop_reward_fn(reward_cfg)

    if obs_dim_target is None:
        def obs_builder(core_env: Any) -> np.ndarray:
            return build_shop_obs(core_env)
    else:
        def obs_builder(core_env: Any) -> np.ndarray:
            obs_raw = build_shop_obs(core_env)
            D = obs_raw.shape[0]
            if D == obs_dim_target:
                return obs_raw.astype(np.float32)
            elif D < obs_dim_target:
                pad = np.zeros((obs_dim_target - D,), dtype=np.float32)
                return np.concatenate([obs_raw, pad], axis=0)
            else:
                return obs_raw[:obs_dim_target].astype(np.float32)

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

    # 运行时检查：obs 维度必须和 agent.cfg.obs_dim 一致
    if obs.shape[0] != cfg.obs_dim:
        raise RuntimeError(
            f"[BUG] lower obs dim mismatch in epsilon-greedy: "
            f"got {obs.shape[0]}, expect {cfg.obs_dim}"
        )

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

    if obs.shape[0] != cfg.obs_dim:
        raise RuntimeError(
            f"[BUG] lower obs dim mismatch in greedy: "
            f"got {obs.shape[0]}, expect {cfg.obs_dim}"
        )

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

def normalize_lower_reward_scheme(name: str) -> str:
    """
    统一 reward scheme 名称，避免脚本侧写法不一致。
    """
    s = str(name).strip().lower()
    alias = {
        "dense+epi": "dense_epi",
        "dense_epi": "dense_epi",
        "terminal": "terminal_only",
        "terminal_only": "terminal_only",
        "dense": "dense_only",
        "dense_only": "dense_only",
    }
    s = alias.get(s, s)
    allowed = {"dense_epi", "terminal_only", "dense_only"}
    if s not in allowed:
        raise ValueError(
            f"Unknown lower_reward_scheme={name}, allowed={sorted(allowed)}"
        )
    return s


def clone_obs(obs: np.ndarray) -> np.ndarray:
    """
    防御性复制，避免后续环境内部复用同一块内存导致 replay 里的状态被污染。
    """
    return np.asarray(obs, dtype=np.float32).copy()


def postprocess_lower_traj_rewards(
    traj: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
    reward_scheme: str,
    epi_reward: float,
    epi_weight: float,
) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
    """
    对一整条 lower episode 轨迹做 reward 后处理。

    三种模式：
    1) dense_epi:
         r'_t = r_t^dense
         最后一步额外 + epi_weight * r_epi
    2) terminal_only:
         中间步全为 0
         最后一步 = epi_weight * r_epi
    3) dense_only:
         仅保留 dense reward，不加 episodic reward
    """
    scheme = normalize_lower_reward_scheme(reward_scheme)
    out: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    n = len(traj)
    for i, (obs, action, reward, next_obs, done) in enumerate(traj):
        r_new = float(reward)
        is_last = (i == n - 1)

        if scheme == "terminal_only":
            r_new = 0.0
            if is_last:
                r_new += float(epi_weight) * float(epi_reward)

        elif scheme == "dense_epi":
            if is_last:
                r_new += float(epi_weight) * float(epi_reward)

        elif scheme == "dense_only":
            pass

        out.append((obs, action, float(r_new), next_obs, done))

    return out


def maybe_cuda_sync(device: torch.device, enabled: bool) -> None:
    """
    后续 runtime 统计使用；Phase A 先放好，不影响当前逻辑。
    """
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)

def run_lower_episode_and_learn(
    instance: InstanceData,
    buffers: Sequence[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    lower_reward_scheme: str = DEFAULT_LOWER_REWARD_SCHEME,
    buffer_cost_weight: float = DEFAULT_BUFFER_COST_WEIGHT,
    deadlock_penalty: float = DEFAULT_DEADLOCK_PENALTY,
    lower_epi_reward_weight: float = DEFAULT_LOWER_EPI_REWARD_WEIGHT,
) -> Dict[str, Any]:
    """
    运行一条 lower episode，并在 episode 结束后统一做 reward 后处理。

    说明：
    - 之所以不再“边走边写 replay”，是因为 terminal_only / dense_epi 都依赖
      episode 结束后才知道的 shared episodic reward。
    - 为保证更新次数与原先量级一致，轨迹写入 replay 后，按 transition 数量做等量 update。
    """
    cfg: LowerAgentConfig = agent["cfg"]
    reward_scheme = normalize_lower_reward_scheme(lower_reward_scheme)

    env = make_shop_env(
        instance=instance,
        buffers=list(buffers),
        reward_cfg=reward_cfg,
        max_steps=cfg.max_steps_per_episode,
        obs_dim_target=cfg.obs_dim,
    )

    obs = env.reset()
    done = False
    steps = 0
    dense_reward_sum = 0.0
    last_info: Dict[str, Any] = {}

    # 先暂存整条轨迹，等 episode 结束后再统一改 reward
    traj: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    while (not done) and (steps < cfg.max_steps_per_episode):
        action = select_lower_action_epsilon_greedy(env, obs, agent, device)
        next_obs, reward, done, info = env.step(action)

        if next_obs.shape[0] != cfg.obs_dim:
            raise RuntimeError(
                f"[BUG] lower next_obs dim mismatch: got {next_obs.shape[0]}, expect {cfg.obs_dim}"
            )

        traj.append(
            (
                clone_obs(obs),
                int(action),
                float(reward),
                clone_obs(next_obs),
                bool(done),
            )
        )

        obs = next_obs
        steps += 1
        dense_reward_sum += float(reward)
        last_info = info

    makespan = float(last_info.get("makespan", cfg.max_steps_per_episode))
    deadlock_flag = bool(last_info.get("deadlock", False))

    # shared episodic reward：与论文 Table 3 / Eq.(16) 口径对齐
    epi_reward = compute_shared_epi_reward(
        makespan=makespan,
        buffers=buffers,
        deadlock=deadlock_flag,
        buffer_cost_weight=buffer_cost_weight,
        deadlock_penalty=deadlock_penalty,
    )

    processed_traj = postprocess_lower_traj_rewards(
        traj=traj,
        reward_scheme=reward_scheme,
        epi_reward=epi_reward,
        epi_weight=lower_epi_reward_weight,
    )

    effective_ep_reward = 0.0
    update_losses: List[float] = []

    # episode 结束后统一写入 replay，并做等量更新
    for obs_i, action_i, reward_i, next_obs_i, done_i in processed_traj:
        agent["replay_buffer"].add(obs_i, action_i, reward_i, next_obs_i, done_i)
        agent["global_step"] += 1
        effective_ep_reward += float(reward_i)

        loss_i = dqn_update_step(
            agent["q_net"],
            agent["target_q_net"],
            agent["optimizer"],
            agent["replay_buffer"],
            cfg.batch_size,
            cfg.gamma,
            device,
        )
        update_losses.append(float(loss_i))

        if agent["global_step"] % cfg.target_update_interval == 0:
            agent["target_q_net"].load_state_dict(agent["q_net"].state_dict())

    return dict(
        reward_scheme=reward_scheme,
        dense_reward_sum=float(dense_reward_sum),
        epi_reward=float(epi_reward),
        effective_ep_reward=float(effective_ep_reward),
        steps=steps,
        makespan=makespan,
        deadlock=deadlock_flag,
        avg_update_loss=float(np.mean(update_losses)) if update_losses else 0.0,
    )

def evaluate_lower_greedy(
    instance: InstanceData,
    buffers: List[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    cfg: LowerAgentConfig = agent["cfg"]

    env = make_shop_env(
        instance=instance,
        buffers=buffers,
        reward_cfg=reward_cfg,
        max_steps=max_steps,
        obs_dim_target=cfg.obs_dim,  # ⬅ 统一 obs 维度
    )
    obs = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while (not done) and (steps < max_steps):
        action = select_lower_action_greedy(env, obs, agent, device)
        obs, reward, done, info = env.step(action)

        if obs.shape[0] != cfg.obs_dim:
            raise RuntimeError(
                f"[BUG] lower obs dim mismatch in eval: got {obs.shape[0]}, expect {cfg.obs_dim}"
            )

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
    lower_reward_scheme: str = DEFAULT_LOWER_REWARD_SCHEME,
    buffer_cost_weight: float = DEFAULT_BUFFER_COST_WEIGHT,
    deadlock_penalty: float = DEFAULT_DEADLOCK_PENALTY,
    lower_epi_reward_weight: float = DEFAULT_LOWER_EPI_REWARD_WEIGHT,
):
    """
    lower_train_mode:
        - "da": 训练阶段，下层先 inner 训练 num_inner_episodes 次，再评估 greedy
        - "rd": 测试阶段，只评估 greedy（不更新下层）
    """
    mode = lower_train_mode.lower()

    def evaluate_fn(instance: InstanceData, buffers: Sequence[int]) -> Dict[str, float]:
        if mode == "da":
            for _ in range(num_inner_episodes):
                run_lower_episode_and_learn(
                    instance=instance,
                    buffers = list(buffers),
                    agent=lower_agent,
                    reward_cfg=lower_reward_cfg,
                    device=device,
                    lower_reward_scheme=lower_reward_scheme,
                    buffer_cost_weight=buffer_cost_weight,
                    deadlock_penalty=deadlock_penalty,
                    lower_epi_reward_weight=lower_epi_reward_weight,
                )

        eval_metrics = evaluate_lower_greedy(
            instance=instance,
            buffers = list(buffers),
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
    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES,
) -> None:
    if not test_instances:
        print("[TEST] No test instances, skip.")
        return

    out_dir = Path(out_dir)
    summary_path = out_dir / "eval_test_summary_detail.csv"
    detail_path = out_dir / "eval_test_detail.csv"

    eval_fn = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=0,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode="rd",
    )

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
        info_U: Dict[str, Any] = {}

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

    print(f"[SAVE] Test summary saved to {summary_path}")
    print(f"[SAVE] Test detail saved to {detail_path}")


# ============================================================
# 构造 config & 单个 seed 的训练 + eval
# ============================================================

def build_group4_config(
    experiment_name: str,
    method_name: str,
    seed: int,
    num_outer_episodes: int,
    device_str: str,
    lower_reward_scheme: str = DEFAULT_LOWER_REWARD_SCHEME,
    lower_epi_reward_weight: float = DEFAULT_LOWER_EPI_REWARD_WEIGHT,
) -> Group4TwoLevelConfig:
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
        num_inner_episodes=DEFAULT_NUM_INNER_EPISODES,
        max_steps_per_episode=DEFAULT_MAX_LOWER_STEPS,
    )

    upper_agent_cfg = UpperAgentConfig(
        obs_dim=0,
        action_dim=0,
        gamma=DEFAULT_UPPER_GAMMA,
        lr=DEFAULT_UPPER_LR,
        batch_size=DEFAULT_UPPER_BATCH_SIZE,
        buffer_capacity=DEFAULT_UPPER_BUFFER_CAPACITY,
        target_update_interval=DEFAULT_UPPER_TARGET_UPDATE_INTERVAL,
        buffer_cost_weight=DEFAULT_BUFFER_COST_WEIGHT,
        algo_type="d3qn",
        replay_type="uniform",
    )

    cfg = Group4TwoLevelConfig(
        experiment_name=experiment_name,
        method_name=method_name,
        num_outer_episodes=num_outer_episodes,
        device=device_str,
        random_seed=seed,
        lower_agent_cfg=lower_agent_cfg,
        upper_agent_cfg=upper_agent_cfg,
        lower_reward_cfg=lower_reward_cfg,
        buffer_cost_weight=DEFAULT_BUFFER_COST_WEIGHT,
        deadlock_penalty=DEFAULT_DEADLOCK_PENALTY,
        lower_reward_scheme=normalize_lower_reward_scheme(lower_reward_scheme),
        lower_epi_reward_weight=float(lower_epi_reward_weight),
        enable_train_log=True,
        enable_runtime_log=True,
        runtime_cuda_sync=True,
    )
    return cfg


def train_and_eval_one_seed(
    cfg: Group4TwoLevelConfig,
    train_instances: List[InstanceData],
    test_instances: List[InstanceData],
    base_out_dir: Path,
    skip_final_test_eval: bool = False,
    return_trained_objects: bool = False,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dummy_instance = train_instances[0]
    machines_per_stage = dummy_instance.machines_per_stage
    num_stages = len(machines_per_stage)
    num_buffers = num_stages - 1

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"seed{cfg.random_seed}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[RUN] experiment={cfg.experiment_name}, method={cfg.method_name}, "
        f"seed={cfg.random_seed}, out_dir={out_dir}"
    )
    print(f"[INFO] Using device: {device}")

    # ========= 1) 先确定下层 obs_dim / action_dim =========
    num_jobs = len(dummy_instance.jobs)
    lower_action_dim = num_jobs

    # 用 train + test（可选再加 val）所有实例的“原始 obs 维度”取最大值
    all_for_dim: List[InstanceData] = list(train_instances) + list(test_instances)
    if not all_for_dim:
        raise RuntimeError("No instances for determining lower_obs_dim.")

    raw_dims = [compute_raw_lower_obs_dim_for_instance(inst) for inst in all_for_dim]
    lower_obs_dim = max(raw_dims)

    cfg.lower_agent_cfg.obs_dim = lower_obs_dim
    cfg.lower_agent_cfg.action_dim = lower_action_dim

    # 构造下层 agent
    lower_agent = create_lower_agent(cfg.lower_agent_cfg, device)

    # ========= 2) 先确定上层 action_dim，再用“真实训练 env”拿 obs_dim =========
    # 上层动作：a_k ∈ {0,...,max_a}
    max_a = 0
    for inst in train_instances:
        bounds = compute_buffer_upper_bounds(inst)
        if bounds:
            max_a = max(max_a, max(bounds))
    if max_a <= 0:
        max_a = 1
    upper_action_dim = max_a + 1

    # 用和训练时完全一致的 BufferDesignEnv 配置，先建一个临时 env 拿 obs_dim
    upper_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=True,   # 和真正训练一致
        max_total_buffer=None,
    )

    # 训练阶段 evaluate_fn（BA+DA 联合），这里先给临时 env 用
    train_eval_fn_tmp = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=cfg.lower_agent_cfg.num_inner_episodes,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode="da",
        lower_reward_scheme=cfg.lower_reward_scheme,
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        lower_epi_reward_weight=cfg.lower_epi_reward_weight,
    )

    upper_env_tmp = BufferDesignEnv(
        instances=train_instances,
        evaluate_fn=train_eval_fn_tmp,
        cfg=upper_env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )
    obs_U0 = upper_env_tmp.reset()
    upper_obs_dim = obs_U0.shape[0]

    cfg.upper_agent_cfg.obs_dim = upper_obs_dim
    cfg.upper_agent_cfg.action_dim = upper_action_dim

    print(
        f"[INFO] lower_obs_dim={lower_obs_dim}, lower_action_dim={lower_action_dim}, "
        f"upper_obs_dim={upper_obs_dim}, upper_action_dim={upper_action_dim}"
    )

    # ========= 3) 构造上层 agent =========
    upper_agent: UpperAgent = create_upper_agent(cfg.upper_agent_cfg, device)

    # 重新构造一个“正式训练用”的 evaluate_fn 和 env（避免用刚才那个已经 reset 过的临时 env）
    train_eval_fn = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=cfg.lower_agent_cfg.num_inner_episodes,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode="da",
        lower_reward_scheme=cfg.lower_reward_scheme,
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        lower_epi_reward_weight=cfg.lower_epi_reward_weight,
    )

    upper_env = BufferDesignEnv(
        instances=train_instances,
        evaluate_fn=train_eval_fn,
        cfg=upper_env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )

    # ========= 4) 保存 cfg.json =========
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
                    lower_reward_scheme=cfg.lower_reward_scheme,
                    lower_epi_reward_weight=cfg.lower_epi_reward_weight,
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

    # ========= 5) 打开训练日志 =========
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

    # ========= 6) 训练主循环 =========
    try:
        for outer_ep in range(cfg.num_outer_episodes):
            upper_agent.global_episode = outer_ep
            epsilon_U = upper_agent.compute_epsilon()
            lower_agent["global_episode"] = outer_ep  # 下层 epsilon 同步

            obs_U = upper_env.reset()
            done_U = False
            info_U: Dict[str, Any] = {}
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

            # 回放到上层 buffer
            for i, (s, a, r, s_next) in enumerate(traj):
                done_flag = (i == len(traj) - 1)
                upper_agent.add_transition(s, a, r, s_next, done_flag)

            upper_agent.update_one_step()
            if (outer_ep + 1) % cfg.upper_agent_cfg.target_update_interval == 0:
                upper_agent.update_target()

            if isinstance(bufs, (list, tuple)):
                buffer_str = " ".join(str(b) for b in bufs)
            else:
                buffer_str = str(bufs)

            if log_writer is not None and log_file is not None:
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

        # 保存 upper q_net
        upper_ckpt_path = out_dir / "upper_q_last.pth"
        torch.save(upper_agent.q_net.state_dict(), upper_ckpt_path)
        print(f"[CKPT] Saved upper_agent q_net to {upper_ckpt_path}")

        # 保存 lower q_net（用于后续 two-level 继承推理）
        lower_ckpt_path = out_dir / "lower_q_last.pth"
        torch.save(lower_agent["q_net"].state_dict(), lower_ckpt_path)
        print(f"[CKPT] Saved lower_agent q_net to {lower_ckpt_path}")

        # 可选：同时保存 lower target_q_net，便于严格恢复/续训排查
        lower_target_ckpt_path = out_dir / "lower_target_q_last.pth"
        torch.save(lower_agent["target_q_net"].state_dict(), lower_target_ckpt_path)
        print(f"[CKPT] Saved lower_agent target_q_net to {lower_target_ckpt_path}")

    finally:
        if log_file is not None:
            log_file.close()

    # ========= 7) test eval =========
    # ========= 最终 test 评估 =========
    if not skip_final_test_eval:
        evaluate_group4_on_test(
            upper_agent=upper_agent,
            lower_agent=lower_agent,
            cfg=cfg,
            test_instances=test_instances,
            out_dir=out_dir,
            device=device,
            num_eval_episodes=DEFAULT_NUM_EVAL_EPISODES,
        )

    if return_trained_objects:
        return {
            "upper_agent": upper_agent,
            "lower_agent": lower_agent,
            "out_dir": out_dir,
            "cfg": cfg,
            "device": device,
        }

    return None


# ============================================================
# 对外接口：run_group4_for_experiment
# ============================================================

def run_group4_for_experiment(
    experiment_name: str,
    seeds: List[int],
    device_str: str = "cuda",
    method_name: str = "group4_two_level",
    num_outer_episodes: int = DEFAULT_NUM_OUTER_EPISODES,
    lower_reward_scheme: str = DEFAULT_LOWER_REWARD_SCHEME,
    lower_epi_reward_weight: float = DEFAULT_LOWER_EPI_REWARD_WEIGHT,
) -> None:
    """
    对一个算例 experiment_name，在 experiments/raw/<exp_name>/{train,test} 上
    依次对 seeds 中的每个 seed 训练 + 测试 group4 two-level。

    结果目录：
        results/<experiment_name>/<method_name>/seed{seed}_YYYYMMDD_HHMMSS/
    """
    train_instances, val_instances, test_instances = load_instances_for_experiment(
        experiment_name
    )
    if not train_instances:
        print(f"[ERROR] No train instances for {experiment_name}, skip Group4.")
        return
    if not test_instances:
        print(f"[WARN] No test instances for {experiment_name}, test eval will be skipped.")

    base_out_dir = Path(ROOT_DIR) / "results" / experiment_name / method_name
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        cfg = build_group4_config(
            experiment_name=experiment_name,
            method_name=method_name,
            seed=seed,
            num_outer_episodes=num_outer_episodes,
            device_str=device_str,
            lower_reward_scheme=lower_reward_scheme,
            lower_epi_reward_weight=lower_epi_reward_weight,
        )
        try:
            train_and_eval_one_seed(
                cfg=cfg,
                train_instances=train_instances,
                test_instances=test_instances,
                base_out_dir=base_out_dir,
            )
        except Exception as e:
            print(f"[ERROR] Group4 failed on experiment={experiment_name}, seed={seed}: {e}")

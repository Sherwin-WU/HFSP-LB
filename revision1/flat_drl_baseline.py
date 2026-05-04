# revision1/flat_drl_baseline.py
from __future__ import annotations

import os
import sys
import csv
import json
import math
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 路径设置
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(EXAMPLES_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# 项目内模块导入
# ============================================================

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
from utils.replay_buffer import ReplayBuffer
from group4_two_level import (
    load_instances_for_experiment,
    compute_raw_lower_obs_dim_for_instance,
    make_shop_env,
)


# ============================================================
# 默认常量
# ============================================================

DEFAULT_NUM_OUTER_EPISODES = 400
DEFAULT_NUM_EVAL_EPISODES = 100
DEFAULT_MAX_LOWER_STEPS = 1000

DEFAULT_BUFFER_COST_WEIGHT = 1.0
DEFAULT_DEADLOCK_PENALTY = 2000.0

DEFAULT_GAMMA = 0.99
DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 128
DEFAULT_REPLAY_CAPACITY = 50_000
DEFAULT_TARGET_UPDATE_INTERVAL = 200

DEFAULT_EPS_START = 0.7
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY = 0.992

DEFAULT_HIDDEN_DIM = 128

DEFAULT_DISPATCH_REWARD_SCHEME = "dense_epi"
DEFAULT_DISPATCH_EPI_REWARD_WEIGHT = 1.0


# ============================================================
# 小工具
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).copy()


def normalize_reward_scheme(name: str) -> str:
    s = str(name).strip().lower()
    alias = {
        "dense+epi": "dense_epi",
        "dense_epi": "dense_epi",
        "dense": "dense_only",
        "dense_only": "dense_only",
        "terminal": "terminal_only",
        "terminal_only": "terminal_only",
    }
    s = alias.get(s, s)
    allowed = {"dense_epi", "dense_only", "terminal_only"}
    if s not in allowed:
        raise ValueError(f"Unknown reward scheme: {name}, allowed={sorted(allowed)}")
    return s


def compute_shared_epi_reward(
    makespan: float,
    buffers: Sequence[int],
    deadlock: bool,
    buffer_cost_weight: float,
    deadlock_penalty: float,
) -> float:
    total_buffer = float(sum(int(b) for b in buffers)) if buffers is not None else 0.0
    deadlock_cost = float(deadlock_penalty) if bool(deadlock) else 0.0
    return - float(makespan) - float(buffer_cost_weight) * total_buffer - deadlock_cost


def postprocess_dispatch_traj_rewards(
    traj: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
    reward_scheme: str,
    epi_reward: float,
    epi_weight: float,
) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
    scheme = normalize_reward_scheme(reward_scheme)
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


def pad_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    dim = int(obs.shape[0])
    if dim == target_dim:
        return obs.copy()
    if dim < target_dim:
        pad = np.zeros((target_dim - dim,), dtype=np.float32)
        return np.concatenate([obs, pad], axis=0)
    return obs[:target_dim].copy()


def add_mode_flag(obs: np.ndarray, is_buffer_mode: bool) -> np.ndarray:
    mode = np.array([1.0, 0.0], dtype=np.float32) if is_buffer_mode else np.array([0.0, 1.0], dtype=np.float32)
    return np.concatenate([obs, mode], axis=0)


def make_flat_obs(obs_raw: np.ndarray, base_obs_dim: int, is_buffer_mode: bool) -> np.ndarray:
    return add_mode_flag(pad_obs(obs_raw, base_obs_dim), is_buffer_mode=is_buffer_mode)


def maybe_cuda_sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


# ============================================================
# 配置
# ============================================================

@dataclass
class FlatDRLConfig:
    experiment_name: str
    method_name: str
    random_seed: int
    device: str

    num_outer_episodes: int = DEFAULT_NUM_OUTER_EPISODES
    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    max_lower_steps: int = DEFAULT_MAX_LOWER_STEPS

    gamma: float = DEFAULT_GAMMA
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    target_update_interval: int = DEFAULT_TARGET_UPDATE_INTERVAL
    hidden_dim: int = DEFAULT_HIDDEN_DIM

    epsilon_start: float = DEFAULT_EPS_START
    epsilon_end: float = DEFAULT_EPS_END
    epsilon_decay_rate: float = DEFAULT_EPS_DECAY

    dispatch_reward_scheme: str = DEFAULT_DISPATCH_REWARD_SCHEME
    dispatch_epi_reward_weight: float = DEFAULT_DISPATCH_EPI_REWARD_WEIGHT
    buffer_cost_weight: float = DEFAULT_BUFFER_COST_WEIGHT
    deadlock_penalty: float = DEFAULT_DEADLOCK_PENALTY

    enable_train_log: bool = True
    enable_runtime_log: bool = True
    runtime_cuda_sync: bool = True

    # 下面这些在运行时自动探测
    upper_obs_dim: int = 0
    lower_obs_dim: int = 0
    base_obs_dim: int = 0
    flat_input_dim: int = 0
    upper_action_dim: int = 0
    lower_action_dim: int = 0

    # 下层 dense reward 配置
    lower_reward_cfg: Optional[ShopRewardConfig] = None


# ============================================================
# 网络：共享 trunk + two heads
# ============================================================

class FlatDuelingQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        buffer_action_dim: int,
        dispatch_action_dim: int,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.buffer_value = nn.Linear(hidden_dim, 1)
        self.buffer_adv = nn.Linear(hidden_dim, buffer_action_dim)

        self.dispatch_value = nn.Linear(hidden_dim, 1)
        self.dispatch_adv = nn.Linear(hidden_dim, dispatch_action_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def q_buffer(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs)
        v = self.buffer_value(z)
        a = self.buffer_adv(z)
        return v + a - a.mean(dim=1, keepdim=True)

    def q_dispatch(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs)
        v = self.dispatch_value(z)
        a = self.dispatch_adv(z)
        return v + a - a.mean(dim=1, keepdim=True)


# ============================================================
# Agent：单智能体，两个 replay
# ============================================================

class FlatDRLAgent:
    def __init__(self, cfg: FlatDRLConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        self.q_net = FlatDuelingQNet(
            input_dim=cfg.flat_input_dim,
            hidden_dim=cfg.hidden_dim,
            buffer_action_dim=cfg.upper_action_dim,
            dispatch_action_dim=cfg.lower_action_dim,
        ).to(device)

        self.target_q_net = FlatDuelingQNet(
            input_dim=cfg.flat_input_dim,
            hidden_dim=cfg.hidden_dim,
            buffer_action_dim=cfg.upper_action_dim,
            dispatch_action_dim=cfg.lower_action_dim,
        ).to(device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        self.upper_replay = ReplayBuffer(cfg.replay_capacity, cfg.flat_input_dim)
        self.lower_replay = ReplayBuffer(cfg.replay_capacity, cfg.flat_input_dim)

        self.global_step = 0
        self.global_episode = 0

    def compute_epsilon(self) -> float:
        eps = self.cfg.epsilon_start * (self.cfg.epsilon_decay_rate ** self.global_episode)
        if eps < self.cfg.epsilon_end:
            eps = self.cfg.epsilon_end
        return float(eps)

    def select_buffer_action(
        self,
        obs: np.ndarray,
        epsilon: float,
        greedy: bool = False,
    ) -> int:
        if (not greedy) and (np.random.rand() < epsilon):
            return int(np.random.randint(0, self.cfg.upper_action_dim))

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.q_net.q_buffer(obs_t)[0].cpu().numpy()
        return int(q.argmax())

    def select_dispatch_action(
        self,
        env: ShopEnv,
        obs: np.ndarray,
        epsilon: float,
        greedy: bool = False,
    ) -> int:
        legal_actions = env._core_env.get_legal_actions()
        if not legal_actions:
            return int(np.random.randint(0, self.cfg.lower_action_dim))

        if (not greedy) and (np.random.rand() < epsilon):
            return int(np.random.choice(legal_actions))

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.q_net.q_dispatch(obs_t)[0].cpu().numpy()

        mask = np.ones(self.cfg.lower_action_dim, dtype=bool)
        mask[legal_actions] = False
        q[mask] = -1e9
        return int(q.argmax())

    def add_upper_transition(self, obs, action, reward, next_obs, done) -> None:
        self.upper_replay.add(obs, action, reward, next_obs, done)

    def add_lower_transition(self, obs, action, reward, next_obs, done) -> None:
        self.lower_replay.add(obs, action, reward, next_obs, done)

    def update_one_step(self, mode: str) -> float:
        mode = str(mode).strip().lower()
        if mode not in {"upper", "lower"}:
            raise ValueError(f"Unknown mode={mode}")

        replay = self.upper_replay if mode == "upper" else self.lower_replay
        if replay.size < self.cfg.batch_size:
            return 0.0

        batch = replay.sample(self.cfg.batch_size)

        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)

        if mode == "upper":
            q_all = self.q_net.q_buffer(obs)
            q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = self.q_net.q_buffer(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_q_net.q_buffer(next_obs).gather(1, next_actions).squeeze(1)

        else:
            q_all = self.q_net.q_dispatch(obs)
            q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = self.q_net.q_dispatch(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_q_net.q_dispatch(next_obs).gather(1, next_actions).squeeze(1)

        td_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        loss = nn.functional.mse_loss(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.cfg.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


# ============================================================
# 环境相关辅助
# ============================================================

def build_temp_upper_env(
    instances: Sequence[InstanceData],
    seed: int,
    buffer_cost_weight: float,
    deadlock_penalty: float,
) -> BufferDesignEnv:
    def _dummy_eval_fn(instance: InstanceData, buffers: Sequence[int]) -> Dict[str, float]:
        return {
            "makespan": 0.0,
            "deadlock": False,
        }

    env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=buffer_cost_weight,
        deadlock_penalty=deadlock_penalty,
        randomize_instances=True,
        max_total_buffer=None,
    )

    env = BufferDesignEnv(
        instances=list(instances),
        evaluate_fn=_dummy_eval_fn,
        cfg=env_cfg,
        obs_cfg=None,
        seed=seed,
        custom_reward_fn=None,
    )
    return env


def infer_flat_dims(
    cfg: FlatDRLConfig,
    train_instances: List[InstanceData],
    test_instances: List[InstanceData],
) -> FlatDRLConfig:
    if not train_instances:
        raise RuntimeError("No train instances found.")

    dummy_instance = train_instances[0]
    num_jobs = len(dummy_instance.jobs)
    lower_action_dim = num_jobs

    all_for_dim = list(train_instances) + list(test_instances)
    raw_lower_dims = [compute_raw_lower_obs_dim_for_instance(inst) for inst in all_for_dim]
    lower_obs_dim = max(raw_lower_dims)

    temp_upper_env = build_temp_upper_env(
        instances=train_instances,
        seed=cfg.random_seed,
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
    )
    upper_obs0 = temp_upper_env.reset()
    upper_obs_dim = int(upper_obs0.shape[0])

    max_a = 0
    for inst in train_instances:
        bounds = compute_buffer_upper_bounds(inst)
        if bounds:
            max_a = max(max_a, max(bounds))
    if max_a <= 0:
        max_a = 1
    upper_action_dim = int(max_a + 1)

    cfg.lower_obs_dim = int(lower_obs_dim)
    cfg.upper_obs_dim = int(upper_obs_dim)
    cfg.base_obs_dim = int(max(lower_obs_dim, upper_obs_dim))
    cfg.flat_input_dim = int(cfg.base_obs_dim + 2)
    cfg.lower_action_dim = int(lower_action_dim)
    cfg.upper_action_dim = int(upper_action_dim)

    return cfg


# ============================================================
# Phase A：buffer phase
# ============================================================

def run_buffer_phase(
    instance: InstanceData,
    agent: FlatDRLAgent,
    device: torch.device,
    cfg: FlatDRLConfig,
    epsilon: float,
    greedy: bool = False,
) -> Dict[str, Any]:
    def _dummy_eval_fn(inst: InstanceData, buffers: Sequence[int]) -> Dict[str, float]:
        return {
            "makespan": 0.0,
            "deadlock": False,
        }

    env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=False,
        max_total_buffer=None,
    )

    env = BufferDesignEnv(
        instances=[instance],
        evaluate_fn=_dummy_eval_fn,
        cfg=env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )

    obs_raw = env.reset()
    obs = make_flat_obs(obs_raw, cfg.base_obs_dim, is_buffer_mode=True)
    done = False
    steps = 0
    traj: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    info_U: Dict[str, Any] = {}

    while not done:
        action = agent.select_buffer_action(obs, epsilon=epsilon, greedy=greedy)
        next_obs_raw, reward, done, info_U = env.step(action)
        next_obs = make_flat_obs(next_obs_raw, cfg.base_obs_dim, is_buffer_mode=True)

        traj.append((clone_obs(obs), int(action), float(reward), clone_obs(next_obs), bool(done)))
        obs = next_obs
        steps += 1

    metrics = info_U.get("metrics", {})
    buffers = info_U.get("buffers", metrics.get("buffers", []))
    if not isinstance(buffers, (list, tuple)):
        buffers = list(buffers) if buffers is not None else []

    total_buffer = float(sum(int(b) for b in buffers)) if buffers else 0.0

    return {
        "traj": traj,
        "buffers": list(buffers),
        "total_buffer": total_buffer,
        "upper_steps": steps,
    }


# ============================================================
# Phase B：dispatch phase
# ============================================================

def run_dispatch_phase(
    instance: InstanceData,
    buffers: Sequence[int],
    agent: FlatDRLAgent,
    device: torch.device,
    cfg: FlatDRLConfig,
    epsilon: float,
    greedy: bool = False,
) -> Dict[str, Any]:
    if cfg.lower_reward_cfg is None:
        raise RuntimeError("cfg.lower_reward_cfg is None.")

    env = make_shop_env(
        instance=instance,
        buffers=list(buffers),
        reward_cfg=cfg.lower_reward_cfg,
        max_steps=cfg.max_lower_steps,
        obs_dim_target=cfg.lower_obs_dim,
    )

    obs_raw = env.reset()
    obs = make_flat_obs(obs_raw, cfg.base_obs_dim, is_buffer_mode=False)

    done = False
    steps = 0
    dense_reward_sum = 0.0
    last_info: Dict[str, Any] = {}
    traj_raw: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    while (not done) and (steps < cfg.max_lower_steps):
        action = agent.select_dispatch_action(env, obs, epsilon=epsilon, greedy=greedy)
        next_obs_raw, reward, done, info = env.step(action)

        next_obs = make_flat_obs(next_obs_raw, cfg.base_obs_dim, is_buffer_mode=False)
        traj_raw.append((clone_obs(obs), int(action), float(reward), clone_obs(next_obs), bool(done)))

        obs = next_obs
        steps += 1
        dense_reward_sum += float(reward)
        last_info = info

    makespan = float(last_info.get("makespan", cfg.max_lower_steps))
    deadlock_flag = bool(last_info.get("deadlock", False))

    epi_reward = compute_shared_epi_reward(
        makespan=makespan,
        buffers=buffers,
        deadlock=deadlock_flag,
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
    )

    processed_traj = postprocess_dispatch_traj_rewards(
        traj=traj_raw,
        reward_scheme=cfg.dispatch_reward_scheme,
        epi_reward=epi_reward,
        epi_weight=cfg.dispatch_epi_reward_weight,
    )

    effective_ep_reward = float(sum(r for _, _, r, _, _ in processed_traj))

    return {
        "traj": processed_traj,
        "dense_reward_sum": float(dense_reward_sum),
        "epi_reward": float(epi_reward),
        "effective_ep_reward": float(effective_ep_reward),
        "lower_steps": steps,
        "makespan": makespan,
        "deadlock": deadlock_flag,
    }


# ============================================================
# 训练：单个 seed
# ============================================================

def evaluate_flat_drl_on_test(
    cfg: FlatDRLConfig,
    agent: FlatDRLAgent,
    test_instances: List[InstanceData],
    device: torch.device,
    out_dir: Path,
) -> None:
    if not test_instances:
        print("[TEST] No test instances, skip.")
        return

    summary_path = out_dir / "eval_test_summary.csv"
    detail_path = out_dir / "eval_test_detail.csv"

    rng = np.random.RandomState(cfg.random_seed + 999)

    total_ms = 0.0
    total_dead = 0
    total_buffers: List[float] = []
    details: List[Dict[str, Any]] = []

    t0 = time.perf_counter()

    for ep in range(cfg.num_eval_episodes):
        inst_idx = int(rng.randint(len(test_instances)))
        inst = test_instances[inst_idx]

        buffer_res = run_buffer_phase(
            instance=inst,
            agent=agent,
            device=device,
            cfg=cfg,
            epsilon=0.0,
            greedy=True,
        )

        dispatch_res = run_dispatch_phase(
            instance=inst,
            buffers=buffer_res["buffers"],
            agent=agent,
            device=device,
            cfg=cfg,
            epsilon=0.0,
            greedy=True,
        )

        ms = float(dispatch_res["makespan"])
        dl = bool(dispatch_res["deadlock"])
        total_buffer = float(buffer_res["total_buffer"])

        total_ms += ms
        total_dead += 1 if dl else 0
        total_buffers.append(total_buffer)

        buf_str = " ".join(str(b) for b in buffer_res["buffers"])

        details.append(
            {
                "episode": ep + 1,
                "instance_idx": inst_idx,
                "buffers": buf_str,
                "total_buffer": total_buffer,
                "makespan": ms,
                "deadlock": int(dl),
            }
        )

    maybe_cuda_sync(device, cfg.runtime_cuda_sync)
    t1 = time.perf_counter()

    episodes = len(details)
    avg_ms = total_ms / max(1, episodes)
    dead_rate = total_dead / max(1, episodes)
    avg_total_buffer = float(np.mean(total_buffers)) if total_buffers else math.nan
    std_total_buffer = float(np.std(total_buffers)) if total_buffers else math.nan

    with summary_path.open("w", newline="", encoding="utf-8") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            [
                "num_eval_episodes",
                "avg_makespan",
                "deadlock_rate",
                "num_deadlocks",
                "avg_total_buffer",
                "std_total_buffer",
                "ckpt",
            ]
        )
        writer.writerow(
            [
                int(episodes),
                float(avg_ms),
                float(dead_rate),
                int(total_dead),
                float(avg_total_buffer),
                float(std_total_buffer),
                "flat_q_last.pth",
            ]
        )

    with detail_path.open("w", newline="", encoding="utf-8") as f_det:
        writer = csv.writer(f_det)
        writer.writerow(
            [
                "episode",
                "instance_idx",
                "buffers",
                "total_buffer",
                "makespan",
                "deadlock",
            ]
        )
        for row in details:
            writer.writerow(
                [
                    int(row["episode"]),
                    int(row["instance_idx"]),
                    row["buffers"],
                    float(row["total_buffer"]),
                    float(row["makespan"]),
                    int(row["deadlock"]),
                ]
            )

    if cfg.enable_runtime_log:
        runtime_eval_path = out_dir / "runtime_eval.csv"
        with runtime_eval_path.open("w", newline="", encoding="utf-8") as f_rt:
            writer = csv.writer(f_rt)
            writer.writerow(["eval_wall_clock_sec"])
            writer.writerow([float(t1 - t0)])

    print(
        f"[Eval][TEST] episodes={episodes}, avg_makespan={avg_ms:.3f}, "
        f"deadlock_rate={dead_rate:.3f}, avg_total_buffer={avg_total_buffer:.3f}"
    )
    print(f"[SAVE] Test summary -> {summary_path}")
    print(f"[SAVE] Test detail  -> {detail_path}")


def train_and_eval_one_seed(
    cfg: FlatDRLConfig,
    base_out_dir: Path,
    skip_final_test_eval: bool = False,
    return_trained_objects: bool = False,
):
    set_global_seed(cfg.random_seed)

    train_instances, val_instances, test_instances = load_instances_for_experiment(cfg.experiment_name)
    _ = val_instances  # 当前版本先不用

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # reward config：先沿用当前主线的 progress shaping
    if cfg.lower_reward_cfg is None:
        cfg.lower_reward_cfg = ShopRewardConfig(mode="progress")

    cfg = infer_flat_dims(cfg, train_instances, test_instances)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"seed{cfg.random_seed}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[RUN] experiment={cfg.experiment_name}, method={cfg.method_name}, "
        f"seed={cfg.random_seed}, out_dir={out_dir}"
    )
    print(f"[INFO] device={device}")
    print(
        f"[INFO] upper_obs_dim={cfg.upper_obs_dim}, lower_obs_dim={cfg.lower_obs_dim}, "
        f"base_obs_dim={cfg.base_obs_dim}, flat_input_dim={cfg.flat_input_dim}, "
        f"upper_action_dim={cfg.upper_action_dim}, lower_action_dim={cfg.lower_action_dim}"
    )

    agent = FlatDRLAgent(cfg, device)

    cfg_json_path = out_dir / "cfg.json"
    with cfg_json_path.open("w", encoding="utf-8") as f_cfg:
        json.dump(
            {
                "experiment_name": cfg.experiment_name,
                "method_name": cfg.method_name,
                "random_seed": cfg.random_seed,
                "num_outer_episodes": cfg.num_outer_episodes,
                "num_eval_episodes": cfg.num_eval_episodes,
                "max_lower_steps": cfg.max_lower_steps,
                "gamma": cfg.gamma,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "replay_capacity": cfg.replay_capacity,
                "target_update_interval": cfg.target_update_interval,
                "hidden_dim": cfg.hidden_dim,
                "epsilon_start": cfg.epsilon_start,
                "epsilon_end": cfg.epsilon_end,
                "epsilon_decay_rate": cfg.epsilon_decay_rate,
                "dispatch_reward_scheme": cfg.dispatch_reward_scheme,
                "dispatch_epi_reward_weight": cfg.dispatch_epi_reward_weight,
                "buffer_cost_weight": cfg.buffer_cost_weight,
                "deadlock_penalty": cfg.deadlock_penalty,
                "upper_obs_dim": cfg.upper_obs_dim,
                "lower_obs_dim": cfg.lower_obs_dim,
                "base_obs_dim": cfg.base_obs_dim,
                "flat_input_dim": cfg.flat_input_dim,
                "upper_action_dim": cfg.upper_action_dim,
                "lower_action_dim": cfg.lower_action_dim,
                "lower_reward_cfg": cfg.lower_reward_cfg.__dict__ if cfg.lower_reward_cfg is not None else None,
            },
            f_cfg,
            indent=2,
            ensure_ascii=False,
        )

    log_file = None
    log_writer = None
    if cfg.enable_train_log:
        log_path = out_dir / "train_log.csv"
        log_file = log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "outer_episode",
                "instance_idx",
                "upper_steps",
                "lower_steps",
                "upper_ep_reward",
                "dispatch_dense_reward_sum",
                "dispatch_epi_reward",
                "dispatch_effective_ep_reward",
                "final_makespan",
                "deadlock",
                "total_buffer",
                "avg_upper_loss",
                "avg_lower_loss",
                "epsilon",
                "buffers",
                "wall_clock_sec",
            ]
        )
        log_file.flush()

    train_t0 = time.perf_counter()
    rng = np.random.RandomState(cfg.random_seed + 123)

    try:
        for outer_ep in range(cfg.num_outer_episodes):
            ep_t0 = time.perf_counter()

            agent.global_episode = outer_ep
            epsilon = agent.compute_epsilon()

            inst_idx = int(rng.randint(len(train_instances)))
            inst = train_instances[inst_idx]

            # 1) buffer phase
            buffer_res = run_buffer_phase(
                instance=inst,
                agent=agent,
                device=device,
                cfg=cfg,
                epsilon=epsilon,
                greedy=False,
            )

            # 2) dispatch phase
            dispatch_res = run_dispatch_phase(
                instance=inst,
                buffers=buffer_res["buffers"],
                agent=agent,
                device=device,
                cfg=cfg,
                epsilon=epsilon,
                greedy=False,
            )

            # 3) 用真实 episodic reward 覆盖上层 reward
            upper_traj = buffer_res["traj"]
            shared_epi_reward = float(dispatch_res["epi_reward"])
            processed_upper_traj: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

            for i, (obs, action, reward, next_obs, done) in enumerate(upper_traj):
                r_new = 0.0
                if i == len(upper_traj) - 1:
                    r_new = shared_epi_reward
                processed_upper_traj.append((obs, action, r_new, next_obs, done))

            # 4) 入 replay
            for obs, action, reward, next_obs, done in processed_upper_traj:
                agent.add_upper_transition(obs, action, reward, next_obs, done)

            for obs, action, reward, next_obs, done in dispatch_res["traj"]:
                agent.add_lower_transition(obs, action, reward, next_obs, done)

            # 5) 更新
            upper_losses: List[float] = []
            lower_losses: List[float] = []

            n_upper_updates = max(1, int(buffer_res["upper_steps"]))
            n_lower_updates = max(1, int(dispatch_res["lower_steps"]))

            for _ in range(n_upper_updates):
                loss_u = agent.update_one_step(mode="upper")
                if loss_u > 0:
                    upper_losses.append(float(loss_u))

            for _ in range(n_lower_updates):
                loss_l = agent.update_one_step(mode="lower")
                if loss_l > 0:
                    lower_losses.append(float(loss_l))

            total_buffer = float(buffer_res["total_buffer"])
            deadlock_flag = bool(dispatch_res["deadlock"])
            makespan = float(dispatch_res["makespan"])
            buffer_str = " ".join(str(b) for b in buffer_res["buffers"])

            maybe_cuda_sync(device, cfg.runtime_cuda_sync)
            ep_t1 = time.perf_counter()

            if log_writer is not None and log_file is not None:
                log_writer.writerow(
                    [
                        outer_ep,
                        inst_idx,
                        int(buffer_res["upper_steps"]),
                        int(dispatch_res["lower_steps"]),
                        float(shared_epi_reward),
                        float(dispatch_res["dense_reward_sum"]),
                        float(dispatch_res["epi_reward"]),
                        float(dispatch_res["effective_ep_reward"]),
                        float(makespan),
                        int(deadlock_flag),
                        float(total_buffer),
                        float(np.mean(upper_losses)) if upper_losses else 0.0,
                        float(np.mean(lower_losses)) if lower_losses else 0.0,
                        float(epsilon),
                        buffer_str,
                        float(ep_t1 - ep_t0),
                    ]
                )
                if (outer_ep + 1) % 50 == 0:
                    log_file.flush()

            if (outer_ep + 1) % 50 == 0 or outer_ep == 0:
                print(
                    f"[OuterEP {outer_ep+1:04d}] "
                    f"makespan={makespan:.3f}, deadlock={int(deadlock_flag)}, "
                    f"total_buffer={total_buffer:.1f}, epsilon={epsilon:.3f}"
                )

        ckpt_path = out_dir / "flat_q_last.pth"
        torch.save(agent.q_net.state_dict(), ckpt_path)
        print(f"[CKPT] Saved flat agent q_net -> {ckpt_path}")

    finally:
        if log_file is not None:
            log_file.close()

    maybe_cuda_sync(device, cfg.runtime_cuda_sync)
    train_t1 = time.perf_counter()

    if cfg.enable_runtime_log:
        runtime_train_path = out_dir / "runtime_train.json"
        with runtime_train_path.open("w", encoding="utf-8") as f_rt:
            json.dump(
                {
                    "train_wall_clock_sec": float(train_t1 - train_t0),
                    "num_outer_episodes": int(cfg.num_outer_episodes),
                },
                f_rt,
                indent=2,
                ensure_ascii=False,
            )

    if not skip_final_test_eval:
        evaluate_flat_drl_on_test(
            cfg=cfg,
            agent=agent,
            test_instances=test_instances,
            device=device,
            out_dir=out_dir,
        )

    if return_trained_objects:
        return {
            "agent": agent,
            "out_dir": out_dir,
            "cfg": cfg,
            "device": device,
            "test_instances": test_instances,
        }

    return None


# ============================================================
# 构造 config & 对外接口
# ============================================================

def build_flat_drl_config(
    experiment_name: str,
    method_name: str,
    seed: int,
    num_outer_episodes: int,
    device_str: str,
    dispatch_reward_scheme: str = DEFAULT_DISPATCH_REWARD_SCHEME,
    dispatch_epi_reward_weight: float = DEFAULT_DISPATCH_EPI_REWARD_WEIGHT,
) -> FlatDRLConfig:
    return FlatDRLConfig(
        experiment_name=experiment_name,
        method_name=method_name,
        random_seed=seed,
        device=device_str,
        num_outer_episodes=int(num_outer_episodes),
        num_eval_episodes=DEFAULT_NUM_EVAL_EPISODES,
        max_lower_steps=DEFAULT_MAX_LOWER_STEPS,
        gamma=DEFAULT_GAMMA,
        lr=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE,
        replay_capacity=DEFAULT_REPLAY_CAPACITY,
        target_update_interval=DEFAULT_TARGET_UPDATE_INTERVAL,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        epsilon_start=DEFAULT_EPS_START,
        epsilon_end=DEFAULT_EPS_END,
        epsilon_decay_rate=DEFAULT_EPS_DECAY,
        dispatch_reward_scheme=normalize_reward_scheme(dispatch_reward_scheme),
        dispatch_epi_reward_weight=float(dispatch_epi_reward_weight),
        buffer_cost_weight=DEFAULT_BUFFER_COST_WEIGHT,
        deadlock_penalty=DEFAULT_DEADLOCK_PENALTY,
        lower_reward_cfg=ShopRewardConfig(mode="progress"),
        enable_train_log=True,
        enable_runtime_log=True,
        runtime_cuda_sync=True,
    )


def run_flat_drl_for_experiment(
    experiment_name: str,
    seeds: List[int],
    device_str: str = "cuda",
    method_name: str = "flat_drl_single_agent",
    num_outer_episodes: int = DEFAULT_NUM_OUTER_EPISODES,
    dispatch_reward_scheme: str = DEFAULT_DISPATCH_REWARD_SCHEME,
    dispatch_epi_reward_weight: float = DEFAULT_DISPATCH_EPI_REWARD_WEIGHT,
) -> None:
    out_root = PROJECT_ROOT / "results" / experiment_name / method_name
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        cfg = build_flat_drl_config(
            experiment_name=experiment_name,
            method_name=method_name,
            seed=seed,
            num_outer_episodes=num_outer_episodes,
            device_str=device_str,
            dispatch_reward_scheme=dispatch_reward_scheme,
            dispatch_epi_reward_weight=dispatch_epi_reward_weight,
        )
        train_and_eval_one_seed(cfg, out_root)


# ============================================================
# 本文件自带一个最小冒烟入口
# ============================================================

if __name__ == "__main__":
    run_flat_drl_for_experiment(
        experiment_name="j50s3",
        seeds=[0],
        device_str="cuda",
        method_name="flat_drl_single_agent",
        num_outer_episodes=20,
        dispatch_reward_scheme="dense_epi",
        dispatch_epi_reward_weight=1.0,
    )
# examples/train_dispatch_d3qn_fixedbuf.py
"""
组3：固定缓存 + 下层 D3QN 派工 Agent

- 在给定 experiment_name 的 train 集上训练下层 D3QN 派工；
- 固定缓冲向量（默认 [3,3]，可通过命令行修改）；
- 定期在 val 上用贪婪策略评估，并根据 (deadlock_rate, avg_makespan) 选 best_val；
- 训练结束后在 test 集上评估 best_val（若无则 last）。

用法示例（在项目根目录）：

  # j50s3m3，固定 [3,3]，10 个种子
  python examples/train_dispatch_d3qn_fixedbuf.py \
      --experiment_name j50s3m3 \
      --fixed_buffers 3,3 \
      --seeds 0,1,2,3,4,5,6,7,8,9

"""

from __future__ import annotations

import os
import sys
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------- 路径设置：把 src/ 加进 sys.path ---------
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
from policies.dqn_common import dqn_style_update_step

import argparse


# ============================================================
# 1. Config 定义
# ============================================================

@dataclass
class DispatchAgentConfig:
    obs_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    buffer_capacity: int = 100_000
    target_update_interval: int = 500

    # epsilon(episode) = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_rate: float = 0.992

    hidden_dim: int = 256



@dataclass
class DispatchTrainConfig:
    experiment_name: str = "j50s3m3"
    device: str = "cuda"
    random_seed: int = 0

    fixed_buffers: List[int] | None = None

    reward_cfg: ShopRewardConfig | None = None

    num_episodes: int = 400
    max_steps_per_episode: int = 2_000

    eval_interval: int = 2_000
    log_interval: int = 100

    agent_cfg: DispatchAgentConfig | None = None


# ============================================================
# 2. 环境构造 & 工具函数
# ============================================================

def make_shop_env(
    instance: InstanceData,
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    max_steps: int,
    obs_dim_target: int | None = None,
) -> ShopEnv:
    """
    下层调度环境。若 obs_dim_target 不为 None，则对 build_shop_obs 的输出做
    padding / 截断到固定长度 obs_dim_target，以保证所有实例的 obs 维度一致。
    """
    core_env = FlowShopCoreEnv(instance=instance, buffers=buffers)
    reward_fn = make_shop_reward_fn(reward_cfg)

    if obs_dim_target is None:
        def obs_builder(core_env: FlowShopCoreEnv, cfg: Any | None = None) -> np.ndarray:
            return build_shop_obs(core_env, cfg=cfg)
    else:
        def obs_builder(core_env: FlowShopCoreEnv, cfg: Any | None = None) -> np.ndarray:
            obs_raw = build_shop_obs(core_env, cfg=cfg)
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



def load_instances_for_experiment(
    experiment_name: str,
) -> Tuple[List[InstanceData], List[InstanceData], List[InstanceData]]:
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

def compute_raw_obs_dim_for_instance(inst: InstanceData) -> int:
    """
    使用 FlowShopCoreEnv + build_shop_obs 计算该实例的原始观测维度。
    注意：必须先 reset，否则 core_env.machines 为空。
    """
    num_buffers = max(0, len(inst.machines_per_stage) - 1)
    buffers = [0] * num_buffers
    core_env = FlowShopCoreEnv(instance=inst, buffers=buffers)
    core_env.reset()
    obs = build_shop_obs(core_env)
    return int(obs.shape[0])

# ============================================================
# 3. Agent 构造 & 动作选择
# ============================================================

def create_dispatch_agent(cfg: DispatchAgentConfig, device: torch.device) -> Dict[str, Any]:
    q_net = DQNNet(cfg.obs_dim, cfg.action_dim).to(device)
    target_q_net = DQNNet(cfg.obs_dim, cfg.action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay_buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

    agent: Dict[str, Any] = dict(
        q_net=q_net,
        target_q_net=target_q_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        cfg=cfg,
        global_step=0,
    )
    return agent


def compute_epsilon(agent: Dict[str, Any]) -> float:
    """
    以 episode 为粒度的 epsilon 衰减：
      epsilon(ep) = max(epsilon_end, epsilon_start * (decay_rate ** ep))

    其中 ep = agent["global_episode"]，由训练循环在每个 episode 开头设置。
    """
    cfg: DispatchAgentConfig = agent["cfg"]
    ep = agent.get("global_episode", 0)
    eps = cfg.epsilon_start * (cfg.epsilon_decay_rate ** ep)
    if eps < cfg.epsilon_end:
        eps = cfg.epsilon_end
    return float(eps)



def select_action_epsilon_greedy(
    env: ShopEnv,
    obs: np.ndarray,
    agent: Dict[str, Any],
    device: torch.device,
) -> int:
    cfg: DispatchAgentConfig = agent["cfg"]
    q_net: nn.Module = agent["q_net"]
    epsilon = compute_epsilon(agent)
    action_dim = cfg.action_dim
    
    if obs.shape[0] != cfg.obs_dim:
        raise RuntimeError(
            f"[BUG] obs dim mismatch in select_action_epsilon_greedy: "
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


def select_action_greedy(
    env: ShopEnv,
    obs: np.ndarray,
    agent: Dict[str, Any],
    device: torch.device,
) -> int:
    cfg: DispatchAgentConfig = agent["cfg"]
    q_net: nn.Module = agent["q_net"]
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


# ============================================================
# 4. 单次 episode 训练 & 评估
# ============================================================

def run_one_episode_and_learn(
    instance: InstanceData,
    buffers: List[int],
    agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    cfg: DispatchAgentConfig = agent["cfg"]

    cfg_agent: DispatchAgentConfig = agent["cfg"]

    env = make_shop_env(
        instance=instance,
        buffers=buffers,
        reward_cfg=reward_cfg,
        max_steps=max_steps,
        obs_dim_target=cfg_agent.obs_dim,   # ★ 强制统一维度
    )
    obs = env.reset()
    if obs.shape[0] != cfg_agent.obs_dim:
        raise RuntimeError(
            f"[BUG] obs dim mismatch in run_one_episode_and_learn: "
            f"got {obs.shape[0]}, expect {cfg_agent.obs_dim}"
        )

    done = False
    steps = 0
    ep_reward = 0.0
    last_info: Dict[str, Any] = {}

    while (not done) and (steps < max_steps):
        action = select_action_epsilon_greedy(env, obs, agent, device)
        next_obs, reward, done, info = env.step(action)

        agent["replay_buffer"].add(obs, action, reward, next_obs, done)
        agent["global_step"] += 1

        obs = next_obs
        steps += 1
        ep_reward += float(reward)
        last_info = info

        # D3QN 更新
        loss = dqn_style_update_step(
            algo_type="d3qn",
            q_net=agent["q_net"],
            target_q_net=agent["target_q_net"],
            optimizer=agent["optimizer"],
            buffer=agent["replay_buffer"],
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            device=device,
        )

        # target 网络间隔式更新
        if agent["global_step"] % cfg.target_update_interval == 0:
            agent["target_q_net"].load_state_dict(agent["q_net"].state_dict())

    makespan = float(last_info.get("makespan", max_steps))
    deadlock = bool(last_info.get("deadlock", False))
    if (not last_info) or (steps >= max_steps and "makespan" not in last_info):
        deadlock = True

    return dict(
        ep_reward=float(ep_reward),
        steps=int(steps),
        makespan=makespan,
        deadlock=deadlock,
    )


def evaluate_dispatch_on_instances(
    agent: Dict[str, Any],
    instances: List[InstanceData],
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
) -> Dict[str, float]:
    if not instances:
        return dict(avg_makespan=float("inf"), deadlock_rate=0.0, episodes=0)

    total_makespan = 0.0
    total_deadlock = 0
    episodes = 0
    cfg_agent: DispatchAgentConfig = agent["cfg"]  # ★ 从 agent 里拿到 DispatchAgentConfig
    for inst in instances:
        env = make_shop_env(
            instance=inst,
            buffers=buffers,
            reward_cfg=reward_cfg,
            max_steps=max_steps,
            obs_dim_target=cfg_agent.obs_dim,
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

    if episodes == 0:
        return dict(avg_makespan=float("inf"), deadlock_rate=0.0, episodes=0)

    avg_makespan = total_makespan / episodes
    deadlock_rate = total_deadlock / episodes
    return dict(
        avg_makespan=float(avg_makespan),
        deadlock_rate=float(deadlock_rate),
        episodes=int(episodes),
    )


# ============================================================
# 5. 训练主循环
# ============================================================

def train_dispatch_d3qn_fixedbuf(
    cfg: DispatchTrainConfig,
    train_instances: List[InstanceData],
    val_instances: List[InstanceData],
    test_instances: List[InstanceData],
    out_dir: str,
    skip_final_test_eval: bool = False,
    return_trained_objects: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # seed
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # buffers
    if cfg.fixed_buffers is None:
        raise ValueError("cfg.fixed_buffers must be set, e.g. [3,3]")
    if cfg.agent_cfg is None:
        raise ValueError("cfg.agent_cfg must be set.")
    if cfg.reward_cfg is None:
        raise ValueError("cfg.reward_cfg must be set.")

    fixed_buffers = list(cfg.fixed_buffers)
    agent_cfg = cfg.agent_cfg
    reward_cfg = cfg.reward_cfg

    # 基于第一个 train instance 确定 obs_dim / action_dim，并 sanity check buffers 长度
    if not train_instances:
        raise ValueError("No train instances loaded.")
    # === 统一 obs 维度：用 train+val+test 所有实例的原始维度最大值 ===
    all_for_dim: List[InstanceData] = list(train_instances) + list(val_instances) + list(test_instances)
    if not all_for_dim:
        raise RuntimeError("No instances available to determine obs_dim.")
    
    raw_dims = [compute_raw_obs_dim_for_instance(inst) for inst in all_for_dim]
    obs_dim = max(raw_dims)
    
    dummy_instance = train_instances[0]
    num_jobs = len(dummy_instance.jobs)
    action_dim = num_jobs
    
    agent_cfg.obs_dim = obs_dim
    agent_cfg.action_dim = action_dim
    
    print(
        f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, "
        f"fixed_buffers={cfg.fixed_buffers}"
    )


    # 构造 agent
    agent = create_dispatch_agent(agent_cfg, device)

    # 日志文件
    log_path = os.path.join(out_dir, "train_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        [
            "episode",
            "instance_idx",
            "ep_reward",
            "makespan",
            "deadlock",
            "steps",
            "epsilon",
        ]
    )
    log_file.flush()

    # eval 日志
    eval_path = os.path.join(out_dir, "eval_val.csv")
    eval_file = open(eval_path, "w", newline="")
    eval_writer = csv.writer(eval_file)
    eval_writer.writerow(
        [
            "episode",
            "split",
            "avg_makespan",
            "deadlock_rate",
            "episodes",
        ]
    )
    eval_file.flush()

    # 保存 cfg.json
    try:
        cfg_json_path = os.path.join(out_dir, "cfg.json")
        with open(cfg_json_path, "w", encoding="utf-8") as f_cfg:
            json.dump(asdict(cfg), f_cfg, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved cfg.json to {cfg_json_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cfg.json: {e}")

    best_val_tuple = None
    best_val_episode = None

    try:
        for ep in range(cfg.num_episodes):
            # 1) 把当前 episode 写进 agent，供 compute_epsilon 使用
            agent["global_episode"] = ep

            # 2) 随机抽取一个 train instance
            inst_idx = np.random.randint(len(train_instances))
            inst = train_instances[inst_idx]

            metrics = run_one_episode_and_learn(
                instance=inst,
                buffers=fixed_buffers,
                agent=agent,
                reward_cfg=reward_cfg,
                device=device,
                max_steps=cfg.max_steps_per_episode,
            )

            epsilon = compute_epsilon(agent)
            if (ep + 1) % cfg.log_interval == 0 or ep == 0:
                print(
                    f"[EP {ep+1:06d}] inst_idx={inst_idx}, "
                    f"reward={metrics['ep_reward']:.3f}, "
                    f"makespan={metrics['makespan']:.1f}, "
                    f"deadlock={int(metrics['deadlock'])}, "
                    f"steps={metrics['steps']}, "
                    f"epsilon={epsilon:.3f}"
                )

            # 写 train_log
            log_writer.writerow(
                [
                    ep + 1,
                    inst_idx,
                    float(metrics["ep_reward"]),
                    float(metrics["makespan"]),
                    int(metrics["deadlock"]),
                    int(metrics["steps"]),
                    float(epsilon),
                ]
            )
            if (ep + 1) % cfg.log_interval == 0:
                log_file.flush()

            # # 定期在 val 上做评估
            # if cfg.eval_interval > 0 and (ep + 1) % cfg.eval_interval == 0 and val_instances:
            #     val_metrics = evaluate_dispatch_on_instances(
            #         agent=agent,
            #         instances=val_instances,
            #         buffers=fixed_buffers,
            #         reward_cfg=cfg.reward_cfg,
            #         device=device,
            #         max_steps=cfg.max_steps_per_episode,
            #     )
            #     eval_writer.writerow(
            #         [
            #             ep + 1,
            #             "val",
            #             float(val_metrics["avg_makespan"]),
            #             float(val_metrics["deadlock_rate"]),
            #             int(val_metrics["episodes"]),
            #         ]
            #     )
            #     eval_file.flush()

            #     print(
            #         f"[Eval][EP={ep+1}] split=val, "
            #         f"avg_makespan={val_metrics['avg_makespan']:.3f}, "
            #         f"deadlock_rate={val_metrics['deadlock_rate']:.3f}, "
            #         f"episodes={val_metrics['episodes']}"
            #     )

            #     # 以 (deadlock_rate, avg_makespan) 作为排序 key
            #     candidate = (
            #         float(val_metrics["deadlock_rate"]),
            #         float(val_metrics["avg_makespan"]),
            #     )
            #     if (best_val_tuple is None) or (candidate < best_val_tuple):
            #         best_val_tuple = candidate
            #         best_val_episode = ep + 1
            #         ckpt_path = os.path.join(out_dir, "dispatch_q_best_val.pth")
            #         torch.save(agent["q_net"].state_dict(), ckpt_path)
            #         print(
            #             f"[CKPT] New best_val at EP={ep+1}, "
            #             f"avg_ms={candidate[1]:.3f}, dead={candidate[0]:.3f}, "
            #             f"saved to {ckpt_path}"
            #         )

        # 训练结束后保存 last
        last_ckpt_path = os.path.join(out_dir, "dispatch_q_last.pth")
        torch.save(agent["q_net"].state_dict(), last_ckpt_path)
        print(f"[CKPT] Saved last checkpoint to {last_ckpt_path}")

    finally:
        log_file.close()
        eval_file.close()

    # ============================
    # 6. 在 test 集上做最终 100 次推理评估（greedy）
    # ============================
        # =========================
    # 训练结束：在 test 集上评估
    # =========================
    if not skip_final_test_eval:
        if test_instances:
            print("[TEST] Final greedy evaluation on test set ...")
            test_metrics = run_greedy_eval_episodes(
                agent=agent,
                instances=test_instances,
                buffers=fixed_buffers,
                reward_cfg=reward_cfg,
                device=device,
                max_steps=cfg.max_steps_per_episode,
                num_eval_episodes=100,
            )

            test_csv_path = os.path.join(out_dir, "eval_test.csv")
            with open(test_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["avg_makespan", "deadlock_rate", "episodes"])
                writer.writerow(
                    [
                        test_metrics["avg_makespan"],
                        test_metrics["deadlock_rate"],
                        test_metrics["episodes"],
                    ]
                )

            print(
                f"[TEST] avg_makespan={test_metrics['avg_makespan']:.4f}, "
                f"deadlock_rate={test_metrics['deadlock_rate']:.4f}, "
                f"episodes={test_metrics['episodes']}"
            )
        else:
            print("[TEST] No test instances, skip final evaluation.")

    if return_trained_objects:
        return {
            "agent": agent,
            "out_dir": out_dir,
            "cfg": cfg,
            "device": device,
            "fixed_buffers": fixed_buffers,
        }

    return None
        # 明细
        # detail_path = os.path.join(out_dir, "eval_test_detail.csv")
        # with open(detail_path, "w", newline="") as f_det:
        #     writer = csv.writer(f_det)
        #     writer.writerow(["episode", "instance_idx", "makespan", "deadlock", "steps"])
        #     for row in test_metrics["details"]:
        #         writer.writerow([
        #             row["episode"],
        #             row["instance_idx"],
        #             row["makespan"],
        #             row["deadlock"],
        #             row["steps"],
        #         ])


def run_greedy_eval_episodes(
    agent: Dict[str, Any],
    instances: List[InstanceData],
    buffers: List[int],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
    num_eval_episodes: int = 100,
) -> Dict[str, Any]:
    """
    从给定 instances 中随机采样，每次用 greedy 策略跑一集，总共 num_episodes 次。
    返回整体的平均 makespan / deadlock_rate。
    """

    cfg_agent = agent["cfg"]  # ★ 从 agent 里拿到 DispatchAgentConfig
    if (not instances) or num_eval_episodes <= 0:
        return dict(avg_makespan=float("inf"), deadlock_rate=0.0, episodes=0)

    total_makespan = 0.0
    total_deadlock = 0
    episodes = 0

    details = []  # 每一局一条 dict

    for ep in range(num_eval_episodes):
        inst_idx = np.random.randint(len(instances))
        inst = instances[inst_idx]

        env = make_shop_env(
            instance=inst,
            buffers=buffers,
            reward_cfg=reward_cfg,
            max_steps=max_steps,
            obs_dim_target=cfg_agent.obs_dim,  # ★ 用上面取出来的 cfg_agent
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

        # ★ 新增：把这一局的结果放到 details 里
        details.append(
            dict(
                episode=ep + 1,
                instance_idx=inst_idx,
                makespan=makespan,
                deadlock=int(deadlock),
                steps=steps,
            )
        )

        total_makespan += makespan
        total_deadlock += 1 if deadlock else 0
        episodes += 1

    avg_makespan = total_makespan / episodes
    deadlock_rate = total_deadlock / episodes
    return dict(
        avg_makespan=float(avg_makespan),
        deadlock_rate=float(deadlock_rate),
        episodes=int(episodes),
        details=details,
    )

# ============================================================
# 7. CLI & main
# ============================================================

def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train dispatch D3QN agent with fixed buffers on flow shop instances."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="j50s3m3",
        help="Experiment name, corresponds to experiments/raw/<name>/ directories.",
    )
    parser.add_argument(
        "--fixed_buffers",
        type=str,
        default="3,3",
        help="Comma-separated buffer capacities per buffer segment. Default: 3,3",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated random seeds. Default: 0..9",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device: "cuda" or "cpu". Default: cuda',
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=400,
        help="Number of training episodes. Default: 200000",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=2_000,
        help="Max steps per episode. Default: 2000",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=2_000,
        help="Evaluate on val set every N episodes. Default: 2000",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print & flush train log every N episodes. Default: 100",
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name
    fixed_buffers = parse_int_list(args.fixed_buffers)
    seeds = [int(s) for s in parse_int_list(args.seeds)]
    device = args.device

    print(f"[INFO] experiment_name={experiment_name}")
    print(f"[INFO] fixed_buffers={fixed_buffers}")
    print(f"[INFO] seeds={seeds}")
    print(f"[INFO] device={device}")

    # 加载实例
    train_instances, val_instances, test_instances = load_instances_for_experiment(experiment_name)

    if not train_instances:
        raise RuntimeError("No train instances loaded; please check experiments/raw/<exp>/train/")

    # base config
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
    base_agent_cfg = DispatchAgentConfig(obs_dim=0, action_dim=0)

    for seed in seeds:
        cfg = DispatchTrainConfig(
            experiment_name=experiment_name,
            device=device,
            random_seed=seed,
            fixed_buffers=fixed_buffers,
            reward_cfg=base_reward_cfg,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
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
            f"\n[RUN] experiment={experiment_name}, seed={seed}, "
            f"out_dir={out_dir}"
        )

        train_dispatch_d3qn_fixedbuf(
            cfg=cfg,
            train_instances=train_instances,
            val_instances=val_instances,
            test_instances=test_instances,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()

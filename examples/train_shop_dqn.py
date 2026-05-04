# 文件：examples/train_shop_dqn.py
# 作用：
#   在固定缓冲配置下（例：[1,2,2]），
#   使用 FlowShopCoreEnv + ShopEnv + DQN 训练下层调度策略。

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ----------------------------
# 把 src 加入 sys.path，方便导入 envs / instances
# ----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# === 实例相关 ===
from instances.generators import FlowShopGeneratorConfig, generate_random_instance

# === 下层 env / obs / reward ===
from envs.ffs_core_env import FlowShopCoreEnv
from envs.shop_env import ShopEnv
from envs.observations.shop_obs_dense import build_shop_obs, ShopObsConfig
from envs.reward import ShopRewardConfig, make_shop_reward_fn


# ============================================================
# 1. DQN 组件：网络 / replay buffer / 配置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch_obs = torch.from_numpy(self.obs_buf[idx])
        batch_next_obs = torch.from_numpy(self.next_obs_buf[idx])
        batch_actions = torch.from_numpy(self.actions[idx])
        batch_rewards = torch.from_numpy(self.rewards[idx])
        batch_dones = torch.from_numpy(self.dones[idx])
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones


@dataclass
class TrainConfig:
    num_episodes: int = 300
    max_steps_per_episode: int = 2000

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 50_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    seed: int = 0


# ============================================================
# 2. 准备实例 + env 构建函数
# ============================================================

def make_toy_instance(seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = FlowShopGeneratorConfig(
        num_jobs=5,
        num_stages=4,
        machines_per_stage=[2, 2, 2, 2],
        proc_time_low=1,
        proc_time_high=10,
        seed=seed,
    )
    inst = generate_random_instance(cfg, rng)
    return inst


def make_shop_env(instance, buffers, seed: int = 0, max_steps: int = 2000) -> ShopEnv:
    core_env = FlowShopCoreEnv(
        instance=instance,
        buffers=buffers,
        dispatch_machine_rule="min_index",
        seed=seed,
    )

    obs_cfg = ShopObsConfig()
    obs_builder = lambda core_state: build_shop_obs(core_state, obs_cfg)

    # 定义下层奖励：惩罚时间 + 阻塞 + 非法动作，不直接用 makespan
    reward_cfg = ShopRewardConfig(
        mode="progress",
        time_weight=1.0,           # -Δt
        makespan_weight=0.0,       # 我们不用额外 -makespan，改用 -Δt
        per_operation_reward=0.05,
        per_job_reward=0.1,
        blocking_penalty=0.2,
        terminal_bonus=0.5,
        invalid_action_weight=0.2,
    )
    reward_fn = make_shop_reward_fn(reward_cfg)

    env = ShopEnv(
        core_env=core_env,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        max_steps=max_steps,
    )
    return env


# ============================================================
# 3. epsilon 调度函数
# ============================================================

def compute_epsilon(cfg: TrainConfig, global_step: int) -> float:
    if global_step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = global_step / float(cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


# ============================================================
# 4. 主训练逻辑
# ============================================================

def train_shop_dqn():
    cfg = TrainConfig()

    # 固定随机种子
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 1) 准备一个算例 + 固定缓冲配置
    instance = make_toy_instance(seed=cfg.seed)
    # 这里直接固定 buffers，[1,2,2] 可以和你上面的测试保持一致
    buffers = [1, 2, 2]

    # 2) 构建 env，并用 reset() 探测 obs_dim / action_dim
    env = make_shop_env(instance, buffers, seed=cfg.seed, max_steps=cfg.max_steps_per_episode)
    obs = env.reset()
    obs_dim = obs.shape[0]

    # 动作数量 = 作业数量
    if hasattr(instance, "num_jobs"):
        action_dim = int(instance.num_jobs)
    else:
        action_dim = len(instance.jobs)

    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}")

    # 3) 初始化 DQN 网络与 target 网络
    q_net = DQNNetwork(obs_dim, action_dim).to(device)
    target_net = DQNNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)

    # 4) 初始化 replay buffer
    replay_buffer = ReplayBuffer(capacity=cfg.buffer_capacity, obs_dim=obs_dim)

    global_step = 0

    # 用于简单日志记录
    recent_rewards: Deque[float] = deque(maxlen=20)
    recent_makespans: Deque[float] = deque(maxlen=20)

    for ep in range(cfg.num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        last_info = {}

        while (not done) and (ep_steps < cfg.max_steps_per_episode):
            epsilon = compute_epsilon(cfg, global_step)

            # 1) 从 env 的核心环境拿“当前合法动作集合”
            legal_actions = env._core_env.get_legal_actions()

            # 如果当前没有任何合法动作，就只能随便选一个 job（或完全 no-op）
            if not legal_actions:
                action = int(np.random.randint(0, action_dim))
            else:
                if np.random.rand() < epsilon:
                    # 2) 探索：在合法集合里随机选一个
                    action = int(np.random.choice(legal_actions))
                else:
                    # 3) 利用：只在合法动作里选 argmax Q
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                        q_values = q_net(obs_tensor)[0].cpu().numpy()  # [action_dim]

                    # 构造掩码：默认全部非法，然后把 legal_actions 标记为合法
                    mask = np.ones(action_dim, dtype=bool)
                    mask[legal_actions] = False  # False 表示“不屏蔽”
                    q_values[mask] = -1e9       # 把非法动作的 Q 置成很小

                    action = int(q_values.argmax())

            next_obs, reward, done, info = env.step(action)

            replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            global_step += 1
            last_info = info

            # --- DQN 更新 ---
            if replay_buffer.size >= cfg.batch_size and global_step > cfg.learning_starts:
                if global_step % cfg.train_freq == 0:
                    batch = replay_buffer.sample(cfg.batch_size)
                    loss = dqn_update_step(
                        q_net=q_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        batch=batch,
                        gamma=cfg.gamma,
                    )

                # 定期更新 target 网络
                if global_step % cfg.target_update_interval == 0:
                    target_net.load_state_dict(q_net.state_dict())

        # 记录日志
        recent_rewards.append(ep_reward)
        makespan = float(last_info.get("makespan", 0.0))
        if makespan > 0:
            recent_makespans.append(makespan)

        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_makespan = np.mean(recent_makespans) if recent_makespans else 0.0

        print(
            f"[EP {ep:03d}] steps={ep_steps:4d} "
            f"reward={ep_reward:8.3f} (avg={avg_reward:8.3f}) "
            f"makespan={makespan:6.1f} (avg={avg_makespan:6.1f}) "
            f"epsilon={compute_epsilon(cfg, global_step):.3f}"
        )

    # 5) 训练完成后保存模型
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "shop_dqn.pt")
    torch.save(q_net.state_dict(), ckpt_path)
    print(f"[INFO] Saved DQN model to {ckpt_path}")

     # === 训练后评估 ===
    print("\n===== Post-training evaluation =====")
    evaluate_greedy_policy(instance, buffers, q_net, device, num_episodes=50)
    evaluate_random_policy(instance, buffers, num_episodes=50)

# ============================================================
# 4.5 通用评估函数
from typing import Callable  # 如果文件前面还没有导入，可以加上这一行
def run_eval_episode(
    env: ShopEnv,
    select_action: Callable[[np.ndarray], int],
    max_steps: int,
):
    """
    通用评估：给一个 env 和一个“选动作函数”，跑完整个 episode，
    返回 (episode_reward, makespan, finished, steps)

    - finished: True 表示所有工件完成且 env 在 info 里给了 makespan
    - makespan: 若 finished=False，则用核心环境当前时间或 max_steps 作为“伪 makespan”
    """
    obs = env.reset()
    done = False
    ep_reward = 0.0
    last_info = {}
    steps = 0

    while (not done) and (steps < max_steps):
        action = select_action(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        last_info = info
        steps += 1

    # 计算 makespan
    if "makespan" in last_info and last_info["makespan"] is not None:
        makespan = float(last_info["makespan"])
        finished = True
    else:
        finished = False
        # fallback：用核心环境当前时间或 max_steps 作为“伪 makespan”
        core_env = getattr(env, "_core_env", None)
        if (core_env is not None) and hasattr(core_env, "time"):
            makespan = float(core_env.time)
        else:
            makespan = float(max_steps)

    return ep_reward, makespan, finished, steps


def dqn_update_step(
    q_net: DQNNetwork,
    target_net: DQNNetwork,
    optimizer: optim.Optimizer,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
) -> float:
    """
    对一个 batch 的 (s,a,r,s',done) 做一次 DQN 更新，返回 loss 标量。
    """
    batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

    batch_obs = batch_obs.to(device)
    batch_actions = batch_actions.to(device)
    batch_rewards = batch_rewards.to(device)
    batch_next_obs = batch_next_obs.to(device)
    batch_dones = batch_dones.to(device)

    # Q(s,a) for taken actions
    q_values = q_net(batch_obs)  # [B, A]
    # gather 用 actions 索引对应列
    q_sa = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

    # target: r + gamma * (1-done) * max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_values = target_net(batch_next_obs)  # [B, A]
        max_next_q = next_q_values.max(dim=1).values
        targets = batch_rewards + gamma * (1.0 - batch_dones) * max_next_q

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_sa, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()

    return float(loss.item())

# ============================================================
# 5. 评估函数：贪心策略 / 随机策略
def evaluate_greedy_policy(
    instance,
    buffers,
    q_net: DQNNetwork,
    device: torch.device,
    num_episodes: int = 50,
    max_steps: int = 2000,
):
    """
    用“纯贪心 DQN 策略”（epsilon=0）评估若干局，
    计算平均 reward 和平均 makespan，并打印前几局的细节。
    """
    # 动作维度 = 作业数
    if hasattr(instance, "num_jobs"):
        action_dim = int(instance.num_jobs)
    else:
        action_dim = len(instance.jobs)

    total_reward = 0.0
    total_makespan = 0.0

    for ep in range(num_episodes):
        # 每个 episode 新建一个 env，避免状态脏掉
        env = make_shop_env(instance, buffers, seed=1000 + ep, max_steps=max_steps)

        # 定义贪心策略：给 obs -> argmax_a Q(s,a)
        def greedy_policy(obs_np: np.ndarray) -> int:
            # 评估时也要尊重当前合法动作集合
            legal_actions = env._core_env.get_legal_actions()
            if not legal_actions:
                # 理论上很少发生，发生时随便选一个 job
                return int(np.random.randint(0, action_dim))

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
                q_values = q_net(obs_tensor)[0].cpu().numpy()  # [action_dim]

            # 掩码非法动作
            mask = np.ones(action_dim, dtype=bool)
            mask[legal_actions] = False
            q_values[mask] = -1e9

            return int(q_values.argmax())


        ep_reward, makespan, finished, steps = run_eval_episode(
            env, greedy_policy, max_steps
        )

        total_reward += ep_reward
        total_makespan += makespan

        if ep < 5:  # 只打印前几局的调试信息
            num_finished_jobs = sum(
                1 for j in env._core_env.jobs
                if getattr(j, "finished", False)
            )
            total_jobs = len(env._core_env.jobs)
            print(
                f"[Eval Greedy][ep={ep}] finished={finished}, "
                f"jobs_done={num_finished_jobs}/{total_jobs}, "
                f"steps={steps}, makespan={makespan:.1f}, ep_reward={ep_reward:.1f}"
            )

    avg_reward = total_reward / num_episodes
    avg_makespan = total_makespan / num_episodes
    print(
        f"[Eval Greedy] episodes={num_episodes}, "
        f"avg_reward={avg_reward:.3f}, avg_makespan={avg_makespan:.3f}"
    )


# ============================================================
# 6. 评估函数：随机策略
def evaluate_random_policy(
    instance,
    buffers,
    num_episodes: int = 50,
    max_steps: int = 2000,
):
    """
    纯随机策略评估：每步 uniform 随机选一个 job index。
    用作 DQN 的 baseline。
    """
    if hasattr(instance, "num_jobs"):
        action_dim = int(instance.num_jobs)
    else:
        action_dim = len(instance.jobs)

    total_reward = 0.0
    total_makespan = 0.0

    for ep in range(num_episodes):
        env = make_shop_env(instance, buffers, seed=2000 + ep, max_steps=max_steps)

        def random_policy(obs_np: np.ndarray) -> int:
            legal_actions = env._core_env.get_legal_actions()
            if legal_actions:
                return int(np.random.choice(legal_actions))
            return int(np.random.randint(0, action_dim))


        ep_reward, makespan, finished, steps = run_eval_episode(
            env, random_policy, max_steps
        )

        total_reward += ep_reward
        total_makespan += makespan

        if ep < 3:
            num_finished_jobs = sum(
                1 for j in env._core_env.jobs
                if getattr(j, "finished", False)
            )
            total_jobs = len(env._core_env.jobs)
            print(
                f"[Eval Random][ep={ep}] finished={finished}, "
                f"jobs_done={num_finished_jobs}/{total_jobs}, "
                f"steps={steps}, makespan={makespan:.1f}, ep_reward={ep_reward:.1f}"
            )

    avg_reward = total_reward / num_episodes
    avg_makespan = total_makespan / num_episodes
    print(
        f"[Eval Random] episodes={num_episodes}, "
        f"avg_reward={avg_reward:.3f}, avg_makespan={avg_makespan:.3f}"
    )


if __name__ == "__main__":
    train_shop_dqn()

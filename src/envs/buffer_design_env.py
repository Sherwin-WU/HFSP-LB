# ============================
# 文件：envs/buffer_design_env.py
# 作用：
#   上层“缓存 agent”的环境：
#     - 一次 episode 中，逐段选择中间缓冲容量 b_k；
#     - 所有 b_k 选完后，调用 evaluate_fn(instance, B) 去跑下层调度，
#       得到 makespan 等指标，并计算一次终局奖励；
#     - 中间步骤 reward = 0，适合 DQN/DDQN/D3QN。
#
# 关键特性：
#   - 动作空间：当前阶段 i 本段缓冲容量 b_i 的整数候选（action 被视为 b_i，超界会被裁剪到 [0, a_i]）。
#   - 状态空间：通过 envs/observations/buffer_obs_dense.build_buffer_obs 构造，
#               包含实例特征 + 缓冲上界 + 已选 b + 当前阶段索引等信息。
#   - 奖励：默认 R = -(makespan + lambda * sum(b_k))，也支持自定义 reward_fn。
#
# 依赖：
#   - instances.types.InstanceData
#   - instances.features.build_instance_feature_vector（在 buffer_obs_dense 内部使用）
#   - envs.constraints.compute_buffer_upper_bounds / project_buffers_to_feasible
#   - envs.observations.buffer_obs_dense.build_buffer_obs / BufferObsConfig
#   - envs.base_env.BaseEnv
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base_env import BaseEnv
from instances.types import InstanceData
from .observations.buffer_obs_dense import build_buffer_obs, BufferObsConfig

# 优先尝试从 envs.constraints 导入上界/投影函数；
# 若用户暂未实现该模块，则使用本文件内的 fallback 实现。
try:
    from .constraints import compute_buffer_upper_bounds, project_buffers_to_feasible
except ImportError:  # pragma: no cover
    def compute_buffer_upper_bounds(instance: InstanceData) -> List[int]:
        """
        根据实例的机器数，给每个缓冲段一个上界：
            a_k = max{ m_k, m_{k+1} }.

        参数：
            instance: InstanceData，要求有 machines_per_stage 属性，
                    例如 [m_0, m_1, ..., m_{S-1}].

        返回：
            长度为 num_stages - 1 的列表 [a_0, ..., a_{S-2}].
        """
        mps = list(instance.machines_per_stage)
        num_stages = len(mps)

        if num_stages <= 1:
            # 只有一个 stage 或更少，就不存在中间缓冲
            return []

        upper_bounds: List[int] = []
        for k in range(num_stages - 1):
            a_k = max(int(mps[k]), int(mps[k + 1]))
            upper_bounds.append(a_k)

        return upper_bounds

    def project_buffers_to_feasible(
        b: Sequence[int],
        a: Sequence[int],
        total_limit: Optional[int] = None,
    ) -> List[int]:
        """
        fallback: 简单裁剪到 [0, a_k] 并可选加入总和约束。
        正式实现建议移到 envs/constraints.py。
        """
        assert len(b) == len(a)
        proj = [max(0, min(int(bk), int(ak))) for bk, ak in zip(b, a)]
        if total_limit is not None:
            s = sum(proj)
            if s > total_limit and s > 0:
                ratio = total_limit / s
                proj = [int(np.floor(bk * ratio)) for bk in proj]
        return proj


@dataclass
class BufferDesignEnvConfig:
    """
    BufferDesignEnv 的基本配置。

    字段：
        buffer_cost_weight (lambda):
            缓冲成本的权重 λ，若使用默认 reward，则：
                reward = -(makespan + λ * sum(b_k))
        randomize_instances:
            是否在 reset 时随机选择实例；若为 False，则按顺序循环。
        max_total_buffer (可选):
            若不为 None，则对 sum(b_k) 加一个上界，用于 project_buffers_to_feasible。
        deadlock_penalty:
            死锁惩罚（会加在 makespan 上），例如 1000.0。
    """
    # ★ 统一静态上层 reward 的默认设置（Standard-Reward-Static）
    buffer_cost_weight: float = 0.5
    randomize_instances: bool = True
    max_total_buffer: Optional[int] = None
    deadlock_penalty: float = 1000.0



class BufferDesignEnv(BaseEnv):
    """
    上层“缓存 agent”环境。

    用法示例（伪代码）::

        from envs.observations.buffer_obs_dense import BufferObsConfig, build_buffer_obs
        from envs.reward import BufferRewardConfig, make_buffer_reward_fn

        buf_obs_cfg = BufferObsConfig(...)
        buf_reward_cfg = BufferRewardConfig(makespan_weight=1.0, buffer_cost_weight=0.1)
        custom_reward_fn = make_buffer_reward_fn(buf_reward_cfg)

        env = BufferDesignEnv(
            instances=train_instances,
            evaluate_fn=my_eval_fn,  # InstanceData, List[int] -> metrics dict
            cfg=BufferDesignEnvConfig(buffer_cost_weight=buf_reward_cfg.buffer_cost_weight),
            obs_cfg=buf_obs_cfg,
            seed=42,
            custom_reward_fn=custom_reward_fn,
        )

        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)  # DQN 输出一个整数
            obs, reward, done, info = env.step(action)

    其中：
        - evaluate_fn(instance, buffers) 内部可用下层 AllocAgent + ShopEnv 去跑一轮完整调度，
          返回 metrics，例如 {"makespan": ..., "blocking_time": ..., ...}
        - 状态 obs 由 build_buffer_obs(self, obs_cfg) 构造。
    """

    def __init__(
        self,
        instances: Sequence[InstanceData],
        evaluate_fn: Callable[[InstanceData, Sequence[int]], Dict[str, float]],
        cfg: Optional[BufferDesignEnvConfig] = None,
        obs_cfg: Optional[BufferObsConfig] = None,
        seed: Optional[int] = None,
        custom_reward_fn: Optional[
            Callable[[Dict[str, Any], Sequence[int], Any], float]
        ] = None,
    ) -> None:
        """
        参数：
            instances:
                一组 InstanceData，假定它们的阶段数相同（即 num_stages 一致），
                因为上层 obs 维度会依赖 num_buffers = num_stages - 1。
            evaluate_fn:
                用户提供的评估函数：给定 (instance, buffers)，返回 metrics dict。
                metrics 至少应包含 "makespan" 键。
            cfg:
                BufferDesignEnvConfig，若为 None 则用默认配置。
            obs_cfg:
                BufferObsConfig，用于控制状态编码的细节；
                若为 None，则使用默认配置。
            seed:
                随机种子，用于实例采样顺序。
            custom_reward_fn:
                若提供，将用该函数计算终局 reward，格式：
                    reward = custom_reward_fn(metrics, buffers, cfg_like)
                否则使用默认：
                    reward = -(metrics["makespan"] + cfg.buffer_cost_weight * sum(buffers))

                注意：这里的第三个参数类型为 Any，以兼容不同的 reward 配置类型
                      （例如 envs.reward.BufferRewardConfig）。
        """
        super().__init__()
        assert len(instances) > 0, "BufferDesignEnv 需要至少一个实例"

        self._instances: List[InstanceData] = list(instances)
        self._evaluate_fn = evaluate_fn
        self._cfg = cfg or BufferDesignEnvConfig()
        self._obs_cfg = obs_cfg or BufferObsConfig()
        self._rng = np.random.default_rng(seed)
        self._custom_reward_fn = custom_reward_fn

        # 检查阶段数一致性
        first_jobs = self._instances[0].jobs
        if len(first_jobs) == 0:
            raise ValueError("实例中 jobs 为空，无法推断阶段数")
        num_stages_0 = len(first_jobs[0].ops)
        for inst in self._instances[1:]:
            if len(inst.jobs) == 0:
                raise ValueError("某个实例 jobs 为空，无法推断阶段数")
            if len(inst.jobs[0].ops) != num_stages_0:
                raise ValueError(
                    "BufferDesignEnv 要求所有实例的阶段数 num_stages 相同，"
                    f"发现 {num_stages_0} 和 {len(inst.jobs[0].ops)} 不一致"
                )

        self._num_stages: int = num_stages_0
        self._num_buffers: int = max(0, self._num_stages - 1)
        if self._num_buffers == 0:
            raise ValueError("阶段数 <= 1 时没有中间缓冲，BufferDesignEnv 没有意义")

        # 内部状态
        self._episode_index: int = 0          # 当前使用的是第几个实例
        self._current_instance: Optional[InstanceData] = None
        self._current_instance_idx: Optional[int] = None

        self._upper_bounds: Optional[np.ndarray] = None  # a_k
        self._buffers: Optional[np.ndarray] = None       # 当前 b_k（未选用 -1 表示）
        self._current_buf_index: int = 0                 # 当前要决策的段索引 i
        self._done: bool = False

    # ------------------------------------------------------------------
    #  BaseEnv 接口
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        采样一个新实例，计算各段缓冲上界 a_k，并清空已选缓冲 b_k。
        返回上层 agent 的观测向量 obs。
        """
        self._done = False

        # 选择实例
        if self._cfg.randomize_instances:
            idx = int(self._rng.integers(0, len(self._instances)))
        else:
            idx = self._episode_index % len(self._instances)
            self._episode_index += 1

        self._current_instance_idx = idx
        self._current_instance = self._instances[idx]

        # 计算上界 a_k
        a_list = compute_buffer_upper_bounds(self._current_instance)
        if len(a_list) != self._num_buffers:
            raise ValueError(
                f"compute_buffer_upper_bounds 返回长度 {len(a_list)}，"
                f"与 num_buffers={self._num_buffers} 不一致"
            )
        self._upper_bounds = np.asarray(a_list, dtype=np.int32)

        # 初始化 b_k 为 -1（表示尚未决策）
        self._buffers = np.full(shape=(self._num_buffers,), fill_value=-1, dtype=np.int32)
        self._current_buf_index = 0

        obs = self._build_obs()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        在当前缓冲段 i 上选择一个缓冲容量 b_i。

        参数：
            action:
                DQN 输出的整数，被解释为当前段的候选容量 b_i_raw，
                实际应用时会被裁剪到 [0, a_i]。

        返回：
            obs_next:
                下一时刻的观测向量；若本轮决策为最后一段，则为“终局状态”的观测。
            reward:
                若本次尚未完成全部缓冲决策，则 reward=0；
                若本次为最后一段，则 reward 为终局奖励。
            done:
                是否 episode 结束（所有缓冲段都已决策，并完成 evaluate_fn 调用）。
            info:
                附加信息字典，包括：
                    - "instance_idx": 当前实例索引
                    - "buffers":      当前 episode 的完整缓冲配置（若 done=True）
                    - "metrics":      evaluate_fn 返回的指标（若 done=True）
        """
        if self._done:
            raise RuntimeError("调用 step 之前需要先 reset：当前 episode 已经结束。")

        assert self._current_instance is not None
        assert self._upper_bounds is not None
        assert self._buffers is not None

        i = self._current_buf_index
        if i >= self._num_buffers:
            # 理论上不应发生
            raise RuntimeError("当前缓冲段索引越界，请检查逻辑。")

        # 1) 将 action 裁剪为当前段的合法 b_i
        a_i = int(self._upper_bounds[i])
        b_i_raw = int(action)
        b_i = max(0, min(b_i_raw, a_i))

        self._buffers[i] = b_i

        # 2) 判断是否已经选完所有缓冲段
        is_last = (i == self._num_buffers - 1)

        if not is_last:
            # 中间阶段：不触发调度仿真，reward=0，done=False
            self._current_buf_index += 1
            obs_next = self._build_obs()
            reward = 0.0
            done = False
            info: Dict[str, Any] = {
                "instance_idx": self._current_instance_idx,
                "buffers_partial": self._buffers.copy(),
                "buffer_index": i,
            }
            return obs_next, reward, done, info

        # 3) 最后一段：凑齐 B 后调用 evaluate_fn
        # 先投影到可行集合（考虑 [0,a_k] 和总上界）
        proj_buffers = project_buffers_to_feasible(
            b=self._buffers.tolist(),
            a=self._upper_bounds.tolist(),
            total_limit=self._cfg.max_total_buffer,
        )

        # 调用用户提供的评估函数，内部可用下层 ShopEnv + AllocAgent
        metrics = self._evaluate_fn(self._current_instance, proj_buffers)

        # 计算 reward
        reward = self._compute_final_reward(metrics, proj_buffers)

        self._done = True

        # 终局也构造一个 obs（通常 DQN 不太用得到，但保持接口完整）
        obs_next = self._build_obs()

        info = {
            "instance_idx": self._current_instance_idx,
            "buffers": proj_buffers,
            "metrics": metrics,
        }

        return obs_next, reward, True, info

    # ------------------------------------------------------------------
    #  内部工具函数
    # ------------------------------------------------------------------
    def _compute_final_reward(
        self,
        metrics: Dict[str, Any],
        buffers: Sequence[int],
    ) -> float:
        """
        计算终局 reward。

        若用户提供了 custom_reward_fn，则直接调用它：
            reward = custom_reward_fn(metrics, buffers, cfg_like)

        否则使用默认：
            reward = -(metrics["makespan"] + λ * sum(buffers))
        """
        if self._custom_reward_fn is not None:
            return float(self._custom_reward_fn(metrics, buffers, self._cfg))

        if "makespan" not in metrics:
            raise KeyError(
                "metrics 字典中缺少 'makespan'，默认 reward 计算需要该字段。"
            )

        makespan = float(metrics["makespan"])
        total_buffer = float(sum(buffers))
        lam = float(self._cfg.buffer_cost_weight)

        # 若 metrics 中带有 deadlock 标志，则在 makespan 上额外加一个较大的惩罚
        deadlock_flag = bool(metrics.get("deadlock", False))
        deadlock_penalty = float(getattr(self._cfg, "deadlock_penalty", 0.0))
        if deadlock_flag:
            makespan = makespan + deadlock_penalty

        return -(makespan + lam * total_buffer)

    def _build_obs(self) -> np.ndarray:
        """
        构造上层 agent 的观测向量。

        具体逻辑委托给 envs.observations.buffer_obs_dense.build_buffer_obs，
        保证与 BufferDesignEnv 内部状态结构保持一致。
        """
        return build_buffer_obs(self, self._obs_cfg)

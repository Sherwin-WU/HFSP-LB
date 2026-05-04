# ============================
# 文件：envs/dynamic_buffer_env.py
# 作用：
#   上层“动态缓存 agent”的环境：
#     - 一次 episode 内，按生产进度分成若干阶段（K 段）；
#     - 每段开始前，上层选择一套缓冲配置 buffers_t；
#     - 下层用该 buffers_t + greedy 派工，模拟到下一个进度阈值或 deadlock；
#     - 前 K-1 段 reward = 0，最后一段给一次终局奖励：
#           R = -(makespan + lambda * avg_total_buffer + penalty * deadlock)
#
# 说明：
#   - 为了不强行入侵你的 ShopEnv，本文件在“如何跑一段调度”处留了一个
#     _simulate_segment(...) 的 TODO，你可以按你现在 BufferDesignEnv 里
#     evaluate_fn 的调用方式来实现。
#   - 设计目标是：接口上和 BufferDesignEnv 尽量一致，便于在 train_two_level
#     中通过一个 flag 切换静态 / 动态环境。
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np

from .base_env import BaseEnv
from instances.types import InstanceData


@dataclass
class DynamicBufferEnvConfig:
    """
    DynamicBufferEnv 的基本配置。

    字段：
        buffer_cost_weight (lambda):
            缓冲成本权重 λ_B，reward 中会乘在 avg_total_buffer 上。
        deadlock_penalty:
            deadlock 惩罚（加在 makespan 上）。
        progress_thresholds:
            进度阈值列表（严格递增的 0~1 之间的小数），例如 [0.25, 0.5, 0.75, 1.0]，
            表示一共 K=4 个决策阶段。
        randomize_instances:
            reset 时是否随机选择实例；否则按顺序轮流。
        seed:
            随机种子。
    """
    buffer_cost_weight: float = 1.0
    deadlock_penalty: float = 2000.0
    progress_thresholds: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
    randomize_instances: bool = True
    seed: int = 0


class DynamicBufferEnv(BaseEnv):
    """
    动态缓冲环境：episode 内多次调整缓冲。

    使用方式（伪代码）::

        env = DynamicBufferEnv(
            instances=train_instances,
            buffer_candidates=buffer_candidate_list,   # List[List[int]]
            simulate_segment_fn=my_sim_fn,            # 见 __init__ 文档
            cfg=DynamicBufferEnvConfig(...),
        )

        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)   # 动作 = candidate 索引
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        instances: Sequence[InstanceData],
        buffer_candidates: Sequence[Sequence[int]],
        simulate_segment_fn: Callable[
            [InstanceData, Sequence[int], float, Dict[str, Any]],
            Dict[str, Any],
        ],
        cfg: Optional[DynamicBufferEnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        参数：
            instances:
                一组算例，假定阶段数一致。
            buffer_candidates:
                上层可选的缓冲向量候选集，例如：
                    [(3, 3), (2, 2), (3, 2), (2, 1), (1, 1), ...]
                动作空间就是这些候选的索引。
            simulate_segment_fn:
                用户提供的“跑一段生产”的函数，签名为：
                    metrics = simulate_segment_fn(instance, buffers, target_progress, sim_state)
                其中：
                    - instance: 当前 InstanceData；
                    - buffers:  当前阶段生效的缓冲向量；
                    - target_progress: 本段目标进度（0~1，例：0.25 表示做到 25% 完工）；
                    - sim_state: 一个可变字典，用于跨段累积下层模拟状态；
                返回 metrics 字典，至少包含：
                    - "makespan"        : 当前累计 makespan（从 episode 开始算起）；
                    - "avg_total_buffer": 截止目前的平均 total_buffer；
                    - "deadlock"        : 是否 deadlock（bool 或 0/1）；
                    - "progress"        : 当前已完工进度（0~1）。
                注意：如果 deadlock 提前发生，simulate_segment_fn 应该设置
                    metrics["deadlock"]=True，progress 不必到达 target_progress。
            cfg:
                DynamicBufferEnvConfig，如果为 None 则使用默认配置。
            seed:
                随机种子，若不为 None 则覆盖 cfg.seed。
        """
        super().__init__()
        self._instances: List[InstanceData] = list(instances)
        if len(self._instances) == 0:
            raise ValueError("DynamicBufferEnv: instances 为空")

        self._buffer_candidates = [list(b) for b in buffer_candidates]
        if len(self._buffer_candidates) == 0:
            raise ValueError("DynamicBufferEnv: buffer_candidates 为空")

        self._simulate_segment_fn = simulate_segment_fn

        self._cfg = cfg or DynamicBufferEnvConfig()
        if seed is not None:
            self._cfg.seed = seed

        # RNG
        self._rng = np.random.RandomState(self._cfg.seed)

        # 进度阈值
        self._progress_thresholds = list(self._cfg.progress_thresholds)
        if len(self._progress_thresholds) == 0:
            raise ValueError("progress_thresholds 不能为空")
        if not all(0 < p <= 1 for p in self._progress_thresholds):
            raise ValueError("progress_thresholds 必须在 (0, 1] 范围内")
        if any(self._progress_thresholds[i] >= self._progress_thresholds[i + 1]
               for i in range(len(self._progress_thresholds) - 1)):
            raise ValueError("progress_thresholds 必须严格递增")

        # 内部状态
        self._num_segments: int = len(self._progress_thresholds)
        self._current_segment: int = 0
        self._current_instance_idx: int = 0
        self._sim_state: Dict[str, Any] = {}
        self._last_metrics: Dict[str, Any] = {}

        # 为 observation 简化设计：直接拼成一个定长向量
        # obs = [progress, segment_idx/K, 当前 buffers, 上一段 metrics 的若干标量...]
        self._num_buffers: int = len(self._buffer_candidates[0])
        self._obs_dim: int = 2 + self._num_buffers + 3  # 进度 + 段索引 + buffers + (makespan, avg_buf, deadlock)

    # ============ Gym API ============

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return len(self._buffer_candidates)

    def reset(self) -> np.ndarray:
        """开始新 episode，重置实例 / 段索引 / 下层模拟状态等。"""
        # 选实例
        if self._cfg.randomize_instances:
            self._current_instance_idx = int(self._rng.randint(0, len(self._instances)))
        else:
            self._current_instance_idx = (self._current_instance_idx + 1) % len(self._instances)

        self._current_segment = 0
        self._sim_state = {
            "progress": 0.0,
            "makespan": 0.0,
            "avg_total_buffer": 0.0,
            "deadlock": False,
        }
        self._last_metrics = dict(self._sim_state)

        # 当前 buffer 先置为全 0（尚未决策）
        self._current_buffers = [0] * self._num_buffers

        return self._build_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        在当前阶段选择一个 buffer 向量（通过索引），然后模拟到下一进度阈值。
        """
        done = False
        reward = 0.0

        # 1) 解释动作：索引 -> buffers
        a = int(np.clip(action, 0, self.action_dim - 1))
        self._current_buffers = list(self._buffer_candidates[a])

        # 2) 运行本段模拟
        instance = self._instances[self._current_instance_idx]
        target_progress = self._progress_thresholds[self._current_segment]

        metrics = self._simulate_segment_fn(
            instance=instance,
            buffers=self._current_buffers,
            target_progress=target_progress,
            sim_state=self._sim_state,
        )
        # 更新 sim_state & last_metrics
        for key in ["progress", "makespan", "avg_total_buffer", "deadlock"]:
            if key in metrics:
                self._sim_state[key] = metrics[key]
        self._last_metrics = dict(self._sim_state)

        # 3) 判断是否终局
        deadlock_flag = bool(self._sim_state.get("deadlock", False))
        progress = float(self._sim_state.get("progress", 0.0))

        # 正常情况下 progress 应该 >= target_progress；
        # 若 deadlock 提前发生，progress 可能达不到 1.0，用 deadlock_flag 区分。
        is_last_segment = (self._current_segment == self._num_segments - 1) or deadlock_flag or (progress >= 0.9999)

        if is_last_segment:
            done = True
            reward = self._compute_final_reward(self._last_metrics)
        else:
            # 进入下一段
            self._current_segment += 1

        obs_next = self._build_obs()
        info = {
            "instance_idx": self._current_instance_idx,
            "segment_idx": self._current_segment,
            "buffers": list(self._current_buffers),
            "metrics": dict(self._last_metrics),
        }
        return obs_next, reward, done, info

    # ============ 内部工具函数 ============

    def _compute_final_reward(self, metrics: Dict[str, Any]) -> float:
        """终局 reward：-(makespan + lambda * avg_total_buffer + penalty * deadlock)."""
        makespan = float(metrics.get("makespan", 0.0))
        avg_total_buffer = float(metrics.get("avg_total_buffer", 0.0))
        deadlock_flag = bool(metrics.get("deadlock", False))

        lam = float(self._cfg.buffer_cost_weight)
        penalty = float(self._cfg.deadlock_penalty)

        if deadlock_flag:
            makespan = makespan + penalty

        return -(makespan + lam * avg_total_buffer)

    def _build_obs(self) -> np.ndarray:
        """
        一个简单的观测向量构造：
            [ progress,
              current_segment / num_segments,
              current_buffers (归一化),
              last_makespan_norm,
              last_avg_buffer_norm,
              last_deadlock_flag ]
        你可以以后再换成更复杂的 obs builder。
        """
        progress = float(self._sim_state.get("progress", 0.0))
        seg_ratio = self._current_segment / max(1, self._num_segments - 1)

        # 简单归一化：假设每个 buffer 候选 <= 10
        buf_norm = np.array(self._current_buffers, dtype=np.float32) / 10.0

        last_ms = float(self._last_metrics.get("makespan", 0.0))
        last_buf = float(self._last_metrics.get("avg_total_buffer", 0.0))
        last_dead = 1.0 if bool(self._last_metrics.get("deadlock", False)) else 0.0

        # 粗略归一化（可以按你的实例上界再细化）
        last_ms_norm = last_ms / 1000.0
        last_buf_norm = last_buf / 10.0

        obs = np.concatenate(
            [
                np.array([progress, seg_ratio], dtype=np.float32),
                buf_norm,
                np.array([last_ms_norm, last_buf_norm, last_dead], dtype=np.float32),
            ]
        )
        return obs

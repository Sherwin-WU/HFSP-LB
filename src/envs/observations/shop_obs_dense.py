# ============================
# 文件：envs/observations/shop_obs_dense.py
# 说明：
#   为下层调度环境（基于 FlowShopCoreEnv）构造 DQN 可用的
#   “稠密向量”状态表示。
#
#   结构（拼接顺序）：
#
#     obs = concat(
#         global_features,        # [t_norm, frac_completed, current_stage_one_hot[K], avg_buf_occ]
#         buffer_features,        # [K-1] 各缓冲占用率
#         machine_features,       # [4 * M_tot] 每台机器 [idle, busy, blocked, rem_time_norm]
#         job_features,           # [3 * J] 每个工件 [finished, stage_frac, ready]
#     )
#
#   其中：
#     - J       = num_jobs
#     - K       = num_stages
#     - M_tot   = sum_k m(k) = 总机器数
#
#   设计目标：
#     - 维度固定（在 J, K, machines_per_stage 固定的情况下）；
#     - 能直接对接 DQN / DDQN / D3QN 等离散动作算法；
#     - 与 envs/ffs_core_env.FlowShopCoreEnv 的字段严格对齐。
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.ffs_core_env import FlowShopCoreEnv
else:
    FlowShopCoreEnv = Any  # type: ignore

# import numpy as np

# try:
#     # 仅用于类型提示，不强依赖
#     from envs.ffs_core_env import FlowShopCoreEnv
# except ImportError:  # pragma: no cover
#     FlowShopCoreEnv = Any  # type: ignore


@dataclass
class ShopObsConfig:
    """
    下层调度状态编码的配置。

    字段：
        time_norm_ref:
            用于归一化时间的参考值 T_ref，t_norm = time / T_ref。
            若为 None 或 <= 0，则在构造 obs 时自动推断一个参考值：
                T_ref ≈ num_stages * max_processing_time(inst)。
        proc_time_norm_ref:
            用于归一化剩余加工时间的参考值 P_ref，rem_norm = rem_time / P_ref。
            若为 None 或 <= 0，则自动设为
                P_ref = max_processing_time(inst)。
    """
    time_norm_ref: Optional[float] = None
    proc_time_norm_ref: Optional[float] = None


# ----------------------------------------------------------------------
# 内部工具：推断归一化参考尺度
# ----------------------------------------------------------------------

def _infer_proc_time_ref(core_env: FlowShopCoreEnv) -> float:
    """
    从 InstanceData 中推断一个加工时间参考尺度 P_ref：
        P_ref = max_{j,k,m} proc_time(j,k,m)
    若所有加工时间均为 0，则返回 1.0。
    """
    inst = core_env.instance
    max_p = 0.0
    for job in inst.jobs:
        for op in job.ops:
            for p in op.proc_times:
                if p > max_p:
                    max_p = float(p)
    if max_p <= 0.0:
        max_p = 1.0
    return max_p


def _infer_time_ref(core_env: FlowShopCoreEnv, proc_time_ref: float) -> float:
    """
    推断时间归一化参考尺度 T_ref。
    简单取：
        T_ref = num_stages * proc_time_ref
    作为一个粗略上界，保证 t_norm ≈ O(1) 的量级。
    """
    k = core_env.num_stages
    t_ref = k * proc_time_ref
    if t_ref <= 0.0:
        t_ref = 1.0
    return t_ref


# ----------------------------------------------------------------------
# 主接口：构建下层调度环境的稠密状态向量
# ----------------------------------------------------------------------

def build_shop_obs(
    core_env: FlowShopCoreEnv,
    cfg: Optional[ShopObsConfig] = None,
) -> np.ndarray:
    """
    基于 FlowShopCoreEnv 构建 DQN 的状态向量。

    参数：
        core_env:
            核心调度环境实例（envs/ffs_core_env.FlowShopCoreEnv）。
        cfg:
            ShopObsConfig，控制时间和加工时间的归一化尺度；
            若为 None，则使用自动推断的参考值。

    返回：
        obs: np.ndarray, dtype=float32, shape = [D]
             拼接顺序为：
                [ global_features,
                  buffer_features,
                  machine_features,
                  job_features ]
    """
    if cfg is None:
        cfg = ShopObsConfig()

    # -----------------------------
    # 1. 基本规模参数
    # -----------------------------
    K = core_env.num_stages
    J = core_env.num_jobs
    machines_per_stage = core_env.machines_per_stage
    M_tot = sum(int(mk) for mk in machines_per_stage)

    # -----------------------------
    # 2. 归一化参考尺度
    # -----------------------------
    # 2.1 加工时间参考尺度 P_ref
    if cfg.proc_time_norm_ref is not None and cfg.proc_time_norm_ref > 0.0:
        P_ref = float(cfg.proc_time_norm_ref)
    else:
        P_ref = _infer_proc_time_ref(core_env)

    # 2.2 时间参考尺度 T_ref
    if cfg.time_norm_ref is not None and cfg.time_norm_ref > 0.0:
        T_ref = float(cfg.time_norm_ref)
    else:
        T_ref = _infer_time_ref(core_env, P_ref)

    # -----------------------------
    # 3. 全局特征块 g
    #    [ t_norm, frac_completed, current_stage_one_hot[K], avg_buf_occ ]
    # -----------------------------
    t_norm = float(core_env.time) / T_ref if T_ref > 0.0 else 0.0

    num_finished = sum(1 for j in core_env.jobs if j.finished)
    frac_completed = float(num_finished) / float(J) if J > 0 else 0.0

    current_stage_one_hot = np.zeros((K,), dtype=np.float32)
    if core_env.current_stage is not None:
        s = int(core_env.current_stage)
        if 0 <= s < K:
            current_stage_one_hot[s] = 1.0

    # 平均缓冲占用率
    avg_buf_occ = _compute_average_buffer_occupancy(core_env)

    global_features = np.concatenate(
        [
            np.array([t_norm, frac_completed], dtype=np.float32),
            current_stage_one_hot.astype(np.float32),
            np.array([avg_buf_occ], dtype=np.float32),
        ],
        axis=0,
    )

    # -----------------------------
    # 4. 缓冲特征块 b: 各缓冲占用率 [K-1]
    # -----------------------------
    buffer_features = _build_buffer_features(core_env, K)

    # -----------------------------
    # 5. 机器特征块 M: 每台机器 [idle, busy, blocked, rem_time_norm]
    # -----------------------------
    machine_features = _build_machine_features(core_env, K, P_ref)

    # -----------------------------
    # 6. 工件特征块 J: 每个 job [finished, stage_frac, ready]
    # -----------------------------
    job_features = _build_job_features(core_env, K, J)

    # -----------------------------
    # 7. 拼接所有部分
    # -----------------------------
    obs = np.concatenate(
        [
            global_features,
            buffer_features,
            machine_features,
            job_features,
        ],
        axis=0,
    )

    return obs.astype(np.float32)


# ----------------------------------------------------------------------
# 子模块：缓冲、机器、工件特征构造
# ----------------------------------------------------------------------

def _compute_average_buffer_occupancy(core_env: FlowShopCoreEnv) -> float:
    """
    计算平均缓冲占用率：
        mean_k ( len(queue_k) / capacity_k )，若没有缓冲则为 0。
    """
    buffers = getattr(core_env, "buffers", None)
    if not buffers:
        return 0.0

    ratios = []
    for buf in buffers:
        cap = float(getattr(buf, "capacity", 0))
        if cap > 0:
            occ = len(getattr(buf, "queue", []))
            ratios.append(occ / cap)

    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def _build_buffer_features(core_env: FlowShopCoreEnv, K: int) -> np.ndarray:
    """
    构建缓冲特征块：
        对每个中间缓冲段 k（0..K-2）：
            occ_ratio_k = len(queue_k) / capacity_k  （若 capacity_k=0，则为 0）
    """
    num_buffers = max(0, K - 1)
    if num_buffers == 0:
        return np.zeros((0,), dtype=np.float32)

    buffers = getattr(core_env, "buffers", [])
    if len(buffers) != num_buffers:
        # 容错：若缓冲数量与 K-1 不符，则截断/补齐 0
        feats = np.zeros((num_buffers,), dtype=np.float32)
        for k in range(min(num_buffers, len(buffers))):
            buf = buffers[k]
            cap = float(getattr(buf, "capacity", 0))
            if cap > 0:
                occ = len(getattr(buf, "queue", []))
                feats[k] = float(occ / cap)
        return feats

    feats = np.zeros((num_buffers,), dtype=np.float32)
    for k, buf in enumerate(buffers):
        cap = float(getattr(buf, "capacity", 0))
        if cap > 0:
            occ = len(getattr(buf, "queue", []))
            feats[k] = float(occ / cap)
        else:
            feats[k] = 0.0

    return feats


def _build_machine_features(
    core_env: FlowShopCoreEnv,
    K: int,
    proc_time_ref: float,
) -> np.ndarray:
    """
    构建机器特征块：
        对每台机器 (stage, machine_idx)，构造 4 维特征：
            [ idle_indicator,
              busy_indicator,
              blocked_indicator,
              rem_time_norm ]
        然后按 stage 顺序展开并拼接。
    """
    machines = core_env.machines
    feats_list = []

    for s in range(K):
        stage_machines = machines[s]
        for m in stage_machines:
            status = m.status
            idle = 1.0 if status == "idle" else 0.0
            busy = 1.0 if status == "busy" else 0.0
            blocked = 1.0 if status == "blocked" else 0.0

            # 剩余加工时间：仅对 busy 机器有意义
            rem_time = 0.0
            if status == "busy":
                rem_time = max(0.0, float(m.finish_time) - float(core_env.time))
            rem_time_norm = rem_time / proc_time_ref if proc_time_ref > 0.0 else 0.0

            feats_list.append([idle, busy, blocked, rem_time_norm])

    if not feats_list:
        return np.zeros((0,), dtype=np.float32)

    feats = np.array(feats_list, dtype=np.float32).reshape(-1)
    return feats


def _build_job_features(
    core_env: FlowShopCoreEnv,
    K: int,
    J: int,
) -> np.ndarray:
    """
    构建工件特征块：
        对每个 job j，构造 3 维特征：
            finished_j  = 1 if job.finished else 0
            stage_frac_j = current_stage_j / K    (已完工的 job 当前阶段视为 K)
            ready_j     = 是否在当前决策 stage 上可派工：
                          若 core_env.current_stage is None -> 0；
                          若 current_stage == 0:
                              ready_j = 1 if job_id in stage0_queue
                          若 current_stage > 0:
                              ready_j = 1 if job_id in buffers[current_stage-1].queue
    """
    jobs = core_env.jobs
    stage0_queue = getattr(core_env, "stage0_queue", [])
    buffers = getattr(core_env, "buffers", [])

    feats_list = []

    current_stage = core_env.current_stage
    for job_id in range(J):
        job_state = jobs[job_id]

        # 1) finished_j
        finished_j = 1.0 if job_state.finished else 0.0

        # 2) stage_frac_j：已完工的 job，stage 视为 K
        stage = int(job_state.current_stage)
        if job_state.finished:
            stage = K
        stage_clamped = max(0, min(stage, K))
        stage_frac_j = float(stage_clamped) / float(K) if K > 0 else 0.0

        # 3) ready_j：是否在当前 stage 的待加工队列中
        ready_j = 0.0
        if current_stage is not None:
            s = int(current_stage)
            if s == 0:
                if job_id in stage0_queue:
                    ready_j = 1.0
            elif 0 < s < K:
                buf_idx = s - 1
                if 0 <= buf_idx < len(buffers):
                    if job_id in buffers[buf_idx].queue:
                        ready_j = 1.0

        feats_list.append([finished_j, stage_frac_j, ready_j])

    if not feats_list:
        return np.zeros((0,), dtype=np.float32)

    feats = np.array(feats_list, dtype=np.float32).reshape(-1)
    return feats

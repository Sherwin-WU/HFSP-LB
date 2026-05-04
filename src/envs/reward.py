# ============================
# 文件：envs/reward.py
# 说明：
#   集中管理上下层环境的奖励函数定义，支持多种模式和超参数，
#   方便做奖励函数/超参数/奖励项的消融实验。
#
# 对外提供：
#   - ShopRewardConfig
#   - make_shop_reward_fn(cfg) -> reward_fn(prev_state, next_state, done, info) -> float
#
#   - BufferRewardConfig
#   - compute_buffer_reward(metrics, buffers, cfg) -> float
#   - make_buffer_reward_fn(cfg) -> custom_reward_fn(metrics, buffers, cfg) -> float
#
# 约定：
#   下层 info 字段通常由 envs/ffs_core_env.FlowShopCoreEnv 提供，包含：
#       - "delta_time":            本步推进时间 Δt
#       - "blocking_count":        当前阻塞机器数量
#       - "avg_buffer_occupancy":  当前平均缓冲占用率（0~1）
#       - "invalid_action":        当前动作是否非法（no-op）
#       - "makespan":              done=True 时的完工时间
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence


# ============================================================
# 下层 ShopEnv（调度）的奖励定义
# ============================================================

@dataclass
class ShopRewardConfig:
    """
    下层调度环境的奖励配置。

    mode:
        "terminal" :
            仅在 episode 结束时给一个
                r_T = - makespan_weight * makespan
            中间步骤 reward = 0。
        "dense"    :
            每步给
                r_t = - time_weight * delta_time
            若 done=True 且 info 中含 makespan，可在终局额外加一次性
                - makespan_weight * makespan。
        "blocking" :
            在 "dense" 的基础上，每步额外惩罚阻塞和缓冲占用：
                r_t = - time_weight * Δt
                      - blocking_weight * blocking_count
                      - buffer_occ_weight * avg_buffer_occupancy
            若 done=True，可同样加 - makespan_weight * makespan。

    其它字段：
        time_weight:
            时间消耗的权重（越大代表对总时间越敏感）。
        blocking_weight:
            阻塞机器数惩罚权重。
        buffer_occ_weight:
            缓冲占用惩罚权重。
        makespan_weight:
            终局 makespan 惩罚权重。
        invalid_action_weight:
            非法动作（no-op）惩罚权重：
                若 info["invalid_action"] 为 True，则额外
                    r_t -= invalid_action_weight
    """
    mode: str = "blocking"        # "blocking" / "dense" / "terminal" / "progress"等
    time_weight: float = 1.0
    blocking_weight: float = 0.5
    buffer_occ_weight: float = 0.0
    makespan_weight: float = 0.0
    invalid_action_weight: float = 0.1

    # ==== 新增：progress 模式专用参数 ====
    per_operation_reward: float = 0.0   # 每完成一道工序的 shaping 奖励
    per_job_reward: float = 0.0         # 每完成一个工件的 shaping 奖励
    blocking_penalty: float = 0.0       # 只要存在堵塞就惩罚（额外于 blocking_weight）
    terminal_bonus: float = 0.0         # 所有 job 完成时的终局奖励

def compute_shared_epi_reward(
    makespan: float,
    buffers: Sequence[int],
    deadlock: bool,
    buffer_cost_weight: float,
    deadlock_penalty: float,
) -> float:
    """
    计算 UBA / LDA 共享的 episodic reward。

    与论文中的全局终局奖励口径保持一致：
        r_epi = - makespan
                - lambda_b * sum(buffers)
                - lambda_d * deadlock_flag
    """
    total_buffer = float(sum(int(b) for b in buffers)) if buffers is not None else 0.0
    deadlock_cost = float(deadlock_penalty) if bool(deadlock) else 0.0

    return - float(makespan) - float(buffer_cost_weight) * total_buffer - deadlock_cost


def make_shop_reward_fn(cfg: ShopRewardConfig) -> Callable[[Any, Any, bool, Dict[str, Any]], float]:
    """
    根据 ShopRewardConfig 生成一个 reward_fn，供 ShopEnv 使用。

    返回的 reward_fn 接口：
        reward = reward_fn(prev_state, next_state, done, info)

    约定 info 中可包含的关键字段（由核心 env 提供，如 FlowShopCoreEnv）：
        - "delta_time":            本步推进时间 Δt（若缺省则视为 0.0）
        - "blocking_count":        当前阻塞机器数量（可选）
        - "avg_buffer_occupancy":  当前平均缓冲占用率（可选）
        - "invalid_action":        当前动作是否非法（可选，bool）
        - "makespan":              当前已知 makespan（通常在 done=True 时提供）

    对于 mode="progress"：
        - 认为 next_state 是核心 env（如 FlowShopCoreEnv）实例，
          其中包含：
              * .jobs: 作业状态列表，每个 job 有 .finished, .current_stage 等字段
              * .num_stages: 阶段数
              * .num_jobs:   作业数
    """
    mode = cfg.mode.lower()

    # progress 模式下用于计算 Δops / Δjobs 的内部状态
    prev_total_ops: int = 0
    prev_finished_jobs: int = 0

    def reward_fn(prev_state: Any, next_state: Any, done: bool, info: Dict[str, Any]) -> float:
        nonlocal prev_total_ops, prev_finished_jobs

        r = 0.0

        # -----------------------------
        # 1) 时间惩罚：dense / blocking / progress
        # -----------------------------
        delta_time = float(info.get("delta_time", 0.0))
        if mode in {"dense", "blocking", "progress"}:
            r -= cfg.time_weight * delta_time

        # -----------------------------
        # 2) 阻塞 / 缓冲占用惩罚（blocking 模式）
        # -----------------------------
        if mode == "blocking":
            blocking_count = float(info.get("blocking_count", 0.0))
            avg_buf_occ = float(info.get("avg_buffer_occupancy", 0.0))
            r -= cfg.blocking_weight * blocking_count
            r -= cfg.buffer_occ_weight * avg_buf_occ

        # -----------------------------
        # 3) progress 模式：进度 + 堵塞 + 终局
        # -----------------------------
        if mode == "progress":
            core_env = next_state  # 约定 next_state 即为核心 env

            # 3.1 统计当前已完成工序数 total_ops 和已完成工件数 finished_jobs
            total_ops = 0
            finished_jobs = 0

            num_stages = int(getattr(core_env, "num_stages", 0))
            jobs = getattr(core_env, "jobs", [])

            for job in jobs:
                if getattr(job, "finished", False):
                    finished_jobs += 1
                    total_ops += num_stages
                else:
                    current_stage = int(getattr(job, "current_stage", 0))
                    # 已完成工序数 = current_stage（截断到 [0, num_stages] 范围）
                    total_ops += max(0, min(current_stage, num_stages))

            delta_ops = total_ops - prev_total_ops
            delta_jobs = finished_jobs - prev_finished_jobs

            prev_total_ops = total_ops
            prev_finished_jobs = finished_jobs

            # 3.2 是否有阻塞：优先用 info["blocking_count"]
            if "blocking_count" in info:
                blocking_indicator = 1.0 if info["blocking_count"] > 0 else 0.0
            else:
                # 兜底：从 core_env.machines 扫描
                machines = getattr(core_env, "machines", [])
                any_blocking = any(
                    getattr(m, "status", None) == "blocked"
                    for machines_in_stage in machines
                    for m in machines_in_stage
                )
                blocking_indicator = 1.0 if any_blocking else 0.0

            # 3.3 progress 的 shaping 项：Δops / Δjobs / 堵塞
            r += cfg.per_operation_reward * float(max(delta_ops, 0))
            r += cfg.per_job_reward * float(max(delta_jobs, 0))
            r -= cfg.blocking_penalty * blocking_indicator

            # 3.4 终局 bonus：所有 job 完成且 done=True 时给一次
            terminal_bonus = 0.0
            all_done = False

            all_done_fn = getattr(core_env, "_all_jobs_completed", None)
            if callable(all_done_fn):
                all_done = bool(all_done_fn())
            else:
                num_jobs = int(getattr(core_env, "num_jobs", len(jobs)))
                all_done = (finished_jobs == num_jobs)

            if done and all_done:
                terminal_bonus = cfg.terminal_bonus

            r += terminal_bonus

        # -----------------------------
        # 4) 非法动作惩罚（所有模式共用）
        # -----------------------------
        if info.get("invalid_action", False):
            r -= cfg.invalid_action_weight

        # -----------------------------
        # 5) 终局 makespan 惩罚（可选）
        # -----------------------------
        if "makespan" in info and done and cfg.makespan_weight != 0.0:
            makespan = float(info["makespan"])
            r -= cfg.makespan_weight * makespan

        # -----------------------------
        # 6) progress 模式：episode 结束时重置内部计数，防止跨 episode 污染
        # -----------------------------
        if mode == "progress" and done:
            prev_total_ops = 0
            prev_finished_jobs = 0

        return float(r)

    return reward_fn



# ============================================================
# 上层 BufferDesignEnv 的奖励定义
# ============================================================

@dataclass
class BufferRewardConfig:
    """
    上层缓冲设计的奖励配置。

    默认形式：
        R = -(makespan_weight * makespan + buffer_cost_weight * sum(buffers))

    字段：
        makespan_weight:
            makespan 的惩罚权重。
        buffer_cost_weight:
            缓冲总量 sum(buffers) 的惩罚权重（即 λ）。
    """
    makespan_weight: float = 1.0
    buffer_cost_weight: float = 1.0


def compute_buffer_reward(
    metrics: Dict[str, Any],
    buffers: Sequence[int],
    cfg: BufferRewardConfig,
) -> float:
    """
    根据 metrics 和 buffers 计算上层 reward。

    默认：
        metrics 必须包含 "makespan"；
        R = -(makespan_weight * makespan + buffer_cost_weight * sum(buffers))

    参数：
        metrics:
            evaluate_fn(instance, buffers) 返回的指标字典。
            必须至少包含键 "makespan"。
        buffers:
            当前 episode 学到的缓冲配置向量 B = (b_1, ..., b_K-1)。
        cfg:
            BufferRewardConfig，控制各项权重。

    返回：
        scalar reward (float)。
    """
    if "makespan" not in metrics:
        raise KeyError(
            "compute_buffer_reward: metrics 中缺少 'makespan' 键，"
            "默认奖励需要该字段。"
        )

    makespan = float(metrics["makespan"])
    total_buffer = float(sum(buffers))

    r = -(cfg.makespan_weight * makespan + cfg.buffer_cost_weight * total_buffer)
    return float(r)


def make_buffer_reward_fn(
    cfg: BufferRewardConfig,
) -> Callable[[Dict[str, Any], Sequence[int], BufferRewardConfig], float]:
    """
    构造一个可直接传给 BufferDesignEnv 的 custom_reward_fn。

    用法示例::

        from envs.reward import BufferRewardConfig, make_buffer_reward_fn
        from envs.buffer_design_env import BufferDesignEnv, BufferDesignEnvConfig

        cfg_buf = BufferRewardConfig(makespan_weight=1.0, buffer_cost_weight=0.1)
        custom_fn = make_buffer_reward_fn(cfg_buf)

        env = BufferDesignEnv(
            instances=train_instances,
            evaluate_fn=my_eval_fn,  # (instance, buffers) -> metrics
            cfg=BufferDesignEnvConfig(buffer_cost_weight=cfg_buf.buffer_cost_weight),
            custom_reward_fn=custom_fn,
        )

    注意：
        这里 custom_reward_fn 的签名为：
            reward = custom_reward_fn(metrics, buffers, cfg_arg)
        其中 cfg_arg 会由 BufferDesignEnv 传入（可忽略），真正使用的是此处外层捕获的 cfg。
    """
    def custom_reward_fn(
        metrics: Dict[str, Any],
        buffers: Sequence[int],
        _: BufferRewardConfig,  # 为兼容 BufferDesignEnv 的签名，这里位置传 cfg_arg 但内部使用外层 cfg
    ) -> float:
        return compute_buffer_reward(metrics, buffers, cfg)

    return custom_reward_fn

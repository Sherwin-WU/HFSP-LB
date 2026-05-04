# ============================
# 文件：src/envs/ffs_core_env.py
# 说明：
#   FlowShopCoreEnv：有限缓冲柔性流水车间（FFS）的核心仿真环境。
#
#   特点：
#     - 只负责“真·调度仿真”，不直接和 RL 打交道；
#     - 动作是 job 的索引（action_job_id: int）；
#     - 每次 RL 决策只针对“一个 stage 的一台机器”：
#         · 当前需要调度的 stage 由环境自动选出（有空闲机且有待加工 job）；
#         · 给定 job 后，环境根据 dispatch_machine_rule 在该 stage 内选具体机器；
#     - 非法动作（job 不能在当前 stage 派工）被视为 no-op：
#         · 不派工，系统仍推进到下一个决策点；
#         · info["invalid_action"] = True，供 reward_fn 使用 invalid_action_weight 惩罚。
#
#   与外部的典型配合方式：
#     - 下层 RL 通过 envs/shop_env.ShopEnv 使用本类作为 core_env：
#
#         core_env = FlowShopCoreEnv(instance, buffers, ...)
#         shop_env = ShopEnv(
#             core_env=core_env,
#             obs_builder=build_shop_obs,   # 由你在 envs/observations/ 中实现
#             reward_fn=make_shop_reward_fn(shop_reward_cfg),
#         )
#
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import heapq
import random

import numpy as np

from instances.types import InstanceData


# ---------------------------------------------------------------------------
# 内部数据结构：Job / Machine / Buffer / Event
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    job_id: int
    current_stage: int              # 下一道工序的阶段索引（0..K-1），已完工时可为 K
    finished: bool = False
    completion_time: float = 0.0
    total_block_time: float = 0.0   # 总阻塞时间（可选用于统计）
    last_block_start: Optional[float] = None  # 最近一次开始阻塞的时间


@dataclass
class MachineState:
    stage_id: int
    machine_idx: int                # 该 stage 内的索引 0..m_k-1
    status: str = "idle"            # "idle" / "busy" / "blocked"
    current_job: Optional[int] = None
    finish_time: float = 0.0        # 当前任务计划完工时间（busy 时有效）
    blocked_since: Optional[float] = None  # 若 status="blocked"，记录开始阻塞时间


@dataclass
class BufferState:
    """
    阶段 k 与 k+1 之间的中间缓冲。
    """
    stage_before: int               # 前一阶段 k
    capacity: int                   # b_k
    queue: List[int]                # job_id 队列（FIFO）


@dataclass(order=True)
class Event:
    """
    未来事件：目前只考虑“某台机器上的当前 job 加工完成”这一类事件。
    使用 heapq 按 time 升序组织。
    """
    time: float
    stage_id: int
    machine_idx: int
    job_id: int


# ---------------------------------------------------------------------------
# FlowShopCoreEnv 定义
# ---------------------------------------------------------------------------

class FlowShopCoreEnv:
    """
    有限缓冲 FFS 核心调度环境。

    约定：
        - 实例由 InstanceData 表示（instances.types.InstanceData）；
        - 缓冲配置 buffers 是长度为 K-1 的整数列表，表示每段缓冲的容量 b_k；
        - 动作是一个 job 索引（0..num_jobs-1）：
            · 环境内部选定当前要派工的 stage；
            · 若该 job 在该 stage 的队列中可派工，则为合法动作：
                安排 job 到该 stage 的某台机器（依据 dispatch_machine_rule），
                然后推进 DES 到下一个“决策点”；
            · 否则视为非法动作：
                不派工，当作 no-op，同样推进到下一个决策点，
                并在 info["invalid_action"] = True 中打标。

    reset():
        - 清空所有状态；
        - 时间 t ← 0；
        - 所有 job 处于“待在 stage 0”的状态，其 job_id 放入 stage0_queue；
        - 所有机器 idle；
        - 所有缓冲清空；
        - 不执行任何自动派工，只推进到第一个“需要决策的 stage”。

    step(action_job_id):
        - 使用当前 env.current_stage 确定本次要派工的 stage；
        - 若 action_job_id 对于该 stage 可派工：
            · 从对应队列（stage0_queue 或 buffers[k-1].queue）中取出该 job；
            · 按 dispatch_machine_rule 选择一台 idle 机器；
            · 启动加工，生成完工事件，加入事件堆；
        - 若不可派工：
            · 不派工，标记 invalid_action=True；
        - 然后调用 _advance_until_decision_point()：
            · 自动处理事件，释放阻塞 job，直至下一次有 stage 需要派工，或系统终止；
        - 返回 (core_state, done, info)，其中：
            · core_state 通常就是 self；
            · info 中至少包含：
                - "delta_time": 本步从上一个决策点到当前决策点的时间推进量；
                - "invalid_action": 是否非法动作；
                - "blocking_count": 当前阻塞机器数；
                - "avg_buffer_occupancy": 当前平均缓冲占用率；
                - 若 done=True，则含 "makespan"。
    """

    def __init__(
        self,
        instance: InstanceData,
        buffers: Sequence[int],
        dispatch_machine_rule: str = "min_index",
        seed: Optional[int] = None,
    ) -> None:
        """
        参数：
            instance:
                InstanceData，描述 FFS 实例。
                要求：len(instance.machines_per_stage) = 阶段数 K。
            buffers:
                长度为 K-1 的整数列表，表示每个中间缓冲的容量 b_k。
            dispatch_machine_rule:
                在给定 stage s 与 job j 的前提下，选择具体机器的规则：
                    - "min_index"      : 选择 stage 内机器索引最小的 idle 机器；
                    - "earliest_idle"  : 对于 idle 机等价于 "min_index"；保留接口；
                    - "ect"            : Earliest Completion Time，选择
                                          t_now + proc_time(j,k,机) 最小的 idle 机；
                    - "random"         : 随机选择一个 idle 机。
            seed:
                随机种子（用于 "random" 规则）。
        """
        self.instance = instance
        self.machines_per_stage: List[int] = list(instance.machines_per_stage)
        self.num_stages: int = len(self.machines_per_stage)

        if self.num_stages <= 0:
            raise ValueError("InstanceData 中 machines_per_stage 长度应大于 0。")

        self.num_jobs: int = len(instance.jobs)
        if self.num_jobs == 0:
            raise ValueError("InstanceData 中 jobs 列表为空。")

        if len(buffers) != max(0, self.num_stages - 1):
            raise ValueError(
                f"buffers 长度应为 num_stages-1={self.num_stages - 1}，"
                f"实际为 {len(buffers)}"
            )
        self.buffer_capacities: List[int] = [int(b) for b in buffers]

        self.dispatch_machine_rule: str = dispatch_machine_rule.lower()
        self.rng = random.Random(seed)

        # 运行时状态（在 reset 中初始化）
        self.time: float = 0.0
        self.jobs: List[JobState] = []
        self.machines: List[List[MachineState]] = []
        self.buffers: List[BufferState] = []
        self.stage0_queue: List[int] = []  # stage 0 的待加工 job 队列（job_id）

        self.future_events: List[Event] = []  # heapq

        self.current_stage: Optional[int] = None  # 当前需要派工的 stage（0..K-1）

    # ----------------------------------------------------------------------
    # 公开接口：reset / step
    # ----------------------------------------------------------------------
    def reset(self) -> "FlowShopCoreEnv":
        """
        重置环境到初始状态，并推进到第一个“需要派工的 stage”。

        返回：
            self（作为核心状态；上层 obs_builder 可以直接基于该实例构造观测）。
        """
        # 时间归零
        self.time = 0.0

        # 初始化 JobState
        self.jobs = [
            JobState(job_id=j, current_stage=0, finished=False)
            for j in range(self.num_jobs)
        ]

        # 初始化 MachineState
        self.machines = []
        for k, mk in enumerate(self.machines_per_stage):
            stage_machines = [
                MachineState(stage_id=k, machine_idx=m)
                for m in range(mk)
            ]
            self.machines.append(stage_machines)

        # 初始化 BufferState
        self.buffers = []
        for k, cap in enumerate(self.buffer_capacities):
            self.buffers.append(
                BufferState(stage_before=k, capacity=int(cap), queue=[])
            )

        # stage0_queue：所有工件一开始都等待在阶段 0 的入口
        self.stage0_queue = list(range(self.num_jobs))

        # 清空事件表
        self.future_events = []

        # 当前需要派工的 stage 置空，随后通过 _advance_until_decision_point 选出
        self.current_stage = None

        # 推进到第一个决策点
        self._advance_until_decision_point()

        return self

    def _is_action_legal(self, action_job_id: int) -> bool:
        """
        判断给定 job 在当前决策点是否是“合法可派工”的动作。

        合法的条件：
        - 当前存在决策阶段 self.current_stage 不为 None；
        - job_id 在 [0, num_jobs) 范围内；
        - job 在当前阶段的待派工队列中（stage0_queue 或对应缓冲区的 queue）。
        """
        # 没有当前决策 stage，或者所有 job 已完成 -> 没有合法动作
        if self.current_stage is None:
            return False

        # job index 越界
        if not (0 <= action_job_id < self.num_jobs):
            return False

        stage = int(self.current_stage)

        # stage = 0 时，从 stage0_queue 选；否则从对应缓冲区的 queue 中选
        if stage == 0:
            ready_jobs = self.stage0_queue
        else:
            buf = self.buffers[stage - 1]
            ready_jobs = buf.queue

        return action_job_id in ready_jobs
    
    def get_legal_actions(self) -> List[int]:
        """
        返回当前时刻所有“合法可派工”的 job index 列表。
        若当前没有决策点（current_stage is None），返回空列表。
        """
        if self.current_stage is None:
            return []

        legal: List[int] = []
        for job_id in range(self.num_jobs):
            if self._is_action_legal(job_id):
                legal.append(job_id)
        return legal


    def step(self, action_job_id: int) -> Tuple["FlowShopCoreEnv", bool, Dict[str, Any]]:
        """
        执行一次调度决策。

        参数：
            action_job_id:
                DQN 输出的整数，表示选择哪个 job 进行派工。

        返回：
            core_state: self
            done:       bool，是否所有 job 都已完工或系统进入死锁/无法推进状态
            info:       dict，包含：
                          - "delta_time": 从上一个决策点到当前决策点的时间推进量；
                          - "invalid_action": 是否非法动作；
                          - "blocking_count": 当前阻塞机器数；
                          - "avg_buffer_occupancy": 当前平均缓冲占用率；
                          - 若 done=True，则含 "makespan"；
                          - 若检测到死锁，则含 "deadlock": True。
        """
        if self.current_stage is None and not self._all_jobs_completed():
            # 理论上不该发生：表示没有决策点但还有未完成 job
            # 视为死锁，直接返回 done=True
            info = {
                "delta_time": 0.0,
                "invalid_action": False,
                "blocking_count": self._current_blocking_count(),
                "avg_buffer_occupancy": self._current_average_buffer_occupancy(),
                "deadlock": True,
                "makespan": self._compute_makespan(),
            }
            return self, True, info

        prev_time = self.time
        invalid_action = False

        # 1) 若已经没有决策点（current_stage is None 且所有 job 完成），直接结束
        if self.current_stage is None and self._all_jobs_completed():
            info = {
                "delta_time": 0.0,
                "invalid_action": False,
                "blocking_count": self._current_blocking_count(),
                "avg_buffer_occupancy": self._current_average_buffer_occupancy(),
                "makespan": self._compute_makespan(),
            }
            return self, True, info

        # 2) 正常情况：有一个当前需要决策的 stage
        stage = int(self.current_stage) if self.current_stage is not None else 0

        # 2.1 判断动作是否合法（job 是否在该 stage 的待派工队列中）
        if not (0 <= action_job_id < self.num_jobs):
            invalid_action = True
        else:
            if stage == 0:
                ready_jobs = self.stage0_queue
            else:
                buf = self.buffers[stage - 1]
                ready_jobs = buf.queue

            if action_job_id in ready_jobs:
                # 合法动作：派工
                self._dispatch_job_to_stage(stage, action_job_id)
            else:
                # 非法动作：no-op
                invalid_action = True

        # 3) 推进 DES，直到下一个决策点或系统结束
        self._advance_until_decision_point()

        delta_time = float(self.time - prev_time)

        # 4) 汇总 info
        info: Dict[str, Any] = {
            "delta_time": delta_time,
            "invalid_action": invalid_action,
            "blocking_count": self._current_blocking_count(),
            "avg_buffer_occupancy": self._current_average_buffer_occupancy(),
        }

        done = False
        deadlock = False
        if self._all_jobs_completed():
            done = True
            info["makespan"] = self._compute_makespan()
        elif self.current_stage is None and not self.future_events:
            # 没有决策点也没有未来事件，但还有未完成 job -> 死锁/卡死
            done = True
            deadlock = True
            info["deadlock"] = True
            info["makespan"] = self._compute_makespan()

        return self, done, info

    # ----------------------------------------------------------------------
    # 内部工具：调度与事件推进
    # ----------------------------------------------------------------------
    def _dispatch_job_to_stage(self, stage: int, job_id: int) -> None:
        """
        将 job_id 派工到指定 stage 的某台 idle 机器上（由 dispatch_machine_rule 决定）。

        前置条件：
            - 确保 job_id 在该 stage 的待加工队列中；
            - 至少有一台该 stage 的机器处于 idle 状态。
        """
        # 从队列中移除 job
        if stage == 0:
            self.stage0_queue.remove(job_id)
        else:
            buf = self.buffers[stage - 1]
            buf.queue.remove(job_id)

        # 选择具体机器
        machine_idx = self._select_machine_for_stage(stage, job_id)
        machine = self.machines[stage][machine_idx]

        # 启动加工
        proc_time = self._get_processing_time(job_id, stage, machine_idx)
        start_time = self.time
        finish_time = start_time + proc_time

        machine.status = "busy"
        machine.current_job = job_id
        machine.finish_time = finish_time
        machine.blocked_since = None

        # 记录一个完工事件
        event = Event(
            time=finish_time,
            stage_id=stage,
            machine_idx=machine_idx,
            job_id=job_id,
        )
        heapq.heappush(self.future_events, event)


    def _select_machine_for_stage(self, stage: int, job_id: int) -> int:
        """
        在给定 stage 与 job 的前提下，根据 dispatch_machine_rule 选择一台 idle 机器。

        规则：
            - "min_index"     : 机器索引最小的 idle 机；
            - "earliest_idle" : 对于 idle 机与 "min_index" 等价，保留接口；
            - "ect"           : earliest completion time，
                                选择 self.time + proc_time(j,k,机) 最小的 idle 机；
            - "random"        : 从 idle 机中随机选一个。
        """
        idle_indices = [
            m.machine_idx
            for m in self.machines[stage]
            if m.status == "idle"
        ]
        if not idle_indices:
            raise RuntimeError(
                f"在 stage={stage} 尝试派工 job={job_id} 时，没有 idle 机器。"
            )

        rule = self.dispatch_machine_rule

        if rule in {"min_index", "earliest_idle"}:
            return min(idle_indices)

        if rule == "random":
            return self.rng.choice(idle_indices)

        if rule == "ect":
            # 选择 (time_now + proc_time) 最小的机器
            best_idx = None
            best_finish = None
            for m_idx in idle_indices:
                p = self._get_processing_time(job_id, stage, m_idx)
                finish = self.time + p
                if best_finish is None or finish < best_finish:
                    best_finish = finish
                    best_idx = m_idx
            assert best_idx is not None
            return best_idx

        # 未知规则：回退到 min_index
        return min(idle_indices)

    def _advance_until_decision_point(self) -> None:
        """
        自动推进 DES，直到出现“需要派工的 stage”（即有 idle 机 + 有待加工 job），
        或系统结束（所有 job 完成 / 死锁）。

        逻辑：
            while True:
                1) 释放所有可以从 blocked 状态转入缓冲的 job；
                2) 若所有 job 完成 -> current_stage = None，返回；
                3) 搜索所有 stage，若存在：
                       有 idle 机 且 对应待加工队列非空
                   则选出一个 stage 作为 current_stage，返回；
                4) 若 future_events 为空 -> 无法再推进，current_stage=None，返回；
                5) 否则：
                       弹出最近的事件，推进时间到该事件，处理完工逻辑，
                       循环继续。
        """
        while True:
            # 1) 尝试释放被缓冲容量限制阻塞的工件
            self._release_blocked_jobs()

            # 2) 检查是否所有作业已完成
            if self._all_jobs_completed():
                self.current_stage = None
                return

            # 3) 检查是否有 stage 需要派工
            candidate_stages = self._collect_candidate_stages()
            if candidate_stages:
                # 简单规则：选择 stage 索引最小的
                self.current_stage = min(candidate_stages)
                return

            # 4) 若没有未来事件，则系统无法再推进
            if not self.future_events:
                self.current_stage = None
                return

            # 5) 处理最近的完工事件，推进时间
            event = heapq.heappop(self.future_events)
            self._process_completion_event(event)

    def _process_completion_event(self, event: Event) -> None:
        """
        处理一个加工完工事件：
            - 更新 self.time；
            - 将对应 job 从机器上移出：
                · 若是最后一道工序：标记作业完成；
                · 否则尝试把 job 放入下一段缓冲；
                    · 若缓冲未满：直接入缓冲，机器 idle；
                    · 若缓冲已满：机器转为 blocked，job“卡在机器上”，
                                   等待缓冲有空位时再释放。
        """
        # 时间推进到事件时间
        prev_time = self.time
        self.time = max(self.time, event.time)
        dt = self.time - prev_time
        # 这里 dt 尚未直接用于 reward，reward 由上层 ShopEnv 计算

        stage = event.stage_id
        m_idx = event.machine_idx
        job_id = event.job_id

        machine = self.machines[stage][m_idx]
        if machine.current_job != job_id or machine.status not in {"busy", "blocked"}:
            # 正常情况下不会发生，若发生说明事件表与机器状态不一致
            # 为了稳健性，只是发出警告并忽略该事件
            # （可根据需要改为 raise）
            return

        job_state = self.jobs[job_id]

        if stage == self.num_stages - 1:
            # 最后一个阶段：作业完成
            job_state.finished = True
            job_state.current_stage = self.num_stages
            job_state.completion_time = self.time

            # 机器变为 idle
            machine.status = "idle"
            machine.current_job = None
            machine.finish_time = self.time
            machine.blocked_since = None
        else:
            # 中间阶段：尝试进入下一段缓冲
            buf = self.buffers[stage]  # 阶段 stage 与 stage+1 之间的缓冲
            if len(buf.queue) < buf.capacity:
                # 缓冲有空位，job 进入缓冲，机器 idle
                buf.queue.append(job_id)
                job_state.current_stage = stage + 1

                machine.status = "idle"
                machine.current_job = None
                machine.finish_time = self.time
                machine.blocked_since = None
            else:
                # 缓冲已满：机器阻塞，job 卡在机器上
                machine.status = "blocked"
                machine.blocked_since = self.time
                job_state.last_block_start = self.time
                # job_state.current_stage 可以直接更新为 stage+1；
                # 但由于 ready queue 是以缓冲队列为准，未入缓冲前不会被视为 ready
                job_state.current_stage = stage + 1

    def _release_blocked_jobs(self) -> None:
        """
        尝试将所有被缓冲容量限制阻塞的工件释放到缓冲中（若出现空位）。

        逻辑：
            对每个缓冲段 k，从前一阶段 k 的机器中寻找 status="blocked" 的机器：
                - 若缓冲 queue 未满，则将该 machine.current_job 移入缓冲；
                - 机器转为 idle；
                - 更新对应 job 的 total_block_time；
                - 重复，直到该缓冲段满或没有 blocked 机器。
        """
        for k, buf in enumerate(self.buffers):
            while len(buf.queue) < buf.capacity:
                # 找到前一阶段 k 上处于 blocked 状态的机器
                blocked_machines = [
                    m for m in self.machines[k] if m.status == "blocked"
                ]
                if not blocked_machines:
                    break

                # 简单策略：选择 machine_idx 最小的 blocked 机器
                m = min(blocked_machines, key=lambda x: x.machine_idx)
                job_id = m.current_job
                if job_id is None:
                    # 理论上不应发生
                    m.status = "idle"
                    m.blocked_since = None
                    continue

                job_state = self.jobs[job_id]

                # 将 job 移入缓冲
                buf.queue.append(job_id)

                # 计算阻塞时间贡献
                if job_state.last_block_start is not None:
                    duration = self.time - job_state.last_block_start
                    if duration > 0:
                        job_state.total_block_time += duration
                job_state.last_block_start = None

                # 机器转 idle
                m.status = "idle"
                m.current_job = None
                m.finish_time = self.time
                m.blocked_since = None

    # ----------------------------------------------------------------------
    # 辅助函数：状态检查与统计
    # ----------------------------------------------------------------------
    def _collect_candidate_stages(self) -> List[int]:
        """
        返回所有“当前需要派工”的 stage 列表：
            - 该 stage 上至少有一台 idle 机器；
            - 对应待加工队列非空：
                · stage 0 使用 stage0_queue；
                · stage k>0 使用 buffers[k-1].queue。
        """
        candidates: List[int] = []
        for s in range(self.num_stages):
            # 是否有 idle 机器
            if not any(m.status == "idle" for m in self.machines[s]):
                continue

            # 是否有待加工 job
            if s == 0:
                has_jobs = len(self.stage0_queue) > 0
            else:
                has_jobs = len(self.buffers[s - 1].queue) > 0

            if has_jobs:
                candidates.append(s)
        return candidates

    def _all_jobs_completed(self) -> bool:
        """
        所有作业是否均已完成。
        """
        return all(j.finished for j in self.jobs)

    def _current_blocking_count(self) -> int:
        """
        当前处于 blocked 状态的机器数量，用于 reward shaping。
        """
        cnt = 0
        for stage_machines in self.machines:
            for m in stage_machines:
                if m.status == "blocked":
                    cnt += 1
        return cnt

    def _current_average_buffer_occupancy(self) -> float:
        """
        当前平均缓冲占用率：
            mean_k ( len(queue_k) / capacity_k )，若没有缓冲则为 0。
        """
        if not self.buffers:
            return 0.0
        ratios: List[float] = []
        for buf in self.buffers:
            if buf.capacity > 0:
                ratios.append(len(buf.queue) / float(buf.capacity))
        if not ratios:
            return 0.0
        return float(np.mean(ratios))

    def _compute_makespan(self) -> float:
        """
        计算当前 makespan：
            若所有作业均有 completion_time，则取最大；
            否则回退为当前 self.time。
        """
        completion_times = [j.completion_time for j in self.jobs if j.finished]
        if completion_times:
            return float(max(completion_times))
        return float(self.time)

    def _get_processing_time(self, job_id: int, stage: int, machine_idx: int) -> float:
        """
        获取 job 在某 stage、某台机器上的加工时间。

        这里直接使用 InstanceData.jobs[j].ops[stage].proc_times[machine_idx]，
        不依赖 InstanceData 上的 get_processing_time 扩展方法，保持解耦。
        """
        op = self.instance.jobs[job_id].ops[stage]
        # 容错：若 machine_idx 超出 proc_times 列表长度，则使用列表最后一个元素
        if not op.proc_times:
            return 0.0
        if 0 <= machine_idx < len(op.proc_times):
            return float(op.proc_times[machine_idx])
        return float(op.proc_times[-1])
    
# ============================================================
# 规则调度通路：给上层 / baseline 使用
# ============================================================

def _select_job_by_rule_for_core(
    core: "FlowShopCoreEnv",
    instance: InstanceData,
    stage: int,
    ready_jobs: List[int],
    job_rule: str,
) -> int:
    """
    根据 job_rule 在 ready_jobs 中选择一个 job_id。
    支持规则：
        - "fifo"   : 先进先出
        - "spt"    : 当前工序加工时间最短
        - "lpt"    : 当前工序加工时间最长
        - "srpt"   : 剩余总加工时间最短
        - "random" : 随机
    """
    if not ready_jobs:
        raise ValueError("ready_jobs 为空，无法选择 job。")

    rule = job_rule.lower()

    # 1) FIFO
    if rule == "fifo":
        return ready_jobs[0]

    # 2) SPT / LPT：看当前 stage 的加工时间
    if rule in ("spt", "lpt"):
        best_job = None
        best_val: float | None = None

        for job_id in ready_jobs:
            op = instance.jobs[job_id].ops[stage]
            if not op.proc_times:
                proc = 0.0
            else:
                proc = float(min(op.proc_times))  # 当前 stage, 取最短机时作为代表

            if best_val is None:
                best_job = job_id
                best_val = proc
            else:
                if rule == "spt":
                    if proc < best_val:
                        best_job = job_id
                        best_val = proc
                else:  # "lpt"
                    if proc > best_val:
                        best_job = job_id
                        best_val = proc

        assert best_job is not None
        return best_job

    # 3) SRPT：从当前 stage 到最后一个 stage 的剩余总加工时间
    if rule == "srpt":
        num_stages = core.num_stages

        def remaining_time(job_id: int) -> float:
            total = 0.0
            for s in range(stage, num_stages):
                op = instance.jobs[job_id].ops[s]
                if not op.proc_times:
                    continue
                total += float(min(op.proc_times))
            return total

        best_job = None
        best_val: float | None = None
        for job_id in ready_jobs:
            val = remaining_time(job_id)
            if best_val is None or val < best_val:
                best_job = job_id
                best_val = val

        assert best_job is not None
        return best_job

    # 4) Random：退路
    # FlowShopCoreEnv 内部已经有 self.rng = random.Random(seed)
    return core.rng.choice(ready_jobs)


def simulate_instance_with_job_rule(
    instance: InstanceData,
    buffers: Sequence[int],
    job_rule: str,
    machine_rule: str = "min_index",
    max_steps: int = 10_000,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    使用纯规则调度（不依赖 DQN）对一个 instance + buffers 做一次完整仿真。

    参数：
        instance: InstanceData 算例。
        buffers:  长度为 K-1 的缓冲容量列表。
        job_rule:
            "fifo" / "spt" / "lpt" / "srpt" / "random"
        machine_rule:
            传给 FlowShopCoreEnv 的 dispatch_machine_rule：
                "min_index" / "ect" / "random" / "earliest_idle"
        max_steps:
            安全上限，防止异常死循环。
        seed:
            传给 FlowShopCoreEnv 的随机种子。

    返回：
        {
            "makespan": float,
            "deadlock": bool,
            "avg_blocking": float,
            "avg_buffer_occupancy": float,
            "num_steps": int,
        }
    """
    core = FlowShopCoreEnv(
        instance=instance,
        buffers=buffers,
        dispatch_machine_rule=machine_rule,
        seed=seed,
    )
    core.reset()

    done = False
    steps = 0
    last_info: Dict[str, Any] = {}

    sum_blocking = 0.0
    sum_buf_occ = 0.0
    cnt = 0

    while (not done) and (steps < max_steps):
        stage = core.current_stage

        # 如果当前没有可派工 stage，交给 step(-1) 推进时间
        if stage is None:
            core, done, info = core.step(action_job_id=-1)
        else:
            # 取当前 stage 的 ready_jobs
            if stage == 0:
                ready_jobs = list(core.stage0_queue)
            else:
                buf = core.buffers[stage - 1]
                ready_jobs = list(buf.queue)

            if not ready_jobs:
                # 理论上不常见，保守起见仍然推进一次
                core, done, info = core.step(action_job_id=-1)
            else:
                job_id = _select_job_by_rule_for_core(
                    core=core,
                    instance=instance,
                    stage=stage,
                    ready_jobs=ready_jobs,
                    job_rule=job_rule,
                )
                core, done, info = core.step(action_job_id=job_id)

        steps += 1
        last_info = info

        sum_blocking += float(info.get("blocking_count", 0.0))
        sum_buf_occ += float(info.get("avg_buffer_occupancy", 0.0))
        cnt += 1

    # === 收尾 ===
    if cnt > 0:
        avg_blocking = sum_blocking / float(cnt)
        avg_buf_occ = sum_buf_occ / float(cnt)
    else:
        avg_blocking = 0.0
        avg_buf_occ = 0.0

    deadlock = False
    makespan: float
    if last_info:
        makespan = float(last_info.get("makespan", core._compute_makespan()))
        deadlock = bool(last_info.get("deadlock", False))
    else:
        makespan = float(core._compute_makespan())

    return {
        "makespan": makespan,
        "deadlock": deadlock,
        "avg_blocking": avg_blocking,
        "avg_buffer_occupancy": avg_buf_occ,
        "num_steps": steps,
    }


# ============================
# 文件：src/instances/generators.py
# 说明：
#   - 定义随机 flow shop / job shop 算例的生成配置
#   - 提供生成单个 / 批量 InstanceData 的函数
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .types import InstanceData, Job, Operation


@dataclass
class FlowShopGeneratorConfig:
    """
    随机算例生成配置（flow shop / flexible flow shop 简化版）

    字段说明：
        num_jobs:            工件数量 J
        num_stages:          工序阶段数量 K
        machines_per_stage:  每阶段并行机数量（长度 = K），如未提供则按
                             [min_machines_per_stage, max_machines_per_stage] 随机生成
        min_machines_per_stage, max_machines_per_stage:
                             当 machines_per_stage 为 None 时，用于随机生成各阶段设备数
        proc_time_low, proc_time_high:
                             加工时间整数区间 [low, high]，包含两端（单位任意）
        same_proc_time_across_machines:
                             若为 True，则同一阶段的所有并机加工时间相同；
                             若为 False，则每台并机独立采样加工时间。
        seed:                随机种子（可选）
    """

    num_jobs: int
    num_stages: int

    machines_per_stage: Optional[Sequence[int]] = None
    min_machines_per_stage: int = 1
    max_machines_per_stage: int = 1

    proc_time_low: int = 1
    proc_time_high: int = 30

    same_proc_time_across_machines: bool = True

    seed: Optional[int] = None

    def make_rng(self) -> np.random.Generator:
        """根据 seed 生成一个 numpy 随机数发生器。"""
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed)


def _build_machines_per_stage(cfg: FlowShopGeneratorConfig, rng: np.random.Generator) -> List[int]:
    """
    根据配置生成每阶段并行机数量列表。
    若 cfg.machines_per_stage 不为 None，则直接转为 list 返回；
    否则在 [min_machines_per_stage, max_machines_per_stage] 上为每阶段采样一个整数。
    """
    if cfg.machines_per_stage is not None:
        mps = list(cfg.machines_per_stage)
        assert len(mps) == cfg.num_stages, "machines_per_stage 长度必须等于 num_stages"
        return [int(m) for m in mps]

    assert cfg.min_machines_per_stage <= cfg.max_machines_per_stage, \
        "min_machines_per_stage 应不大于 max_machines_per_stage"

    return [
        int(rng.integers(cfg.min_machines_per_stage, cfg.max_machines_per_stage + 1))
        for _ in range(cfg.num_stages)
    ]


def generate_random_instance(
    cfg: FlowShopGeneratorConfig,
    rng: Optional[np.random.Generator] = None,
) -> InstanceData:
    """
    根据 FlowShopGeneratorConfig 生成一个随机 InstanceData。

    生成逻辑（简化 flow shop）：
      - 对每个 job j = 0..J-1：
        - 对每个 stage k = 0..K-1：
          - 若 same_proc_time_across_machines = True：
              采样一个整数 p ~ U[proc_time_low, proc_time_high]，
              然后该阶段所有并机加工时间都设为 p；
            否则：
              对该阶段每台并机独立采样加工时间。
    """
    if rng is None:
        rng = cfg.make_rng()

    J = cfg.num_jobs
    K = cfg.num_stages
    mps: List[int] = _build_machines_per_stage(cfg, rng)

    jobs: List[Job] = []

    for j in range(J):
        ops: List[Operation] = []
        for k in range(K):
            M_k = mps[k]
            if cfg.same_proc_time_across_machines:
                p = int(rng.integers(cfg.proc_time_low, cfg.proc_time_high + 1))
                proc_times = [float(p)] * M_k
            else:
                # 为该阶段的每台机器独立采样加工时间
                ps = rng.integers(cfg.proc_time_low, cfg.proc_time_high + 1, size=M_k)
                proc_times = [float(v) for v in ps]

            ops.append(Operation(stage_id=k, proc_times=proc_times))

        jobs.append(Job(job_id=j, ops=ops))

    return InstanceData(jobs=jobs, machines_per_stage=mps)


def generate_random_instances(
    cfg: FlowShopGeneratorConfig,
    n_instances: int,
    base_seed: Optional[int] = None,
) -> List[InstanceData]:
    """
    批量生成随机算例。

    参数：
        cfg:          FlowShopGeneratorConfig，描述单个实例的规模和分布
        n_instances:  要生成的实例数量
        base_seed:    若不为 None，则在此基础上为每个实例派生子种子，保证可复现性

    返回：
        List[InstanceData]
    """
    instances: List[InstanceData] = []

    if base_seed is not None:
        master_rng = np.random.default_rng(base_seed)
        seeds = master_rng.integers(0, 2**32 - 1, size=n_instances)
        for s in seeds:
            sub_cfg = FlowShopGeneratorConfig(
                num_jobs=cfg.num_jobs,
                num_stages=cfg.num_stages,
                machines_per_stage=cfg.machines_per_stage,
                min_machines_per_stage=cfg.min_machines_per_stage,
                max_machines_per_stage=cfg.max_machines_per_stage,
                proc_time_low=cfg.proc_time_low,
                proc_time_high=cfg.proc_time_high,
                same_proc_time_across_machines=cfg.same_proc_time_across_machines,
                seed=int(s),
            )
            instances.append(generate_random_instance(sub_cfg))
    else:
        # 不指定 base_seed 时，直接用 cfg 的 seed 或全局随机
        rng = cfg.make_rng()
        for _ in range(n_instances):
            # 每次修改 cfg.seed 以避免完全相同的实例
            sub_seed = int(rng.integers(0, 2**32 - 1))
            sub_cfg = FlowShopGeneratorConfig(
                num_jobs=cfg.num_jobs,
                num_stages=cfg.num_stages,
                machines_per_stage=cfg.machines_per_stage,
                min_machines_per_stage=cfg.min_machines_per_stage,
                max_machines_per_stage=cfg.max_machines_per_stage,
                proc_time_low=cfg.proc_time_low,
                proc_time_high=cfg.proc_time_high,
                same_proc_time_across_machines=cfg.same_proc_time_across_machines,
                seed=sub_seed,
            )
            instances.append(generate_random_instance(sub_cfg))

    return instances

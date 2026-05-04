# ============================
# 文件：src/instance/io.py
# 说明：算例的读写与构造入口
#  - from_json:    兼容旧 JSON 格式，构造 InstanceData
#  - from_matrix_csv: 从矩阵 CSV + 设备信息构造 InstanceData
#  - 内部使用 _infer_mps_from_devices_csv / _matrix_to_instance
#  - 在末尾为 InstanceData 挂载 get_processing_time(j, k, m) 方法
# ============================

import json                                   # 兼容旧 JSON 格式（保留）
from typing import List, Optional             # 类型注解


import os
import re
import pickle
from pathlib import Path

import numpy as np                            # 数组处理
import pandas as pd                           # CSV 读取

from .types import InstanceData, Job, Operation   # 本项目内导入数据结构


def from_json(path: str) -> InstanceData:
    """
    从 JSON 文件构造 InstanceData（兼容旧流程）

    期望 JSON 结构大致为：
    {
        "jobs": [
            {
                "job_id": 0,
                "ops": [
                    {"stage_id": 0, "proc_times": [10, 10, 10]},
                    {"stage_id": 1, "proc_times": [12, 12, 12]},
                    ...
                ]
            },
            ...
        ],
        "machines_per_stage": [3, 3, ...]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs: List[Job] = []
    for jd in data["jobs"]:
        ops = [
            Operation(stage_id=o["stage_id"], proc_times=o["proc_times"])
            for o in jd["ops"]
        ]
        jobs.append(Job(job_id=jd["job_id"], ops=ops))

    return InstanceData(jobs=jobs, machines_per_stage=data["machines_per_stage"])


def _infer_mps_from_devices_csv(devices_csv: str) -> List[int]:
    """
    从两行设备矩阵推断每阶段设备数（行1=阶段1..S；行2=全局设备1..K，1-based）

    设备矩阵 devices_csv 形如（无表头）：
        row 0:  stage index (1..S) for each global machine
        row 1:  global machine id (1..M_tot)  [通常没什么用，只要第一行]

    返回值：
        machines_per_stage: List[int]，长度为阶段数 S。
    """
    dev = pd.read_csv(devices_csv, header=None).values
    stage_row = dev[0, :]          # 第一行：阶段编号（1-based）
    S = int(stage_row.max())       # 阶段总数
    mps = [int(np.sum(stage_row == (s + 1))) for s in range(S)]
    return mps


def _matrix_to_instance(job_matrix: np.ndarray, mps: List[int]) -> InstanceData:
    """
    将 [job_id, stage1..stageS] 矩阵 + machines_per_stage 转为 InstanceData。

    约定：
      - job_matrix 形状为 [J, S+1]
        第 0 列：job_id
        第 1..S 列：各阶段在“同质并机”的单机加工时间（整数）
      - 对于同一阶段的并行机，假定同质：直接复制相同加工时间到该阶段的所有机器上。
    """
    J, cols = job_matrix.shape
    S = cols - 1
    assert len(mps) == S, "machines_per_stage 长度需等于阶段数"

    jobs: List[Job] = []
    for r in range(J):
        job_id = int(job_matrix[r, 0])
        ops: List[Operation] = []
        for s in range(S):
            pt = float(job_matrix[r, s + 1])   # 单机加工时间
            m_s = int(mps[s])                  # 该阶段并行机数量
            # 同质并机：该阶段所有机器加工时间相同
            ops.append(Operation(stage_id=s, proc_times=[pt] * m_s))
        jobs.append(Job(job_id=job_id, ops=ops))

    return InstanceData(jobs=jobs, machines_per_stage=mps)


def from_matrix_csv(
    matrix_csv: str,
    machines_per_stage: Optional[List[int]] = None,
    devices_csv: Optional[str] = None,
) -> InstanceData:
    """
    从算例矩阵 CSV 直接构建 InstanceData（跳过 JSON）。

    参数：
        matrix_csv:
            - 形如：列 = [job_id, stage1, stage2, ..., stageS]
            - 每一行是一道工件，其在各阶段的“单机加工时间”（同质并机）

        machines_per_stage:
            - 可选；若不为 None，则直接使用该列表作为每阶段设备数。

        devices_csv:
            - 可选；若 machines_per_stage 为 None，则必须提供：
              devices_csv 的第一行给出每台设备所在的阶段编号（1-based），
              函数会通过计数方式推断各阶段设备数。

    返回：
        InstanceData 对象，包含 jobs 列表与 machines_per_stage。
    """
    df = pd.read_csv(matrix_csv)
    job_matrix = df.values

    if machines_per_stage is None:
        assert (
            devices_csv is not None
        ), "未提供 machines_per_stage 时必须提供 devices_csv 以推断设备数"
        machines_per_stage = _infer_mps_from_devices_csv(devices_csv)

    return _matrix_to_instance(job_matrix, machines_per_stage)


# ============================================================
# 为 InstanceData 补充一个 get_processing_time(j, k, m) 方法
# 方便调度环境根据 (job, stage, global_machine_id) 访问加工时间。
# ============================================================

from .types import InstanceData as _Inst


def _inst_get_processing_time(self: _Inst, j: int, k: int, m: Optional[int] = None) -> float:
    """
    获取作业 j 在阶段 k、全局机器 m 上的加工时间。

    约定：
      - self.jobs[j].ops[k].proc_times 是一个长度为 machines_per_stage[k] 的列表，
        表示该阶段每台并行机的加工时间（同质并机时各元素相同）。
      - m:
        * 若为 None，则默认取该阶段的第 0 号机器的加工时间；
        * 若为全局机器编号（0-based），
          会先根据 machines_per_stage 把其映射为阶段内的局部机器号。
    """
    mps = self.machines_per_stage

    if m is None:
        # 不区分具体机器，默认取阶段内第一个并机的加工时间
        m_local = 0
    else:
        # 将全局机器编号映射为阶段内局部机器号
        acc = 0
        m_local = 0
        for kk, Mk in enumerate(mps):
            if kk == k:
                m_local = m - acc
                break
            acc += Mk

    return float(self.jobs[j].ops[k].proc_times[m_local])

# ---- 锚点 2：新增，从路径名中解析 "m3" 这样的机器数 ----
def _parse_m_from_path(path: Path) -> Optional[int]:
    """
    从路径中解析形如 'm3' 的片段（例如 'j50s3m3'），返回 m 值。
    若未找到则返回 None。
    """
    pattern = re.compile(r"m(\d+)")
    for part in path.parts:
        m = pattern.search(part)
        if m:
            return int(m.group(1))
    return None

def save_instance_pickle(path: str | Path, instance: InstanceData) -> None:
    """把单个 InstanceData 序列化到指定路径（.pkl）"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(instance, f)


def load_instance_pickle(path: str | Path) -> InstanceData:
    """从 .pkl 文件加载一个 InstanceData。"""
    path = Path(path)
    with path.open("rb") as f:
        inst = pickle.load(f)
    return inst


def load_instances_from_dir(
    dir_path: str | Path,
    override_mps_from_dirname: bool = True,
) -> List[InstanceData]:
    """
    从目录中加载所有 .pkl 实例，按文件名排序返回 List[InstanceData]。
    约定文件名形如 inst_0000.pkl, inst_0001.pkl, ...

    若 override_mps_from_dirname=True，则会尝试从目录名中解析 'm3' 这样的片段，
    并在实例原本 machines_per_stage 全为 1 的情况下，覆盖为 [m_val] * S。
    """
    dir_path = Path(dir_path)
    files = sorted(p for p in dir_path.glob("*.pkl"))
    instances: List[InstanceData] = []
    for p in files:
        instances.append(load_instance_pickle(p))

    # ---- 根据目录名自动覆盖 machines_per_stage ----
    if override_mps_from_dirname and instances:
        m_val = _parse_m_from_path(dir_path)
        if m_val is not None:
            # 先判断这些实例是不是“占位型”：machines_per_stage 全是 1
            need_override = True
            for inst in instances:
                mps = getattr(inst, "machines_per_stage", None)
                # 只要有一个不是 list 或里面有 !=1 的，就认为不是占位型
                if not isinstance(mps, list) or any((x is None) or (x != 1) for x in mps):
                    need_override = False
                    break

            if need_override:
                for inst in instances:
                    mps = getattr(inst, "machines_per_stage", None)
                    if isinstance(mps, list) and len(mps) > 0:
                        # 按原有阶段数长度覆盖
                        S = len(mps)
                        inst.machines_per_stage = [m_val] * S
                    else:
                        # 兜底：根据 jobs[0].ops 推断阶段数
                        if hasattr(inst, "jobs") and getattr(inst, "jobs"):
                            first_job = inst.jobs[0]
                            ops = getattr(first_job, "ops", [])
                            S = len(ops)
                            inst.machines_per_stage = [m_val] * S

                # 打一行 debug，方便你确认是否覆盖成功
                sample_mps = getattr(instances[0], "machines_per_stage", None)
                print(f"[DEBUG] override machines_per_stage for '{dir_path}' to {sample_mps}")
        # else: 没解析出 m 值，保持原样

    return instances




# 将方法挂到 InstanceData 类上，供 env 调用
_Inst.get_processing_time = _inst_get_processing_time

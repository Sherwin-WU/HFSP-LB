# ============================
# 文件：src/instances/types.py
# ============================
from dataclasses import dataclass                   # 导入 dataclass 定义轻量数据结构
from typing import List                              # 导入 List 类型

@dataclass
class Operation:                                     # 定义工序
    stage_id: int                                   # 阶段编号（0-based）
    proc_times: List[float]                         # 在该阶段各并行机上的加工时间（同型机可复制同一时间）

@dataclass
class Job:                                          # 定义工件
    job_id: int                                     # 工件编号（0-based）
    ops: List[Operation]                            # 工件的各阶段工序列表

@dataclass
class InstanceData:                                     # 定义算例实例
    jobs: List[Job]                                 # 作业集合
    machines_per_stage: List[int]                   # 每阶段设备数量列表，例如 [3,4,3,5,5]

@dataclass
class Allocation:                                   # 定义调度分配条目
    job_id: int                                     # 工件编号
    stage_id: int                                   # 阶段编号（0-based）
    global_machine_id: int                          # 全局设备编号（0-based）
    start: float                                    # 开始时间
    end: float                                      # 完成时间
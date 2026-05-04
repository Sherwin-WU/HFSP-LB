# ============================
# 文件：src/instances/features.py
# 说明：
#   针对 InstanceData 提供各种特征提取工具，
#   主要用于上层 MDP 的状态构造 f(I) 和分析。
#
#   主要接口：
#     - compute_processing_time_matrix
#     - compute_stage_loads
#     - compute_job_total_times
#     - build_instance_feature_dict
#     - build_instance_features      （同上别名，方便记）
#     - build_instance_feature_vector（可选，把特征压平成 1D 向量）
# ============================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import InstanceData

# 可选导入 torch：如果用户安装了 torch，就支持直接输出 tensor
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


# ============================================================
# 一些底层工具函数：从 InstanceData 提取加工时间矩阵等
# ============================================================

def compute_basic_sizes(inst: InstanceData) -> Tuple[int, int]:
    """
    计算实例的基本规模信息：
      - num_jobs:   工件数 J
      - num_stages: 阶段数 K

    这里不依赖 InstanceData 是否有 num_jobs/num_stages 属性，
    而是直接根据 jobs 和 machines_per_stage 推断，避免不兼容。
    """
    num_jobs = len(inst.jobs)

    # 阶段数可以由 machines_per_stage 推断，也可以由每个 job 的 ops 数
    if len(inst.machines_per_stage) > 0:
        num_stages = len(inst.machines_per_stage)
    elif num_jobs > 0:
        num_stages = len(inst.jobs[0].ops)
    else:
        num_stages = 0

    return num_jobs, num_stages


def compute_processing_time_matrix(inst: InstanceData) -> np.ndarray:
    """
    把 InstanceData 中的加工时间整理成一个形状为 [J, K] 的矩阵 P：

        P[j, k] = job j 在阶段 k 上的“基准加工时间”

    这里由于 InstanceData 中每个 Operation 存的是
      proc_times: List[float]（长度 = 该阶段并机数量）
    在同质并机的情况下，这个列表的元素通常相同。

    为了不强依赖“完全同质”的假设，这里取：
        P[j, k] = proc_times 的平均值。

    返回：
        P: np.ndarray, shape = [num_jobs, num_stages]
    """
    num_jobs, num_stages = compute_basic_sizes(inst)
    if num_jobs == 0 or num_stages == 0:
        return np.zeros((num_jobs, num_stages), dtype=float)

    P = np.zeros((num_jobs, num_stages), dtype=float)

    for j, job in enumerate(inst.jobs):
        # 容错：若某个 job 的 ops 少于 num_stages，则只填已有部分
        for op in job.ops:
            k = op.stage_id
            if 0 <= k < num_stages:
                if len(op.proc_times) == 0:
                    P[j, k] = 0.0
                else:
                    P[j, k] = float(np.mean(op.proc_times))

    return P


def compute_stage_loads(inst: InstanceData) -> np.ndarray:
    """
    计算每个阶段的总负荷（总加工时间和），形状 [K]：

        load_per_stage[k] = sum_j P[j,k]

    其中 P[j,k] 为 compute_processing_time_matrix 返回的基准加工时间。
    """
    P = compute_processing_time_matrix(inst)
    if P.size == 0:
        return np.zeros((P.shape[1],), dtype=float)
    # axis=0：对行 j 求和，得到每个 k 的总负荷
    load_per_stage = P.sum(axis=0)
    return load_per_stage


def compute_stage_loads_per_machine(inst: InstanceData) -> np.ndarray:
    """
    计算“每台机器”的阶段负荷：对每个阶段 k，做

        load_per_machine[k] = load_per_stage[k] / machines_per_stage[k]

    当某阶段没有机器（m_k = 0）时，负荷置 0。
    """
    load_per_stage = compute_stage_loads(inst)
    _, num_stages = compute_basic_sizes(inst)

    if num_stages == 0:
        return np.zeros((0,), dtype=float)

    mps = np.asarray(inst.machines_per_stage, dtype=float)
    if mps.shape[0] != num_stages:
        # 容错：长度不符时，截断/补齐
        tmp = np.zeros((num_stages,), dtype=float)
        n = min(num_stages, mps.shape[0])
        tmp[:n] = mps[:n]
        mps = tmp

    load_per_machine = np.zeros_like(load_per_stage)
    for k in range(num_stages):
        mk = mps[k]
        if mk > 0:
            load_per_machine[k] = load_per_stage[k] / mk
        else:
            load_per_machine[k] = 0.0

    return load_per_machine


def compute_job_total_times(inst: InstanceData) -> np.ndarray:
    """
    计算每个 job 的总加工时间（跨所有阶段相加），shape = [J]：

        job_total[j] = sum_k P[j,k]

    其中 P[j,k] 为 compute_processing_time_matrix 返回的基准加工时间。
    """
    P = compute_processing_time_matrix(inst)
    if P.size == 0:
        return np.zeros((P.shape[0],), dtype=float)
    job_total = P.sum(axis=1)
    return job_total


# ============================================================
# 对外的特征构造接口
# ============================================================

def build_instance_feature_dict(
    inst: InstanceData,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    构建一个“字典形式”的算例特征集合，主要用于上层 MDP。

    返回的 dict 中包含：
        - "num_jobs":              float 标量
        - "num_stages":            float 标量
        - "machines_per_stage":    np.ndarray, shape [K]
        - "total_machines":        float 标量
        - "stage_loads":           np.ndarray, shape [K]
        - "stage_loads_per_machine": np.ndarray, shape [K]
        - "job_total_times":       np.ndarray, shape [J]

    如果 normalize=True，会额外提供一些简单归一化后的字段（后缀 "_norm"）：
        - "stage_loads_norm":           stage_loads / max(stage_loads + eps)
        - "stage_loads_per_machine_norm"
        - "job_total_times_norm"

    注意：
    - 这里返回的是 numpy 数组/标量，不直接依赖 torch；
      若需要 tensor 形式，可以在上层调用者中自行转换，
      或使用下方 build_instance_features(as_tensor=True)。
    """
    num_jobs, num_stages = compute_basic_sizes(inst)

    machines_per_stage = np.asarray(inst.machines_per_stage, dtype=float)
    if machines_per_stage.shape[0] != num_stages:
        # 容错处理：长度不一致时截断或补齐 0
        tmp = np.zeros((num_stages,), dtype=float)
        n = min(num_stages, machines_per_stage.shape[0])
        tmp[:n] = machines_per_stage[:n]
        machines_per_stage = tmp

    total_machines = float(machines_per_stage.sum())

    stage_loads = compute_stage_loads(inst)
    stage_loads_per_machine = compute_stage_loads_per_machine(inst)
    job_total_times = compute_job_total_times(inst)

    feats: Dict[str, Any] = {
        "num_jobs": float(num_jobs),
        "num_stages": float(num_stages),
        "machines_per_stage": machines_per_stage,
        "total_machines": total_machines,
        "stage_loads": stage_loads,
        "stage_loads_per_machine": stage_loads_per_machine,
        "job_total_times": job_total_times,
    }

    if normalize:
        eps = 1e-8

        if num_stages > 0:
            max_stage_load = float(np.max(stage_loads)) if stage_loads.size > 0 else 1.0
            max_stage_load = max(max_stage_load, eps)

            max_stage_load_pm = (
                float(np.max(stage_loads_per_machine))
                if stage_loads_per_machine.size > 0 else 1.0
            )
            max_stage_load_pm = max(max_stage_load_pm, eps)
        else:
            max_stage_load = 1.0
            max_stage_load_pm = 1.0

        if num_jobs > 0:
            max_job_total = float(np.max(job_total_times)) if job_total_times.size > 0 else 1.0
            max_job_total = max(max_job_total, eps)
        else:
            max_job_total = 1.0

        feats["stage_loads_norm"] = stage_loads / max_stage_load
        feats["stage_loads_per_machine_norm"] = stage_loads_per_machine / max_stage_load_pm
        feats["job_total_times_norm"] = job_total_times / max_job_total

    return feats


def _to_torch_if_needed(
    arr: Any,
    as_tensor: bool,
    device: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,  # type: ignore[name-defined]
) -> Any:
    """
    内部工具：如果 as_tensor=True 且 torch 可用，把 numpy/标量转为 torch.tensor。
    """
    if not as_tensor:
        return arr

    if not TORCH_AVAILABLE:
        raise RuntimeError("build_instance_features(as_tensor=True) 需要安装 PyTorch。")

    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    elif isinstance(arr, (float, int)):
        t = torch.tensor(arr)
    else:
        # 其他类型先不动，由调用方自行处理
        return arr

    if dtype is not None:
        t = t.to(dtype=dtype)  # type: ignore[assignment]
    if device is not None:
        t = t.to(device)       # type: ignore[assignment]
    return t


def build_instance_features(
    inst: InstanceData,
    normalize: bool = True,
    as_tensor: bool = False,
    device: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,  # type: ignore[name-defined]
) -> Dict[str, Any]:
    """
    主接口（推荐用）：构建一个“字典形式”的算例特征集合。

    参数：
        inst:       InstanceData
        normalize:  是否对阶段负荷、工件总加工时间做简单归一化
        as_tensor:  若 True 且环境安装了 torch，则将标量和 np.ndarray 转为 torch.Tensor
        device:     当 as_tensor=True 时，可指定 tensor 所在设备（如 "cpu"/"cuda"）
        dtype:      当 as_tensor=True 时，可指定 tensor 类型（如 torch.float32）

    返回：
        一个 dict，key 与 build_instance_feature_dict 一致；如果 as_tensor=True，
        值中的标量和 np.ndarray 会被转成 torch.Tensor。
    """
    feats_np = build_instance_feature_dict(inst, normalize=normalize)
    if not as_tensor:
        return feats_np

    feats: Dict[str, Any] = {}
    for k, v in feats_np.items():
        feats[k] = _to_torch_if_needed(v, as_tensor=True, device=device, dtype=dtype)
    return feats


def build_instance_feature_vector(
    inst: InstanceData,
    normalize: bool = True,
    as_tensor: bool = False,
    device: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,  # type: ignore[name-defined]
) -> Any:
    """
    可选接口：将部分全局/聚合特征压平成一个 1D 向量，用于简单模型。

    当前 vector 拼接内容：
        [ num_jobs,
          num_stages,
          total_machines,
          mean(machines_per_stage),
          std(machines_per_stage),
          min(machines_per_stage),
          max(machines_per_stage),
          mean(stage_loads_per_machine_norm),
          std(stage_loads_per_machine_norm) ]

    若需要更复杂的向量特征，可以自行在此基础上扩展。

    返回：
        若 as_tensor=False: np.ndarray, shape [D]
        若 as_tensor=True:  torch.Tensor, shape [D]
    """
    feats = build_instance_feature_dict(inst, normalize=normalize)

    num_jobs = feats["num_jobs"]
    num_stages = feats["num_stages"]
    total_machines = feats["total_machines"]

    mps = feats["machines_per_stage"]
    if isinstance(mps, np.ndarray) and mps.size > 0:
        mean_mps = float(mps.mean())
        std_mps = float(mps.std())
        min_mps = float(mps.min())
        max_mps = float(mps.max())
    else:
        mean_mps = std_mps = min_mps = max_mps = 0.0

    slpm_norm = feats.get("stage_loads_per_machine_norm", None)
    if isinstance(slpm_norm, np.ndarray) and slpm_norm.size > 0:
        mean_slpm = float(slpm_norm.mean())
        std_slpm = float(slpm_norm.std())
    else:
        mean_slpm = std_slpm = 0.0

    vec_np = np.array(
        [
            num_jobs,
            num_stages,
            total_machines,
            mean_mps,
            std_mps,
            min_mps,
            max_mps,
            mean_slpm,
            std_slpm,
        ],
        dtype=float,
    )

    if not as_tensor:
        return vec_np

    return _to_torch_if_needed(vec_np, as_tensor=True, device=device, dtype=dtype)

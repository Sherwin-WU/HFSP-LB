# ============================
# 文件：envs/constraints.py
# 说明：
#   限定与缓冲容量相关的约束和工具函数。
#
#   核心约束：
#       对每一段缓冲 k（位于阶段 k 和 k+1 之间）：
#           0 ≤ b_k ≤ a_k，  a_k ≤ max{m(k), m(k+1)}
#
#   对外提供：
#       - compute_buffer_upper_bounds(instance) -> List[int]
#       - project_buffers_to_feasible(b, a, total_limit=None) -> List[int]
#       - check_buffers_feasible(b, a, total_limit=None) -> (bool, str)
# ============================

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from instances.types import InstanceData


def compute_buffer_upper_bounds(instance: InstanceData) -> List[int]:
    """
    计算每个中间缓冲段的上界 a_k。

    当前默认规则：
        对阶段 k 与 k+1 之间的缓冲：
            a_k = max{ m(k), m(k+1) }
        其中 m(k) 为阶段 k 的并行机数量。

    返回：
        upper_bounds: List[int]，长度 = num_stages - 1。
                      若阶段数 <= 1，则返回空列表。
    """
    mps = list(instance.machines_per_stage)
    num_stages = len(mps)
    if num_stages <= 1:
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
    将候选缓冲向量 b 投影到满足约束的可行集合：

        0 ≤ b_k ≤ a_k， 对所有 k；
        若 total_limit 不为 None，则再要求 sum(b_k) ≤ total_limit。

    实现策略：
        1) 先逐元素裁剪到区间 [0, a_k]；
        2) 若给定 total_limit 且 sum(b) > total_limit：
           - 使用简单比例缩放：
             b_k ← floor( b_k * total_limit / sum(b_k) )。

    参数：
        b: 候选缓冲向量，任意整数/浮点数序列；
        a: 各段缓冲上界 a_k，长度必须与 b 相同；
        total_limit: 可选的全局缓冲总量上界。

    返回：
        proj: List[int]，满足上述约束的缓冲向量。
    """
    if len(b) != len(a):
        raise ValueError(
            f"project_buffers_to_feasible: b 和 a 长度不一致，"
            f"len(b)={len(b)}, len(a)={len(a)}"
        )

    # 逐元素裁剪到 [0, a_k]
    proj = [max(0, min(int(bk), int(ak))) for bk, ak in zip(b, a)]

    if total_limit is not None:
        s = sum(proj)
        if s > total_limit and s > 0:
            ratio = float(total_limit) / float(s)
            # 按比例缩放并向下取整
            proj = [int(np.floor(bk * ratio)) for bk in proj]

    return proj


def check_buffers_feasible(
    b: Sequence[int],
    a: Sequence[int],
    total_limit: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    检查缓冲向量 b 是否满足给定约束。

        0 ≤ b_k ≤ a_k，
        若 total_limit 不为 None，则 sum(b_k) ≤ total_limit。

    返回：
        (ok, msg)
        - ok: bool，表示是否可行；
        - msg: 字符串，若可行则为 "OK"，否则给出首个违背约束的原因。
    """
    if len(b) != len(a):
        return False, f"len(b)={len(b)} 与 len(a)={len(a)} 不一致"

    for k, (bk, ak) in enumerate(zip(b, a)):
        if bk < 0:
            return False, f"b[{k}]={bk} < 0"
        if bk > ak:
            return False, f"b[{k}]={bk} > a[{k}]={ak}"

    if total_limit is not None:
        s = sum(b)
        if s > total_limit:
            return False, f"sum(b)={s} > total_limit={total_limit}"

    return True, "OK"

# ============================
# 文件：envs/observations/buffer_obs_dense.py
# 说明：
#   为上层缓冲设计环境（BufferDesignEnv）构造稠密状态向量。
#
#   结构（拼接顺序）：
#
#     obs = concat(
#         instance_features,      # f(I)
#         upper_bounds_norm,      # a_norm[0..K-2]
#         selected_buffers_norm,  # b_norm[0..K-2]
#         selected_mask,          # mask[0..K-2]
#         current_index_one_hot,  # one_hot[0..K-2]
#     )
#
#   其中：
#     - K          = num_stages
#     - num_buf    = K - 1
#     - f(I)       = build_instance_feature_vector(instance, normalize=True)
#
#   该实现与 envs.buffer_design_env.BufferDesignEnv._build_obs 中的逻辑保持一致，
#   只是抽取出来作为一个独立的工具函数，便于在其它地方复用或做单元测试。
# ============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any

import numpy as np

from instances.features import build_instance_feature_vector

if TYPE_CHECKING:
    from envs.buffer_design_env import BufferDesignEnv


@dataclass
class BufferObsConfig:
    """
    上层缓冲设计状态编码的配置。

    当前主要提供一个开关，控制是否对实例特征进行归一化；
    如需做更多消融，可以在这里扩展配置项。
    """
    normalize_instance_features: bool = True


def build_buffer_obs(
    buffer_env: "BufferDesignEnv",
    cfg: Optional[BufferObsConfig] = None,
) -> np.ndarray:
    """
    基于 BufferDesignEnv 构造上层 DQN 的状态向量。

    参数：
        buffer_env:
            上层缓冲设计环境实例（envs.buffer_design_env.BufferDesignEnv）。
            要求其内部字段符合之前定义：
                - _current_instance: 当前 InstanceData
                - _upper_bounds:     np.ndarray[int], shape [num_buffers]
                - _buffers:          np.ndarray[int], shape [num_buffers]，未选为 -1
                - _num_buffers:      int
                - _current_buf_index:int，当前要决策的缓冲段索引
                - _done:             bool，episode 是否结束
        cfg:
            BufferObsConfig，控制实例特征是否归一化等；
            若为 None，则使用默认配置（normalize_instance_features=True）。

    返回：
        obs: np.ndarray, dtype=float32, shape = [D]
             按顺序：
                [ f(I),
                  upper_bounds_norm,
                  selected_buffers_norm,
                  selected_mask,
                  current_index_one_hot ]
    """
    if cfg is None:
        cfg = BufferObsConfig()

    # -----------------------------
    # 1. 取出内部状态
    # -----------------------------
    current_instance = getattr(buffer_env, "_current_instance", None)
    if current_instance is None:
        raise RuntimeError(
            "build_buffer_obs: buffer_env._current_instance 为 None，"
            "请确保在调用前先执行 BufferDesignEnv.reset()。"
        )

    upper_bounds = getattr(buffer_env, "_upper_bounds", None)
    buffers = getattr(buffer_env, "_buffers", None)
    num_buffers = int(getattr(buffer_env, "_num_buffers", 0))
    current_buf_index = int(getattr(buffer_env, "_current_buf_index", 0))
    done = bool(getattr(buffer_env, "_done", False))

    if upper_bounds is None or buffers is None:
        raise RuntimeError(
            "build_buffer_obs: buffer_env._upper_bounds 或 _buffers 为 None，"
            "请确保在调用前先执行 BufferDesignEnv.reset()。"
        )

    upper_bounds = np.asarray(upper_bounds, dtype=np.float32).reshape(-1)
    buffers = np.asarray(buffers, dtype=np.float32).reshape(-1)

    if num_buffers <= 0 or num_buffers != upper_bounds.shape[0] or num_buffers != buffers.shape[0]:
        raise ValueError(
            f"build_buffer_obs: num_buffers={num_buffers} 与 upper_bounds/buffers 长度不一致，"
            f"upper_bounds.shape={upper_bounds.shape}, buffers.shape={buffers.shape}"
        )

    # -----------------------------
    # 2. 实例特征 f(I)
    # -----------------------------
    feat_vec = build_instance_feature_vector(
        current_instance,
        normalize=cfg.normalize_instance_features,
        as_tensor=False,
    )  # np.ndarray, shape [D0]
    feat_vec = feat_vec.astype(np.float32).ravel()

    # -----------------------------
    # 3. 缓冲上界归一化 upper_bounds_norm
    #    upper_bounds_norm[k] = a_k / max_a
    # -----------------------------
    if num_buffers > 0:
        max_a = float(np.max(upper_bounds))
        if max_a < 1.0:
            max_a = 1.0
        upper_bounds_norm = upper_bounds / max_a
    else:
        upper_bounds_norm = np.zeros((0,), dtype=np.float32)

    # -----------------------------
    # 4. 已选缓冲归一化 selected_buffers_norm
    #    若 b_k >= 0 且 a_k > 0，则 b_norm[k] = b_k / a_k，否则为 0.
    # -----------------------------
    selected_mask = (buffers >= 0.0).astype(np.float32)

    selected_buffers_norm = np.zeros_like(buffers, dtype=np.float32)
    for k in range(num_buffers):
        bk = float(buffers[k])
        ak = float(upper_bounds[k])
        if bk >= 0.0 and ak > 0.0:
            selected_buffers_norm[k] = bk / ak
        else:
            selected_buffers_norm[k] = 0.0

    # -----------------------------
    # 5. 当前索引 one-hot
    # -----------------------------
    current_index_one_hot = np.zeros((num_buffers,), dtype=np.float32)
    if (not done) and (0 <= current_buf_index < num_buffers):
        current_index_one_hot[current_buf_index] = 1.0

    # -----------------------------
    # 6. 拼接所有部分
    # -----------------------------
    obs = np.concatenate(
        [
            feat_vec,
            upper_bounds_norm,
            selected_buffers_norm,
            selected_mask,
            current_index_one_hot,
        ],
        axis=0,
    )

    return obs.astype(np.float32)

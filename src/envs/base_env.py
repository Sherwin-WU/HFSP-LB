# ============================
# 文件：envs/base_env.py
# 说明：
#   为本项目中的环境定义一个统一的基类接口，
#   方便 ShopEnv、BufferDesignEnv 等环境统一对接 RL 算法。
# ============================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseEnv(ABC):
    """
    本项目环境的抽象基类。

    所有环境必须实现：
        - reset() -> obs
        - step(action) -> (obs_next, reward, done, info)

    其中：
        - obs / obs_next: 观测，一般是 np.ndarray 或可转为 torch.Tensor 的结构；
        - reward: float 型单步奖励；
        - done: bool，表示 episode 是否结束；
        - info: dict，用于携带调试/统计信息，不参与学习目标。
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self) -> Any:
        """
        重置环境到初始状态，并返回初始观测 obs。
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        执行一个动作，推进环境一步。

        参数：
            action: 任意类型，具体由子类定义（在本项目中通常为 int 的离散动作编号）。

        返回：
            obs_next: 下一时刻观测
            reward:   单步奖励（float）
            done:     是否 episode 结束
            info:     附加信息字典
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> None:
        """
        设置随机种子。默认实现为空，由需要随机性的子类重写。
        """
        return

# ============================
# 文件：envs/shop_env.py
# 作用：
#   下层“分配 agent”的环境封装，用于 DQN/DDQN/D3QN。
#
# 设计思想：
#   - 不在这个文件里硬编码 FFS 的离散事件仿真；
#   - 你可以把任何“核心车间环境(core_env)”塞进来，只要它有：
#       · reset() -> core_state
#       · step(action_id: int) -> (core_state, done: bool, info: dict)
#     其中 action_id 是 DQN 输出的离散动作编号（到具体派工动作的映射可以在 core_env 内部实现，
#     或者在 core_env.step 里调用一个 action 编码工具）。
#   - ShopEnv 负责：
#       · 管理 episode 步数；
#       · 把 core_state 映射为 RL 用的观测 obs（通过 obs_builder 回调）；
#       · 用 reward_fn(prev_state, next_state, done, info) 计算 reward。
#
# 依赖：
#   - envs.base_env.BaseEnv
#   - 一个用户自定义的 core_env 实例（见 __init__ 文档）
# ============================

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .base_env import BaseEnv


class ShopEnv(BaseEnv):
    """
    下层调度 DQN 的环境包装器。

    约定 core_env 接口：
        - state = core_env.reset()                      # 复位，返回内部核心状态
        - next_state, done, info = core_env.step(a_id)  # 执行动作 a_id

    其中：
        - a_id: int，DQN 输出的离散动作编号。
        - state / next_state: 任意 Python 对象，代表核心环境的状态快照；
          由 obs_builder 决定如何把它转成 RL state。
        - info: dict，包含 makespan、当前时间、阻塞统计等信息，reward_fn 可用其做奖励 shaping。

    obs_builder 接口：
        - obs = obs_builder(state)
          · 把核心状态 state 编码成 DQN 的输入（例如 np.ndarray 或 torch.Tensor）。

    reward_fn 接口：
        - r = reward_fn(prev_state, next_state, done, info)
          · 用前后状态、done 标志和 info 计算单步 reward。
          · 对于“终局 reward = -C_max” 的情况，可以只在 done=True 时给非零 reward。

    用法示意（伪代码）::

        core_env = MyFlowShopCoreEnv(instance, buffers, ...)
        env = ShopEnv(
            core_env=core_env,
            obs_builder=my_obs_builder,
            reward_fn=my_reward_fn,
            max_steps=1000,
        )

        obs = env.reset()
        done = False
        while not done:
            action = dqn_agent.select_action(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        core_env: Any,
        obs_builder: Callable[[Any], Any],
        reward_fn: Callable[[Any, Any, bool, Dict[str, Any]], float],
        max_steps: Optional[int] = None,
    ) -> None:
        """
        参数：
            core_env:
                核心车间环境实例，需实现：
                    - reset() -> core_state
                    - step(action_id: int) -> (core_state, done: bool, info: dict)
            obs_builder:
                把 core_state 映射成 RL 用的观测：
                    obs = obs_builder(core_state)
                例如可以使用 envs/observations/shop_obs_dense.py 中的函数。
            reward_fn:
                计算单步 reward 的函数：
                    r = reward_fn(prev_state, next_state, done, info)
                可以在 envs/reward.py 中实现多种模式（terminal / dense / 带阻塞惩罚等），
                然后在这里传入对应的函数。
            max_steps:
                可选的最大步数限制；若为 None，则由 core_env 的 done 标志决定 episode 结束。
                若不为 None，则一旦步数 >= max_steps 即强制 done=True。
        """
        super().__init__()
        self._core_env = core_env
        self._obs_builder = obs_builder
        self._reward_fn = reward_fn
        self._max_steps = max_steps

        self._core_state: Any = None
        self._prev_core_state: Any = None
        self._done: bool = False
        self._step_count: int = 0

    # ------------------------------------------------------------------
    #  BaseEnv 接口
    # ------------------------------------------------------------------
    def reset(self) -> Any:
        """
        重置核心环境，并返回重置后的观测 obs。

        调用顺序：
            core_state = core_env.reset()
            obs = obs_builder(core_state)
        """
        self._core_state = self._core_env.reset()
        self._prev_core_state = None
        self._done = False
        self._step_count = 0

        obs = self._obs_builder(self._core_state)
        return obs

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        在核心环境中执行一个离散动作 action_id，并返回：
            obs_next, reward, done, info

        注意：
            - 若 episode 已结束（_done=True），再次调用 step 会抛出异常；
            - done=True 时，core_env 已经到达终态或达到了 max_steps 限制。
        """
        if self._done:
            raise RuntimeError("调用 step 之前需要先 reset：当前 episode 已结束。")

        self._prev_core_state = self._core_state

        # 把动作直接传给核心环境；具体含义由 core_env 内部决定
        next_state, done_core, info = self._core_env.step(action)

        self._core_state = next_state
        self._step_count += 1

        # 若设置了 max_steps，则覆盖/合并 done
        done = done_core
        if self._max_steps is not None and self._step_count >= self._max_steps:
            done = True
            # 可以在 info 中记录是因为 step 限制导致结束
            info = dict(info) if info is not None else {}
            info.setdefault("truncated", True)

        # 计算 reward
        reward = self._reward_fn(self._prev_core_state, self._core_state, done, info)

        # 构造下一个观测
        obs_next = self._obs_builder(self._core_state)

        self._done = done

        return obs_next, float(reward), done, info

# 文件：examples/test_pipeline.py

from __future__ import annotations
# 最上面，加在所有 from ... import ... 之前
import os
import sys

# 项目根目录 = 本文件的上一级目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

# === 实例相关 ===
from instances.generators import FlowShopGeneratorConfig, generate_random_instance

# === 下层相关 ===
from envs.ffs_core_env import FlowShopCoreEnv
from envs.shop_env import ShopEnv
from envs.observations.shop_obs_dense import build_shop_obs, ShopObsConfig
from envs.reward import ShopRewardConfig, make_shop_reward_fn

# === 上层相关 ===
from envs.buffer_design_env import BufferDesignEnv, BufferDesignEnvConfig
from envs.observations.buffer_obs_dense import BufferObsConfig
from envs.constraints import compute_buffer_upper_bounds

# --------------------------------------------------
# 1. 构造一个简单的算例（你也可以换成读文件）
# --------------------------------------------------


def make_toy_instance(seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = FlowShopGeneratorConfig(
        num_jobs=5,
        num_stages=4,
        machines_per_stage=[2, 2, 2, 2],
        proc_time_low=1,
        proc_time_high=10,
        seed=seed,
    )
    inst = generate_random_instance(cfg, rng)
    return inst


# --------------------------------------------------
# 2. 下层评估函数：给定 (instance, buffers)，随机策略跑一遍
# --------------------------------------------------


def evaluate_with_random_policy(
    instance,
    buffers,
    max_steps: int = 10_000,
    seed: int = 0,
):
    """
    下层评估函数：
        - 使用 FlowShopCoreEnv + ShopEnv；
        - 使用 build_shop_obs 作为状态编码；
        - 使用简单的 ShopRewardConfig（reward 其实不重要，只看 makespan）；
        - 策略：每步随机选择 job_id。

    返回：
        metrics: dict，至少包含 "makespan"。
    """
    rng = np.random.default_rng(seed)

    # 1) 构建核心环境
    core_env = FlowShopCoreEnv(
        instance=instance,
        buffers=buffers,
        dispatch_machine_rule="min_index",  # 简单机器选择规则
        seed=seed,
    )

    # 2) 构建 obs_builder
    obs_cfg = ShopObsConfig()
    obs_builder = lambda core_state: build_shop_obs(core_state, obs_cfg)

    # 3) 构建 reward_fn（这里只是为了 ShopEnv 完整运行，reward 值不重要）
    shop_reward_cfg = ShopRewardConfig(
        mode="dense",
        time_weight=0.0,          # 不惩罚时间
        blocking_weight=0.0,
        buffer_occ_weight=0.0,
        makespan_weight=0.0,      # 不用 makespan 做 reward
        invalid_action_weight=0.1 # 非法动作稍微惩罚一下
    )
    reward_fn = make_shop_reward_fn(shop_reward_cfg)

    # 4) 构建 ShopEnv
    env = ShopEnv(
        core_env=core_env,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        max_steps=max_steps,
    )

    # 5) 随机策略跑一遍
    obs = env.reset()
    done = False
    last_info = {}

    # 决定动作空间大小：优先用 instance.num_jobs，没有则用 len(instance.jobs)
    if hasattr(instance, "num_jobs"):
        num_jobs = int(instance.num_jobs)
    else:
        num_jobs = len(instance.jobs)


    while not done:
        action = int(rng.integers(0, num_jobs))  # 随机 job_id
        obs, reward, done, info = env.step(action)
        last_info = info

        # 为了避免死循环，这里可以打印一些信息看看
        # print(f"step reward={reward}, time={core_env.time}")

    # 6) 获取 makespan
    makespan = None
    if "makespan" in last_info:
        makespan = float(last_info["makespan"])
    else:
        # 回退为核心环境中的计算
        makespan = float(core_env._compute_makespan())

    metrics = {
        "makespan": makespan,
        # 根据需要，你可以把 blocking_time 等指标也加进去
    }
    return metrics


# --------------------------------------------------
# 3. 上层：用 BufferDesignEnv + 上面的 evaluate_fn 测一遍
# --------------------------------------------------


def main():
    # 1) 准备几个算例（这里就用一个）
    instance = make_toy_instance(seed=0)
    instances = [instance]

    # 2) 上层 obs / reward 配置
    buf_obs_cfg = BufferObsConfig()
    buf_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=0.1,    # λ
        randomize_instances=False, # 只有一个实例，无所谓
        max_total_buffer=None,     # 先不加总缓冲上界
    )

    # 3) 构造上层 env，evaluate_fn 使用刚才的随机策略评估
    def evaluate_fn_for_upper(inst, buffers):
        print(f"[evaluate_fn] buffers = {buffers}")
        metrics = evaluate_with_random_policy(inst, buffers, seed=0)
        print(f"[evaluate_fn] metrics = {metrics}")
        return metrics

    upper_env = BufferDesignEnv(
        instances=instances,
        evaluate_fn=evaluate_fn_for_upper,
        cfg=buf_env_cfg,
        obs_cfg=buf_obs_cfg,
        seed=0,
        custom_reward_fn=None,  # 先用默认 reward = -(makespan + λΣb)
    )

    # 4) 上层：随机策略选缓冲 b_k
    obs = upper_env.reset()
    done = False
    step_idx = 0

    # 预取一下各段上界，方便打印
    a_list = compute_buffer_upper_bounds(instance)
    print(f"buffer upper bounds a_k = {a_list}")

    while not done:
        # 当前要决策的是第 i 段
        i = upper_env._current_buf_index  # 简单测试用，正式训练建议通过 obs 来判断

        a_i = a_list[i]
        # 在 [0, a_i] 中随机选一个容量
        action = int(np.random.randint(0, a_i + 1))

        print(f"[upper step {step_idx}] decide buffer[{i}] action={action}")
        obs, reward, done, info = upper_env.step(action)
        print(f"  reward={reward}, done={done}")

        if done:
            print("=== Episode finished ===")
            print("final buffers:", info.get("buffers"))
            print("metrics:", info.get("metrics"))
        step_idx += 1


if __name__ == "__main__":
    main()

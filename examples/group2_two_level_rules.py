# examples/group2_two_level_rules.py
"""
Group2：上层 BA + 下层规则派工（Two-level: Buffer Agent + Rule）

- 上层：D3QN Buffer Agent，在 BufferDesignEnv 上学习缓冲配置；
- 下层：不训练，用给定规则 (fifo/spt/lpt/srpt) 在 FlowShopCoreEnv 上派工。

对外接口（给总控脚本调用）：

    run_group2_for_experiment(
        experiment_name: str,
        rules: List[str],
        seeds: List[int],
        device_str: str = "cuda",
        num_outer_episodes: int = 400,
    )

每个 (experiment, rule, seed) 的结果目录：

    results/<experiment_name>/group2_two_level_<rule>/seed{seed}_YYYYMMDD_HHMMSS/
        - cfg.json
        - train_log.csv
        - eval_test_summary_detail.csv
"""

from __future__ import annotations

import os
import sys
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# ---------- 路径设置：把 src/ 和 examples/ 加入 sys.path ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

# ---------- 项目内部导入 ----------
from instances.types import InstanceData
from instances.io import load_instances_from_dir
from envs.ffs_core_env import simulate_instance_with_job_rule
from envs.buffer_design_env import (
    BufferDesignEnv,
    BufferDesignEnvConfig,
    compute_buffer_upper_bounds,
)
from envs.reward import ShopRewardConfig
from policies.upper_buffer_agent import UpperAgentConfig, UpperAgent, create_upper_agent


# ============================================================
# 工具：加载算例
# ============================================================

def load_instances_for_experiment(
    experiment_name: str,
) -> Tuple[List[InstanceData], List[InstanceData], List[InstanceData]]:
    """
    从 experiments/raw/<experiment_name>/{train,val,test} 加载实例。
    """
    data_root = Path(ROOT_DIR) / "experiments" / "raw" / experiment_name
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_instances = load_instances_from_dir(str(train_dir))
    val_instances = load_instances_from_dir(str(val_dir)) if val_dir.exists() else []
    test_instances = load_instances_from_dir(str(test_dir)) if test_dir.exists() else []

    print(
        f"[DATA] Loaded instances ({experiment_name}): "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )
    return train_instances, val_instances, test_instances


# ============================================================
# Group2 Config
# ============================================================

@dataclass
class Group2Config:
    experiment_name: str
    method_name: str     # e.g. "group2_two_level_fifo"
    rule_name: str       # "fifo" / "spt" / "lpt" / "srpt"
    num_outer_episodes: int
    device: str
    random_seed: int

    upper_agent_cfg: UpperAgentConfig
    reward_cfg: ShopRewardConfig
    buffer_cost_weight: float
    deadlock_penalty: float
    enable_train_log: bool = True


# ============================================================
# Rule evaluate_fn for BufferDesignEnv
# ============================================================

def make_evaluate_fn_for_rule(
    rule_name: str,
    max_steps: int,
) -> Any:
    """
    为上层 BufferDesignEnv 构造 evaluate_fn：
      输入 (instance, buffers)，用给定 job_rule 在底层 FlowShopCoreEnv 中跑一集。

    注意：规则派工是确定性的，这里 seed 固定为 0 不影响结果。
    """

    rule_name = rule_name.lower()

    def evaluate_fn(instance: InstanceData, buffers: List[int]) -> Dict[str, float]:
        metrics = simulate_instance_with_job_rule(
            instance=instance,
            buffers=buffers,
            job_rule=rule_name,
            machine_rule="min_index",
            max_steps=max_steps,
            seed=0,
        )
        makespan = float(metrics.get("makespan", max_steps))
        deadlock = bool(metrics.get("deadlock", False))
        return {
            "makespan": makespan,
            "deadlock": deadlock,
        }

    return evaluate_fn


# ============================================================
# test 集 eval（给定训练好的 upper_agent + rule）
# ============================================================

def evaluate_group2_on_test(
    cfg: Group2Config,
    upper_agent: UpperAgent,
    test_instances: List[InstanceData],
    out_dir: Path,
    device: torch.device | None = None,
    num_eval_episodes: int = 100,
) -> None:
    if not test_instances:
        print("[TEST] No test instances, skip.")
        return

    out_dir = Path(out_dir)
    summary_path = out_dir / "eval_test_summary_detail.csv"
    detail_path = out_dir / "eval_test_detail.csv"

    eval_fn = make_evaluate_fn_for_rule(
        rule_name=cfg.rule_name,
        max_steps=10_000,  # 测试阶段可放宽一点步数上限
    )

    eval_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=False,
        max_total_buffer=None,
    )

    rng = np.random.RandomState(cfg.random_seed + 999)

    total_ms = 0.0
    total_dead = 0
    episodes = 0
    details: List[Dict[str, Any]] = []

    # 兼容外部 runtime 脚本传入的 device；当前函数内部不直接使用
    _ = device

    for ep in range(num_eval_episodes):
        inst_idx = int(rng.randint(len(test_instances)))
        inst = test_instances[inst_idx]

        eval_env = BufferDesignEnv(
            instances=[inst],
            evaluate_fn=eval_fn,
            cfg=eval_env_cfg,
            obs_cfg=None,
            seed=cfg.random_seed + ep,
            custom_reward_fn=None,
        )

        obs_U = eval_env.reset()
        done_U = False
        info_U: Dict[str, Any] = {}

        while not done_U:
            action_U = int(upper_agent.select_greedy_action(obs_U))
            obs_U, r_U, done_U, info_U = eval_env.step(action_U)

        metrics = info_U.get("metrics", {})
        ms = float(metrics.get("makespan", math.inf))
        bufs = metrics.get("buffers", [])
        dl = bool(metrics.get("deadlock", False))

        total_ms += ms
        total_dead += 1 if dl else 0
        episodes += 1

        if isinstance(bufs, (list, tuple)):
            buf_str = " ".join(str(b) for b in bufs)
        else:
            buf_str = str(bufs)

        details.append(
            dict(
                episode=ep + 1,
                instance_idx=inst_idx,
                buffers=buf_str,
                makespan=ms,
                deadlock=int(dl),
            )
        )

    if episodes == 0:
        print("[TEST] No episodes evaluated on test, skip writing csv.")
        return

    avg_ms = total_ms / episodes
    dead_rate = total_dead / episodes

    print(
        f"[Group2-Eval][TEST] rule={cfg.rule_name}, episodes={episodes}, "
        f"avg_makespan={avg_ms:.3f}, deadlock_rate={dead_rate:.3f}"
    )

    with summary_path.open("w", newline="") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            [
                "split",
                "rule",
                "num_eval_episodes",
                "avg_makespan",
                "deadlock_rate",
                "ckpt",
            ]
        )
        writer.writerow(
            [
                "test",
                cfg.rule_name,
                int(episodes),
                float(avg_ms),
                float(dead_rate),
                "upper_q_last.pth",
            ]
        )

    with detail_path.open("w", newline="") as f_det:
        writer = csv.writer(f_det)
        writer.writerow(
            ["episode", "instance_idx", "buffers", "makespan", "deadlock"]
        )
        for row in details:
            writer.writerow(
                [
                    int(row["episode"]),
                    int(row["instance_idx"]),
                    row["buffers"],
                    float(row["makespan"]),
                    int(row["deadlock"]),
                ]
            )

    print(f"[SAVE] Test summary saved to {summary_path}")
    print(f"[SAVE] Test detail saved to {detail_path}")


# ============================================================
# 单个 (rule, seed) 的训练 + eval
# ============================================================

def train_and_eval_one_seed_for_rule(
    cfg: Group2Config,
    train_instances: List[InstanceData],
    test_instances: List[InstanceData],
    base_out_dir: Path,
    skip_final_test_eval: bool = False,
    return_trained_objects: bool = False,
):
    """
    对固定的 rule_name + seed，在给定 experiment_name 的 train/test 实例上
    训练上层 BA，并在 test 上评估。
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"seed{cfg.random_seed}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[Group2-RUN] experiment={cfg.experiment_name}, rule={cfg.rule_name}, "
        f"seed={cfg.random_seed}, out_dir={out_dir}"
    )
    print(f"[INFO] Using device: {device}")

    # ========= 1) 先确定上层 obs_dim / action_dim =========
    # 上层动作：a_k ∈ {0,...,max_a}
    max_a = 0
    for inst in train_instances:
        bounds = compute_buffer_upper_bounds(inst)
        if bounds:
            max_a = max(max_a, max(bounds))
    if max_a <= 0:
        max_a = 1
    upper_action_dim = max_a + 1

    # 用和训练时完全一致的 BufferDesignEnv 配置，先建一个临时 env 拿 obs_dim
    upper_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=True,
        max_total_buffer=None,
    )

    # 临时 evaluate_fn（规则版）
    train_eval_fn_tmp = make_evaluate_fn_for_rule(
        rule_name=cfg.rule_name,
        max_steps=10_000,
    )

    upper_env_tmp = BufferDesignEnv(
        instances=train_instances,
        evaluate_fn=train_eval_fn_tmp,
        cfg=upper_env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )
    obs_U0 = upper_env_tmp.reset()
    upper_obs_dim = obs_U0.shape[0]

    cfg.upper_agent_cfg.obs_dim = upper_obs_dim
    cfg.upper_agent_cfg.action_dim = upper_action_dim

    print(
        f"[INFO] Rule={cfg.rule_name}, upper_obs_dim={upper_obs_dim}, "
        f"upper_action_dim={upper_action_dim}"
    )

    # ========= 2) 构造上层 agent =========
    upper_agent: UpperAgent = create_upper_agent(cfg.upper_agent_cfg, device)

    # ========= 3) 正式训练用 env =========
    train_eval_fn = make_evaluate_fn_for_rule(
        rule_name=cfg.rule_name,
        max_steps=10_000,
    )

    upper_env = BufferDesignEnv(
        instances=train_instances,
        evaluate_fn=train_eval_fn,
        cfg=upper_env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )

    # ========= 4) 保存 cfg.json =========
    cfg_json_path = out_dir / "cfg.json"
    try:
        with cfg_json_path.open("w", encoding="utf-8") as f_cfg:
            json.dump(
                dict(
                    experiment_name=cfg.experiment_name,
                    method_name=cfg.method_name,
                    rule_name=cfg.rule_name,
                    random_seed=cfg.random_seed,
                    num_outer_episodes=cfg.num_outer_episodes,
                    buffer_cost_weight=cfg.buffer_cost_weight,
                    deadlock_penalty=cfg.deadlock_penalty,
                    reward_cfg=cfg.reward_cfg.__dict__,
                    upper_agent_cfg={
                        k: getattr(cfg.upper_agent_cfg, k)
                        for k in cfg.upper_agent_cfg.__dict__.keys()
                    },
                ),
                f_cfg,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] Saved cfg.json to {cfg_json_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cfg.json: {e}")

    # ========= 5) 打开训练日志 =========
    log_file = None
    log_writer = None
    if cfg.enable_train_log:
        log_path = out_dir / "train_log.csv"
        log_file = log_path.open("w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "outer_ep",
                "instance_idx",
                "upper_ep_reward",
                "final_makespan",
                "deadlock",
                "total_buffer",
                "epsilon_U",
                "buffers",
            ]
        )
        log_file.flush()

    # ========= 6) 训练主循环 =========
    try:
        for outer_ep in range(cfg.num_outer_episodes):
            upper_agent.global_episode = outer_ep
            epsilon_U = upper_agent.compute_epsilon()

            obs_U = upper_env.reset()
            done_U = False
            ep_reward_U = 0.0
            traj: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
            info_U: Dict[str, Any] = {}

            while not done_U:
                action_U = upper_agent.select_action(obs_U, epsilon_U)
                next_obs_U, reward_U, done_U, info_U = upper_env.step(action_U)
                traj.append((obs_U, action_U, float(reward_U), next_obs_U))
                obs_U = next_obs_U
                ep_reward_U += float(reward_U)

            metrics_U = info_U.get("metrics", {})
            final_makespan = float(metrics_U.get("makespan", 0.0))
            deadlock_flag = bool(metrics_U.get("deadlock", False))
            bufs = metrics_U.get("buffers", [])
            total_buffer = float(sum(bufs)) if isinstance(bufs, (list, tuple)) else 0.0
            instance_idx = info_U.get("instance_idx", -1)

            # 回放到上层 buffer
            for i, (s, a, r, s_next) in enumerate(traj):
                done_flag = (i == len(traj) - 1)
                upper_agent.add_transition(s, a, r, s_next, done_flag)

            upper_agent.update_one_step()
            if (outer_ep + 1) % cfg.upper_agent_cfg.target_update_interval == 0:
                upper_agent.update_target()

            if isinstance(bufs, (list, tuple)):
                buffer_str = " ".join(str(b) for b in bufs)
            else:
                buffer_str = str(bufs)

            if log_writer is not None:
                log_writer.writerow(
                    [
                        outer_ep,
                        instance_idx,
                        float(ep_reward_U),
                        float(final_makespan),
                        int(deadlock_flag),
                        total_buffer,
                        float(epsilon_U),
                        buffer_str,
                    ]
                )
                if (outer_ep + 1) % 100 == 0 and log_file is not None:
                    log_file.flush()

            if (outer_ep + 1) % 100 == 0 or outer_ep == 0:
                print(
                    f"[Group2][OuterEP {outer_ep+1:04d}] rule={cfg.rule_name}, "
                    f"reward={ep_reward_U:.3f}, makespan={final_makespan:.1f}, "
                    f"deadlock={int(deadlock_flag)}, epsilon_U={epsilon_U:.3f}"
                )

        # 保存 upper q_net
        ckpt_path = out_dir / "upper_q_last.pth"
        torch.save(upper_agent.q_net.state_dict(), ckpt_path)
        print(f"[CKPT] Saved upper_agent q_net to {ckpt_path}")

    finally:
        if log_file is not None:
            log_file.close()

    # ========= 7) test eval =========
    # ========= 最终 test 评估 =========
    if not skip_final_test_eval:
        evaluate_group2_on_test(
            upper_agent=upper_agent,
            cfg=cfg,
            test_instances=test_instances,
            out_dir=out_dir,
            device=device,
            num_eval_episodes=100,
        )

    if return_trained_objects:
        return {
            "upper_agent": upper_agent,
            "out_dir": out_dir,
            "cfg": cfg,
            "device": device,
        }

    return None


# ============================================================
# 构造 Group2Config
# ============================================================

def build_group2_config(
    experiment_name: str,
    rule_name: str,
    seed: int,
    num_outer_episodes: int,
    device_str: str,
) -> Group2Config:
    # 上层 reward 中只用 buffer_cost_weight 和 deadlock_penalty，makespan 由 BufferDesignEnv 内部处理
    reward_cfg = ShopRewardConfig(
        mode="progress",           # 这里字段不直接用在 Group2，但保留结构以便 cfg.json 一致
        time_weight=1.0,
        per_operation_reward=0.0,
        per_job_reward=0.0,
        blocking_penalty=0.0,
        terminal_bonus=0.0,
        invalid_action_weight=0.0,
        makespan_weight=0.0,
    )

    upper_agent_cfg = UpperAgentConfig(
        obs_dim=0,
        action_dim=0,
        gamma=0.99,
        lr=1e-4,
        batch_size=128,
        buffer_capacity=10_000,
        target_update_interval=100,
        buffer_cost_weight=1.0,
        algo_type="d3qn",
        replay_type="uniform",
    )

    method_name = f"group2_two_level_{rule_name.lower()}"

    cfg = Group2Config(
        experiment_name=experiment_name,
        method_name=method_name,
        rule_name=rule_name.lower(),
        num_outer_episodes=num_outer_episodes,
        device=device_str,
        random_seed=seed,
        upper_agent_cfg=upper_agent_cfg,
        reward_cfg=reward_cfg,
        buffer_cost_weight=1.0,
        deadlock_penalty=2_000.0,
        enable_train_log=True,
    )
    return cfg


# ============================================================
# 对外入口：run_group2_for_experiment
# ============================================================

def run_group2_for_experiment(
    experiment_name: str,
    rules: List[str],
    seeds: List[int],
    device_str: str = "cuda",
    num_outer_episodes: int = 400,
) -> None:
    """
    在指定 experiment_name 上，对多条规则 + 多个种子训练上层 Buffer Agent。

    结果目录：
      results/<experiment_name>/group2_two_level_<rule>/seed{seed}_YYYYMMDD_HHMMSS/
    """
    rules = [r.strip().lower() for r in rules if r.strip()]

    print(
        f"[Group2-MASTER] experiment_name={experiment_name}, "
        f"rules={rules}, seeds={seeds}, num_outer_episodes={num_outer_episodes}, "
        f"device={device_str}"
    )

    train_instances, val_instances, test_instances = load_instances_for_experiment(
        experiment_name
    )

    if not train_instances:
        print(f"[Group2][ERROR] No train instances for {experiment_name}, skip.")
        return
    if not test_instances:
        print(f"[Group2][WARN] No test instances for {experiment_name}, test eval will be skipped.")

    for rule in rules:
        method_name = f"group2_two_level_{rule}"
        base_out_dir = Path(ROOT_DIR) / "results" / experiment_name / method_name
        base_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Group2-MASTER] Rule = {rule}, results dir = {base_out_dir}")

        for seed in seeds:
            cfg = build_group2_config(
                experiment_name=experiment_name,
                rule_name=rule,
                seed=seed,
                num_outer_episodes=num_outer_episodes,
                device_str=device_str,
            )
            try:
                train_and_eval_one_seed_for_rule(
                    cfg=cfg,
                    train_instances=train_instances,
                    test_instances=test_instances,
                    base_out_dir=base_out_dir,
                )
            except Exception as e:
                print(
                    f"[Group2][ERROR] Failed for experiment={experiment_name}, "
                    f"rule={rule}, seed={seed}: {e}"
                )

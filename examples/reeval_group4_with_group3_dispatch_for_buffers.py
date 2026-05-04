# examples/reeval_group4_with_group3_dispatch_for_buffers.py
"""
使用 Group3 训练好的下层 D3QN 调度器，作为 Group4 的下层，
重新在 test 集上对 Group4 的上层缓冲 agent 做贪婪推理，
主要目的是得到一份“近似的动态 buffers 轨迹”。

注意：
- 这不是原始 Group4 方法的精确重现，因为当年训练时的下层网络没有保存。
- 上层使用 Group4 训练得到的 upper_q_last.pth；
- 下层使用 Group3 的 dispatch_q_last.pth（同一 experiment、同一 seed）；
- 结果只能作为“Group4 上层 + Group3 下层”的近似分析用。

对每个 experiment / seed 组合：
  - 读取：
        results/<exp>/group4_two_level/seedS_xxx/cfg.json
        results/<exp>/group4_two_level/seedS_xxx/upper_q_last.pth
        results/<exp>/dispatch_d3qn_fixedbuf/seedS_yyy/cfg.json
        results/<exp>/dispatch_d3qn_fixedbuf/seedS_yyy/dispatch_q_last.pth
  - 构建：
        上层 UpperAgent（Group4）
        下层 dispatch agent（Group3）
  - 在 test 集上跑 NUM_EVAL_EPISODES 次 BufferDesignEnv 贪婪策略：
        每次随机选一个 test instance
        记录 instance_idx, buffers, makespan, deadlock
  - 写出：
        eval_test_detail_g3lower.csv
        eval_test_summary_detail_g3lower.csv
    （保存在 Group4 的 seed 目录下，不覆盖原有文件）
"""

from __future__ import annotations

import os
import sys
import json
import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

# ---------- 路径设置 ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

RESULTS_DIR = Path(ROOT_DIR) / "results"

# ---------- 导入项目模块 ----------
from instances.io import load_instances_from_dir
from instances.types import InstanceData

from envs.reward import ShopRewardConfig
from envs.buffer_design_env import BufferDesignEnv, BufferDesignEnvConfig

from policies.upper_buffer_agent import UpperAgentConfig, UpperAgent, create_upper_agent

from train_dispatch_d3qn_fixedbuf import (  # type: ignore
    DispatchAgentConfig,
    create_dispatch_agent,
    make_shop_env,
    select_action_greedy,
)

NUM_EVAL_EPISODES = 100  # 与 Group4 默认一致


# ============================================================
# 工具函数
# ============================================================

def _scan_experiments() -> List[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()])


def _parse_seed_and_runid(name: str) -> Tuple[int | None, str]:
    """
    从目录名 seed<seed>_<runid> 解析 (seed, runid)。
    """
    if not name.startswith("seed"):
        return None, ""
    rest = name[4:]
    if "_" in rest:
        s, runid = rest.split("_", 1)
    else:
        s, runid = rest, ""
    try:
        seed = int(s)
    except ValueError:
        seed = None
    return seed, runid


def _load_group4_cfg(seed_dir: Path) -> Dict[str, Any] | None:
    cfg_path = seed_dir / "cfg.json"
    if not cfg_path.exists():
        print(f"[WARN] Group4 cfg.json not found: {cfg_path}")
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_group3_cfg(seed_dir: Path) -> Dict[str, Any] | None:
    cfg_path = seed_dir / "cfg.json"
    if not cfg_path.exists():
        print(f"[WARN] Group3 cfg.json not found: {cfg_path}")
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_instances_for_experiment(
    exp_name: str,
) -> Tuple[List[InstanceData], List[InstanceData], List[InstanceData]]:
    data_root = Path(ROOT_DIR) / "experiments" / "raw" / exp_name
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_instances = load_instances_from_dir(str(train_dir))
    val_instances = load_instances_from_dir(str(val_dir)) if val_dir.exists() else []
    test_instances = load_instances_from_dir(str(test_dir)) if test_dir.exists() else []

    print(
        f"[DATA] Loaded instances ({exp_name}): "
        f"train={len(train_instances)}, val={len(val_instances)}, test={len(test_instances)}"
    )
    return train_instances, val_instances, test_instances


def _build_upper_agent_from_group4(
    cfg_json: Dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
) -> UpperAgent:
    upper_cfg_dict = cfg_json.get("upper_agent_cfg", {})
    if not upper_cfg_dict:
        raise RuntimeError("upper_agent_cfg not found in Group4 cfg.json")

    upper_cfg = UpperAgentConfig(**upper_cfg_dict)
    upper_agent = create_upper_agent(upper_cfg, device)

    if not ckpt_path.exists():
        raise RuntimeError(f"upper checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    upper_agent.q_net.load_state_dict(state)
    upper_agent.target_q_net.load_state_dict(state)
    return upper_agent


def _build_dispatch_agent_from_group3(
    cfg_json: Dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    agent_cfg_dict = cfg_json.get("agent_cfg", {})
    reward_cfg_dict = cfg_json.get("reward_cfg", {})
    if not agent_cfg_dict:
        raise RuntimeError("agent_cfg not found in Group3 cfg.json")
    if not reward_cfg_dict:
        raise RuntimeError("reward_cfg not found in Group3 cfg.json")

    agent_cfg = DispatchAgentConfig(**agent_cfg_dict)
    agent = create_dispatch_agent(agent_cfg, device)

    if not ckpt_path.exists():
        raise RuntimeError(f"dispatch checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    agent["q_net"].load_state_dict(state)
    agent["target_q_net"].load_state_dict(state)

    reward_cfg = ShopRewardConfig(**reward_cfg_dict)
    return {"agent": agent, "reward_cfg": reward_cfg}


def _make_evaluate_fn_with_group3_lower(
    dispatch_agent: Dict[str, Any],
    reward_cfg: ShopRewardConfig,
    device: torch.device,
    max_steps: int,
):
    """
    返回给 BufferDesignEnv 用的 evaluate_fn(instance, buffers)。
    内部使用 Group3 的下层 D3QN 调度器做 greedy 调度。
    """

    agent = dispatch_agent

    def evaluate_fn(instance: InstanceData, buffers: List[int]) -> Dict[str, float]:
        cfg_agent: DispatchAgentConfig = agent["cfg"]
        env = make_shop_env(
            instance=instance,
            buffers=buffers,
            reward_cfg=reward_cfg,
            max_steps=max_steps,
            obs_dim_target=cfg_agent.obs_dim,
        )
        obs = env.reset()
        done = False
        steps = 0
        last_info: Dict[str, Any] = {}

        while (not done) and (steps < max_steps):
            action = select_action_greedy(env, obs, agent, device)
            obs, reward, done, info = env.step(action)
            steps += 1
            last_info = info

        makespan = float(last_info.get("makespan", max_steps))
        deadlock = bool(last_info.get("deadlock", False))
        if (not last_info) or (steps >= max_steps and "makespan" not in last_info):
            deadlock = True

        return dict(
            makespan=makespan,
            deadlock=deadlock,
            steps=steps,
            buffers=list(buffers),
        )

    return evaluate_fn


# ============================================================
# 主逻辑：对单个 (exp, seed) 组合做 re-eval
# ============================================================

def reeval_one_seed(exp_name: str, g4_seed_dir: Path, g3_seed_dir: Path) -> None:
    print(f"\n[REEVAL-G4G3] experiment={exp_name}, g4_dir={g4_seed_dir.name}, g3_dir={g3_seed_dir.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- 1) 读取 Group4 / Group3 的 cfg 和 checkpoint ---
    cfg_g4 = _load_group4_cfg(g4_seed_dir)
    if cfg_g4 is None:
        print("[WARN] skip due to missing Group4 cfg.")
        return

    cfg_g3 = _load_group3_cfg(g3_seed_dir)
    if cfg_g3 is None:
        print("[WARN] skip due to missing Group3 cfg.")
        return

    upper_ckpt_path = g4_seed_dir / "upper_q_last.pth"
    dispatch_ckpt_path = g3_seed_dir / "dispatch_q_last.pth"

    # --- 2) 构建上层 agent & 下层调度 agent ---
    upper_agent = _build_upper_agent_from_group4(cfg_g4, upper_ckpt_path, device)
    disp_bundle = _build_dispatch_agent_from_group3(cfg_g3, dispatch_ckpt_path, device)
    dispatch_agent = disp_bundle["agent"]
    reward_cfg_lower: ShopRewardConfig = disp_bundle["reward_cfg"]

    # --- 3) 读取 test instances ---
    _, _, test_insts = _load_instances_for_experiment(exp_name)
    if not test_insts:
        print("[WARN] No test instances, skip.")
        return

    # --- 4) 构造给 BufferDesignEnv 用的 evaluate_fn ---
    max_steps = int(cfg_g3.get("max_steps_per_episode", 2000))
    eval_fn = _make_evaluate_fn_with_group3_lower(
        dispatch_agent=dispatch_agent,
        reward_cfg=reward_cfg_lower,
        device=device,
        max_steps=max_steps,
    )

    # BufferDesignEnv 配置用 Group4 的 buffer_cost_deadlock 参数
    buffer_cost_weight = float(cfg_g4.get("buffer_cost_weight", 0.0))
    deadlock_penalty = float(cfg_g4.get("deadlock_penalty", 0.0))
    eval_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=buffer_cost_weight,
        deadlock_penalty=deadlock_penalty,
        randomize_instances=False,
        max_total_buffer=None,
    )

    # --- 5) 随机采样 test 实例进行 NUM_EVAL_EPISODES 次贪婪评估 ---
    seed_base = int(cfg_g4.get("random_seed", 0)) + 1234
    rng = np.random.RandomState(seed_base)

    total_ms = 0.0
    total_dead = 0
    episodes = 0
    details: List[Dict[str, Any]] = []

    for ep in range(NUM_EVAL_EPISODES):
        inst_idx = int(rng.randint(len(test_insts)))
        inst = test_insts[inst_idx]

        eval_env = BufferDesignEnv(
            instances=[inst],
            evaluate_fn=eval_fn,
            cfg=eval_env_cfg,
            obs_cfg=None,
            seed=seed_base + ep,
            custom_reward_fn=None,
        )

        obs_U = eval_env.reset()
        done_U = False
        info_U: Dict[str, Any] = {}

        while not done_U:
            obs_np = np.asarray(obs_U, dtype=np.float32)
            action_U = int(upper_agent.select_greedy_action(obs_np))
            obs_U, r_U, done_U, info_U = eval_env.step(action_U)

        metrics = info_U.get("metrics", {})
        ms = float(metrics.get("makespan", math.inf))
        bufs = metrics.get("buffers", [])
        dl = bool(metrics.get("deadlock", False))

        total_ms += ms
        total_dead += 1 if dl else 0
        episodes += 1

        if isinstance(bufs, (list, tuple)):
            buf_str = " ".join(str(int(b)) for b in bufs)
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
        print("[WARN] No episodes evaluated, skip writing csv.")
        return

    avg_ms = total_ms / episodes
    dead_rate = total_dead / episodes
    print(
        f"[Eval-G4G3] exp={exp_name}, seed_dir={g4_seed_dir.name}, "
        f"episodes={episodes}, avg_makespan={avg_ms:.3f}, deadlock_rate={dead_rate:.3f}"
    )

    # --- 6) 写出新的 summary/detail（不覆盖原文件） ---
    summary_path = g4_seed_dir / "eval_test_summary_detail_g3lower.csv"
    detail_path = g4_seed_dir / "eval_test_detail_g3lower.csv"

    with summary_path.open("w", newline="") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(
            ["split", "num_eval_episodes", "avg_makespan", "deadlock_rate", "note"]
        )
        writer.writerow(
            [
                "test",
                int(episodes),
                float(avg_ms),
                float(dead_rate),
                "upper=Group4, lower=Group3_dispatch",
            ]
        )

    with detail_path.open("w", newline="") as f_det:
        writer = csv.writer(f_det)
        writer.writerow(["episode", "instance_idx", "buffers", "makespan", "deadlock"])
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

    print(f"[SAVE] summary -> {summary_path}")
    print(f"[SAVE] detail  -> {detail_path}")


# ============================================================
# 顶层 main：遍历所有 experiment / seed
# ============================================================

def main():
    exps = _scan_experiments()
    if not exps:
        print(f"[ERROR] No experiments found under {RESULTS_DIR}")
        return

    print("[INFO] Experiments for Group4+Group3 re-eval:")
    for e in exps:
        print(f"  - {e}")

    for exp in exps:
        g4_root = RESULTS_DIR / exp / "group4_two_level"
        g3_root = RESULTS_DIR / exp / "dispatch_d3qn_fixedbuf"

        if not g4_root.exists():
            continue
        if not g3_root.exists():
            print(f"[WARN] No Group3 results for {exp}, skip.")
            continue

        # 为该 experiment 下的每个 Group4 seed 找到同 seed 的 Group3 seed 目录
        g3_seed_dirs_by_seed: Dict[int, List[Path]] = {}
        for sd in g3_root.iterdir():
            if not sd.is_dir():
                continue
            s, _ = _parse_seed_and_runid(sd.name)
            if s is None:
                continue
            g3_seed_dirs_by_seed.setdefault(s, []).append(sd)

        for g4_seed_dir in g4_root.iterdir():
            if not g4_seed_dir.is_dir():
                continue
            seed, runid = _parse_seed_and_runid(g4_seed_dir.name)
            if seed is None:
                continue

            cand_list = g3_seed_dirs_by_seed.get(seed, [])
            if not cand_list:
                print(f"[WARN] No matching Group3 seed for {exp}, seed={seed}, skip.")
                continue

            # 简单地按名称排序取第一个
            cand_list_sorted = sorted(cand_list, key=lambda p: p.name)
            g3_seed_dir = cand_list_sorted[0]

            try:
                reeval_one_seed(exp, g4_seed_dir, g3_seed_dir)
            except Exception as e:
                print(f"[ERROR] Re-eval failed for exp={exp}, seed={seed}: {e}")

    print("\n[FINISHED] Group4+Group3 approximate re-evaluation done.")


if __name__ == "__main__":
    main()

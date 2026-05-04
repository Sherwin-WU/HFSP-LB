# revision1/infer_group4_selected_j80_models.py
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

# ------------------------------------------------------------
# 把项目根目录加入 sys.path，保证可导入 examples/ 与 revision1/
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 复用 group4 主方法现有结构
# ============================================================

from examples.group4_two_level import (  # type: ignore
    build_group4_config,
    load_instances_for_experiment,
    create_lower_agent,
    create_upper_agent,
    make_evaluate_fn_for_upper,
    select_upper_action_greedy,
    compute_raw_lower_obs_dim_for_instance,
)
from envs.buffer_design_env import BufferDesignEnv, BufferDesignEnvConfig


@dataclass
class SelectedModel:
    experiment: str
    selected_seed: int
    selected_run_dir: str
    selected_avg_makespan: float
    selected_deadlock_rate: float
    median_rank_position_1based: int
    num_candidates: int
    note: str


def load_selected_models(selection_json: Path) -> List[SelectedModel]:
    with selection_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [SelectedModel(**row) for row in data]


def load_cfg_dict(cfg_json: Path) -> Dict[str, Any]:
    with cfg_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cfg_from_saved_cfg(cfg_dict: Dict[str, Any], device_str: str):
    """
    先用 group4 原 build_group4_config 生成一个结构完整的 cfg，
    再用保存下来的 obs_dim/action_dim 与 reward 配置覆盖。
    """
    cfg = build_group4_config(
        experiment_name=str(cfg_dict["experiment_name"]),
        method_name=str(cfg_dict.get("method_name", "group4_two_level")),
        seed=int(cfg_dict["random_seed"]),
        num_outer_episodes=int(cfg_dict.get("num_outer_episodes", 400)),
        device_str=device_str,
        lower_reward_scheme=str(cfg_dict.get("lower_reward_scheme", "dense_epi")),
        lower_epi_reward_weight=float(cfg_dict.get("lower_epi_reward_weight", 1.0)),
    )

    saved_lower_cfg = cfg_dict["lower_agent_cfg"]
    saved_upper_cfg = cfg_dict["upper_agent_cfg"]
    saved_reward_cfg = cfg_dict["lower_reward_cfg"]

    for k, v in saved_lower_cfg.items():
        setattr(cfg.lower_agent_cfg, k, v)
    for k, v in saved_upper_cfg.items():
        setattr(cfg.upper_agent_cfg, k, v)
    for k, v in saved_reward_cfg.items():
        setattr(cfg.lower_reward_cfg, k, v)

    cfg.buffer_cost_weight = float(cfg_dict.get("buffer_cost_weight", cfg.buffer_cost_weight))
    cfg.deadlock_penalty = float(cfg_dict.get("deadlock_penalty", cfg.deadlock_penalty))
    cfg.lower_reward_scheme = str(cfg_dict.get("lower_reward_scheme", cfg.lower_reward_scheme))
    cfg.lower_epi_reward_weight = float(cfg_dict.get("lower_epi_reward_weight", cfg.lower_epi_reward_weight))
    cfg.device = device_str
    return cfg


def load_trained_agents_from_run_dir(run_dir: Path, device: torch.device):
    cfg_json = run_dir / "cfg.json"
    upper_ckpt = run_dir / "upper_q_last.pth"
    lower_ckpt = run_dir / "lower_q_last.pth"

    if not cfg_json.exists():
        raise FileNotFoundError(f"Missing cfg.json: {cfg_json}")
    if not upper_ckpt.exists():
        raise FileNotFoundError(f"Missing upper_q_last.pth: {upper_ckpt}")
    if not lower_ckpt.exists():
        raise FileNotFoundError(
            f"Missing lower_q_last.pth: {lower_ckpt}\n"
            f"当前 examples/group4_two_level.py 默认只保存 upper_q_last.pth；"
            f"若历史结果目录里没有 lower_q_last.pth，则不能完整恢复 two-level 推理。"
        )

    cfg_dict = load_cfg_dict(cfg_json)
    cfg = build_cfg_from_saved_cfg(cfg_dict, device_str=str(device))

    lower_agent = create_lower_agent(cfg.lower_agent_cfg, device)
    upper_agent = create_upper_agent(cfg.upper_agent_cfg, device)

    lower_agent["q_net"].load_state_dict(torch.load(lower_ckpt, map_location=device))
    lower_agent["target_q_net"].load_state_dict(lower_agent["q_net"].state_dict())
    upper_agent.q_net.load_state_dict(torch.load(upper_ckpt, map_location=device))
    upper_agent.target_q_net.load_state_dict(upper_agent.q_net.state_dict())

    lower_agent["q_net"].eval()
    lower_agent["target_q_net"].eval()
    upper_agent.q_net.eval()
    upper_agent.target_q_net.eval()

    return cfg, upper_agent, lower_agent


def maybe_cuda_sync(device: torch.device, enabled: bool = True) -> None:
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def infer_required_dims_for_experiment(eval_experiment_name: str, cfg) -> Dict[str, int]:
    """
    基于目标实验的一个测试实例，推断该实验真正需要的维度。
    注意：这里只做 reset，不会触发真正的 two-level 评估。
    """
    _, _, test_instances = load_instances_for_experiment(eval_experiment_name)
    if not test_instances:
        raise RuntimeError(f"No test instances for {eval_experiment_name}")

    inst = test_instances[0]

    dummy_eval_env_cfg = BufferDesignEnvConfig(
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        randomize_instances=False,
        max_total_buffer=None,
    )

    dummy_eval_fn = lambda *args, **kwargs: {}

    dummy_env = BufferDesignEnv(
        instances=[inst],
        evaluate_fn=dummy_eval_fn,
        cfg=dummy_eval_env_cfg,
        obs_cfg=None,
        seed=cfg.random_seed,
        custom_reward_fn=None,
    )

    obs_U = dummy_env.reset()
    upper_obs_dim_req = int(np.asarray(obs_U).shape[0])

    if hasattr(dummy_env, "action_space") and hasattr(dummy_env.action_space, "n"):
        upper_action_dim_req = int(dummy_env.action_space.n)
    else:
        upper_action_dim_req = 0

    lower_obs_dim_req = int(compute_raw_lower_obs_dim_for_instance(inst))
    lower_action_dim_req = int(len(inst.jobs))

    return dict(
        upper_obs_dim_req=upper_obs_dim_req,
        upper_action_dim_req=upper_action_dim_req,
        lower_obs_dim_req=lower_obs_dim_req,
        lower_action_dim_req=lower_action_dim_req,
    )


def check_eval_compatibility(cfg, eval_experiment_name: str) -> Tuple[bool, str, Dict[str, int]]:
    """
    不改维度前提下，只有当目标实验所需维度全部兼容当前模型维度时，才允许评估。
    """
    req = infer_required_dims_for_experiment(eval_experiment_name, cfg)

    src_upper_obs_dim = int(cfg.upper_agent_cfg.obs_dim)
    src_upper_action_dim = int(cfg.upper_agent_cfg.action_dim)
    src_lower_obs_dim = int(cfg.lower_agent_cfg.obs_dim)
    src_lower_action_dim = int(cfg.lower_agent_cfg.action_dim)

    reasons: List[str] = []

    if req["upper_obs_dim_req"] != src_upper_obs_dim:
        reasons.append(
            f"upper_obs_dim mismatch: target={req['upper_obs_dim_req']} vs model={src_upper_obs_dim}"
        )

    if req["upper_action_dim_req"] > src_upper_action_dim:
        reasons.append(
            f"upper_action_dim overflow: target={req['upper_action_dim_req']} vs model={src_upper_action_dim}"
        )

    if req["lower_obs_dim_req"] > src_lower_obs_dim:
        reasons.append(
            f"lower_obs_dim overflow: target={req['lower_obs_dim_req']} vs model={src_lower_obs_dim}"
        )

    if req["lower_action_dim_req"] > src_lower_action_dim:
        reasons.append(
            f"lower_action_dim overflow: target={req['lower_action_dim_req']} vs model={src_lower_action_dim}"
        )

    ok = len(reasons) == 0
    note = "ok" if ok else "; ".join(reasons)
    return ok, note, req


def evaluate_one_model_on_one_experiment(
    cfg,
    upper_agent,
    lower_agent,
    eval_experiment_name: str,
    device: torch.device,
    num_eval_episodes: int,
    runtime_cuda_sync: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, _, test_instances = load_instances_for_experiment(eval_experiment_name)
    if not test_instances:
        raise RuntimeError(f"No test instances for {eval_experiment_name}")

    eval_fn = make_evaluate_fn_for_upper(
        lower_agent=lower_agent,
        lower_reward_cfg=cfg.lower_reward_cfg,
        device=device,
        num_inner_episodes=0,
        max_lower_steps=cfg.lower_agent_cfg.max_steps_per_episode,
        lower_train_mode="rd",
        lower_reward_scheme=cfg.lower_reward_scheme,
        buffer_cost_weight=cfg.buffer_cost_weight,
        deadlock_penalty=cfg.deadlock_penalty,
        lower_epi_reward_weight=cfg.lower_epi_reward_weight,
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
    runtime_list_sec: List[float] = []

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

        maybe_cuda_sync(device, runtime_cuda_sync)
        t0 = time.perf_counter()

        obs_U = eval_env.reset()
        done_U = False
        info_U: Dict[str, Any] = {}

        while not done_U:
            action_U = select_upper_action_greedy(obs_U, upper_agent, device)
            obs_U, _, done_U, info_U = eval_env.step(action_U)

        maybe_cuda_sync(device, runtime_cuda_sync)
        t1 = time.perf_counter()
        runtime_sec = float(t1 - t0)
        runtime_list_sec.append(runtime_sec)

        metrics = info_U.get("metrics", {})
        ms = float(metrics.get("makespan", math.inf))
        bufs = metrics.get("buffers", [])
        dl = bool(metrics.get("deadlock", False))

        total_ms += ms
        total_dead += 1 if dl else 0
        episodes += 1

        if isinstance(bufs, (list, tuple)):
            buf_str = " ".join(str(b) for b in bufs)
            total_buffer = float(sum(bufs))
        else:
            buf_str = str(bufs)
            total_buffer = math.nan

        details.append(
            dict(
                eval_experiment=eval_experiment_name,
                episode=ep + 1,
                instance_idx=inst_idx,
                buffers=buf_str,
                total_buffer=total_buffer,
                makespan=ms,
                deadlock=int(dl),
                runtime_sec=runtime_sec,
            )
        )

    summary = dict(
        eval_experiment=eval_experiment_name,
        num_eval_episodes=int(episodes),
        avg_makespan=float(total_ms / episodes) if episodes > 0 else math.inf,
        deadlock_rate=float(total_dead / episodes) if episodes > 0 else math.inf,
        total_runtime_sec=float(sum(runtime_list_sec)) if runtime_list_sec else math.nan,
        avg_runtime_sec=float(sum(runtime_list_sec) / len(runtime_list_sec)) if runtime_list_sec else math.nan,
        min_runtime_sec=float(min(runtime_list_sec)) if runtime_list_sec else math.nan,
        max_runtime_sec=float(max(runtime_list_sec)) if runtime_list_sec else math.nan,
    )
    return summary, details


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _is_complete_two_level_run_dir(run_dir: Path) -> bool:
    return (
        run_dir.is_dir()
        and (run_dir / "cfg.json").exists()
        and (run_dir / "upper_q_last.pth").exists()
        and (run_dir / "lower_q_last.pth").exists()
    )


def _find_seed_run_dirs_under_method(exp_dir: Path, seed: int) -> List[Path]:
    if not exp_dir.exists():
        return []
    prefix = f"seed{seed}_"
    cands = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    cands.sort()
    return cands


def resolve_model_run_dir(
    model: SelectedModel,
    prefer_method_name: str,
    search_roots: Sequence[Path],
) -> Path:
    """
    优先使用 selection json 里的 selected_run_dir；
    若该目录缺少 lower_q_last.pth，则自动转到新的 keep_lower 目录中查找。
    """
    original = Path(model.selected_run_dir).resolve()

    if _is_complete_two_level_run_dir(original):
        return original

    matched: List[Path] = []
    for root in search_roots:
        exp_method_dir = root.resolve() / model.experiment / prefer_method_name
        for p in _find_seed_run_dirs_under_method(exp_method_dir, model.selected_seed):
            if _is_complete_two_level_run_dir(p):
                matched.append(p)

    if matched:
        matched.sort()
        return matched[-1]

    raise FileNotFoundError(
        f"Cannot resolve a complete two-level run dir for "
        f"experiment={model.experiment}, seed={model.selected_seed}.\n"
        f"Original selected_run_dir from json: {original}\n"
        f"Preferred method folder searched: {prefer_method_name}\n"
        f"Search roots: {[str(p.resolve()) for p in search_roots]}\n"
        f"要求目录中同时存在: cfg.json, upper_q_last.pth, lower_q_last.pth"
    )


def get_same_stage_eval_targets(source_experiment: str) -> List[str]:
    """
    source_experiment 例如：
    j80s3 / j200s4 / j200s5
    返回同 stage 的跨规模目标。
    """
    if "s3" in source_experiment:
        return ["j50s3", "j80s3", "j160s3", "j200s3"]
    if "s4" in source_experiment:
        return ["j50s4", "j80s4", "j160s4", "j200s4"]
    if "s5" in source_experiment:
        return ["j50s5", "j80s5", "j160s5", "j200s5"]
    raise ValueError(f"Unsupported source_experiment: {source_experiment}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对选出的 j200s3/j200s4/j200s5 模型做同 stage 跨规模推理，并记录运行时间。"
    )
    parser.add_argument(
        "--selection_json",
        type=str,
        default="results/revision1_j200_selection_keep_lower/selected_j200_stage_models.json",
        help="选模结果 json。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=100,
        help="每个目标实验的评估回合数。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/revision1_j200_generalization_eval_keep_lower",
        help="输出目录。",
    )
    parser.add_argument(
        "--prefer_method_name",
        type=str,
        default="group4_two_level_keep_lower",
        help="优先搜索的补跑方法目录名。",
    )
    parser.add_argument(
        "--search_roots",
        nargs="*",
        default=["results", "results-initial"],
        help="当 selection json 中的 selected_run_dir 不完整时，用于重定位模型目录的搜索根目录。",
    )
    parser.add_argument(
        "--eval_experiments",
        nargs="*",
        default=None,
        help="可选：手工指定统一的目标实验列表。默认按 source model 自动选择同 stage 的 4 个目标。",
    )
    parser.add_argument(
        "--runtime_cuda_sync",
        type=int,
        default=1,
        help="是否在 GPU 计时前后调用 torch.cuda.synchronize()；1=开启，0=关闭。",
    )
    args = parser.parse_args()
    args.runtime_cuda_sync = bool(args.runtime_cuda_sync)

    selection_json = Path(args.selection_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = load_selected_models(selection_json)

    all_summary_rows: List[List[Any]] = []
    all_detail_rows: List[List[Any]] = []
    search_roots = [Path(p) for p in args.search_roots]

    for model in models:
        original_run_dir = Path(model.selected_run_dir).resolve()
        model_run_dir = resolve_model_run_dir(
            model=model,
            prefer_method_name=args.prefer_method_name,
            search_roots=search_roots,
        )

        if model_run_dir != original_run_dir:
            print(
                f"[LOAD] model_experiment={model.experiment}, seed={model.selected_seed}\n"
                f"       original_run_dir = {original_run_dir}\n"
                f"       resolved_run_dir = {model_run_dir}"
            )
        else:
            print(
                f"[LOAD] model_experiment={model.experiment}, "
                f"seed={model.selected_seed}, run_dir={model_run_dir}"
            )

        cfg, upper_agent, lower_agent = load_trained_agents_from_run_dir(model_run_dir, device=device)

        model_out_dir = out_dir / f"{model.experiment}_seed{model.selected_seed}"
        per_model_summary_rows: List[List[Any]] = []
        per_model_detail_rows: List[List[Any]] = []

        if args.eval_experiments is None or len(args.eval_experiments) == 0:
            eval_targets = get_same_stage_eval_targets(model.experiment)
        else:
            eval_targets = args.eval_experiments

        print(
            f"[TARGETS] source_model={model.experiment}_seed{model.selected_seed} "
            f"-> eval_targets={eval_targets}"
        )

        for eval_exp in eval_targets:
            is_ok, compat_note, req = check_eval_compatibility(cfg, eval_exp)

            if not is_ok:
                print(
                    f"[SKIP] source={model.experiment}_seed{model.selected_seed} -> target={eval_exp} | "
                    f"{compat_note}"
                )
                per_model_summary_rows.append([
                    model.experiment,
                    model.selected_seed,
                    eval_exp,
                    "skipped",
                    compat_note,
                    0,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    str(model_run_dir),
                ])
                continue

            print(
                f"[EVAL] source={model.experiment}_seed{model.selected_seed} -> target={eval_exp} | "
                f"upper_obs={req['upper_obs_dim_req']}, upper_act={req['upper_action_dim_req']}, "
                f"lower_obs={req['lower_obs_dim_req']}, lower_act={req['lower_action_dim_req']}"
            )

            summary, details = evaluate_one_model_on_one_experiment(
                cfg=cfg,
                upper_agent=upper_agent,
                lower_agent=lower_agent,
                eval_experiment_name=eval_exp,
                device=device,
                num_eval_episodes=args.num_eval_episodes,
                runtime_cuda_sync=args.runtime_cuda_sync,
            )

            per_model_summary_rows.append([
                model.experiment,
                model.selected_seed,
                eval_exp,
                "ok",
                "",
                summary["num_eval_episodes"],
                summary["avg_makespan"],
                summary["deadlock_rate"],
                summary["total_runtime_sec"],
                summary["avg_runtime_sec"],
                summary["min_runtime_sec"],
                summary["max_runtime_sec"],
                str(model_run_dir),
            ])

            for d in details:
                per_model_detail_rows.append([
                    model.experiment,
                    model.selected_seed,
                    d["eval_experiment"],
                    d["episode"],
                    d["instance_idx"],
                    d["buffers"],
                    d["total_buffer"],
                    d["makespan"],
                    d["deadlock"],
                    d["runtime_sec"],
                ])

        write_csv(
            model_out_dir / "eval_all12_summary.csv",
            [
                "source_model_experiment",
                "source_model_seed",
                "eval_experiment",
                "status",
                "note",
                "num_eval_episodes",
                "avg_makespan",
                "deadlock_rate",
                "total_runtime_sec",
                "avg_runtime_sec",
                "min_runtime_sec",
                "max_runtime_sec",
                "source_run_dir",
            ],
            per_model_summary_rows,
        )

        write_csv(
            model_out_dir / "eval_all12_detail.csv",
            [
                "source_model_experiment",
                "source_model_seed",
                "eval_experiment",
                "episode",
                "instance_idx",
                "buffers",
                "total_buffer",
                "makespan",
                "deadlock",
                "runtime_sec",
            ],
            per_model_detail_rows,
        )

        all_summary_rows.extend(per_model_summary_rows)
        all_detail_rows.extend(per_model_detail_rows)

    write_csv(
        out_dir / "all_models_eval_all12_summary.csv",
        [
            "source_model_experiment",
            "source_model_seed",
            "eval_experiment",
            "status",
            "note",
            "num_eval_episodes",
            "avg_makespan",
            "deadlock_rate",
            "total_runtime_sec",
            "avg_runtime_sec",
            "min_runtime_sec",
            "max_runtime_sec",
            "source_run_dir",
        ],
        all_summary_rows,
    )

    write_csv(
        out_dir / "all_models_eval_all12_detail.csv",
        [
            "source_model_experiment",
            "source_model_seed",
            "eval_experiment",
            "episode",
            "instance_idx",
            "buffers",
            "total_buffer",
            "makespan",
            "deadlock",
            "runtime_sec",
        ],
        all_detail_rows,
    )

    print(f"[SAVE] all summaries -> {out_dir / 'all_models_eval_all12_summary.csv'}")
    print(f"[SAVE] all details   -> {out_dir / 'all_models_eval_all12_detail.csv'}")


if __name__ == "__main__":
    main()
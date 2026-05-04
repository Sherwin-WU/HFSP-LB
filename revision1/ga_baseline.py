# revision1/ga_baseline.py
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================
# 路径
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(EXAMPLES_DIR), str(SRC_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ============================================================
# 项目内导入
# ============================================================

from instances.types import InstanceData
from envs.reward import ShopRewardConfig
from examples.group4_two_level import load_instances_for_experiment, make_shop_env
from envs.buffer_design_env import compute_buffer_upper_bounds


# ============================================================
# 默认常量
# ============================================================

DEFAULT_METHOD_NAME = "ga_static"

DEFAULT_POP_SIZE = 100
DEFAULT_ELITE_SIZE = 15
DEFAULT_TOURNAMENT_SIZE = 15

DEFAULT_CROSSOVER_PROB = 0.90
DEFAULT_MUTATION_PROB = 0.15
DEFAULT_PRIORITY_MUT_SIGMA = 0.10

DEFAULT_BUFFER_MIN = 1
DEFAULT_BUFFER_MAX = 5

DEFAULT_TIME_BUDGET_SEC = 300.0
DEFAULT_MAX_GENERATIONS = 200

DEFAULT_BUFFER_COST_WEIGHT = 0.5
DEFAULT_DEADLOCK_PENALTY = 1000.0

DEFAULT_NUM_EVAL_EPISODES = 10   # 这里按 test instances 的条数跑；summary 会取实际 episode 数
DEFAULT_MAX_LOWER_STEPS = 2000

DEFAULT_EXPERIMENT_NAMES = [
    "j50s3", "j50s4", "j50s5",
    "j80s3", "j80s4", "j80s5",
    "j160s3", "j160s4", "j160s5",
    "j200s3", "j200s4", "j200s5",
]


# ============================================================
# 工具
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clone_list(xs: Sequence[int]) -> List[int]:
    return [int(x) for x in xs]


def infer_problem_dims(instance: InstanceData) -> Tuple[int, int, int]:
    """
    返回：
      num_jobs
      num_stages
      num_buffers = num_stages - 1

    注意：
    当前项目里的 Job 结构不保证有 job.operations，因此这里优先从
    instance 级字段推断；再做多种 schema 兼容。
    """
    # 1) job 数基本都能直接取到
    num_jobs = len(instance.jobs)


    # 2) 先从 instance 级字段推断 num_stages（最稳）
    candidate_stage_fields = [
        "num_stages",
        "n_stages",
        "stage_count",
        "num_machines_per_stage",   # 常见：长度 = stages
        "machines_per_stage",       # 常见：长度 = stages
        "machines_at_stage",        # 常见：长度 = stages
    ]

    num_stages = None

    for name in candidate_stage_fields:
        if not hasattr(instance, name):
            continue
        val = getattr(instance, name)

        # 直接整数
        if isinstance(val, (int, np.integer)):
            num_stages = int(val)
            break

        # 列表/元组：长度代表 stages
        if isinstance(val, (list, tuple)):
            if len(val) > 0:
                num_stages = int(len(val))
                break

        # numpy array：第一维长度代表 stages
        if isinstance(val, np.ndarray):
            if val.ndim >= 1 and val.shape[0] > 0:
                num_stages = int(val.shape[0])
                break

    # 3) 如果 instance 级字段没有，再尝试从 jobs 内部结构兼容推断
    if num_stages is None:
        if hasattr(instance, "jobs") and len(instance.jobs) > 0:
            j0 = instance.jobs[0]

            # schema A: job.operations
            if hasattr(j0, "operations"):
                ops = getattr(j0, "operations")
                if len(ops) > 0:
                    stage_ids = []
                    for job in instance.jobs:
                        for op in getattr(job, "operations", []):
                            if hasattr(op, "stage_id"):
                                stage_ids.append(int(op.stage_id))
                    if stage_ids:
                        num_stages = max(stage_ids) + 1

            # schema B: job.route / routes
            if num_stages is None and hasattr(j0, "route"):
                route = getattr(j0, "route")
                if isinstance(route, (list, tuple, np.ndarray)) and len(route) > 0:
                    num_stages = int(len(route))

            if num_stages is None and hasattr(j0, "routes"):
                route = getattr(j0, "routes")
                if isinstance(route, (list, tuple, np.ndarray)) and len(route) > 0:
                    num_stages = int(len(route))

            # schema C: job.processing_times / proc_times
            if num_stages is None and hasattr(j0, "processing_times"):
                pt = getattr(j0, "processing_times")
                if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) > 0:
                    num_stages = int(len(pt))

            if num_stages is None and hasattr(j0, "proc_times"):
                pt = getattr(j0, "proc_times")
                if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) > 0:
                    num_stages = int(len(pt))

    if num_stages is None or int(num_stages) <= 0:
        raise RuntimeError(
            "Cannot infer num_stages from instance. "
            "Please inspect InstanceData / Job schema."
        )

    num_stages = int(num_stages)
    num_buffers = max(0, num_stages - 1)
    return int(num_jobs), int(num_stages), int(num_buffers)


def total_buffer(buffers: Sequence[int]) -> float:
    return float(sum(int(b) for b in buffers))

def infer_buffer_bounds(instance: InstanceData, cfg: GABaselineConfig) -> Tuple[List[int], List[int]]:
    """
    返回每一段 buffer 的上下界。
    下界先统一为 1；
    上界优先使用项目已有的 compute_buffer_upper_bounds(instance)。
    """
    _, _, num_buffers = infer_problem_dims(instance)

    if num_buffers <= 0:
        return [], []

    try:
        ub = compute_buffer_upper_bounds(instance)
        if isinstance(ub, (list, tuple)) and len(ub) == num_buffers:
            lower = [cfg.buffer_min] * num_buffers
            upper = [max(cfg.buffer_min, int(x)) for x in ub]
            return lower, upper
    except Exception:
        pass

    return [cfg.buffer_min] * num_buffers, [cfg.buffer_max] * num_buffers


def priority_keys_to_rank(priority_keys: np.ndarray) -> np.ndarray:
    """
    输入 shape=(num_jobs,)
    输出 rank[job_id] = 排名（越小优先级越高）
    """
    order = np.argsort(priority_keys)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0], dtype=order.dtype)
    return rank


def select_action_by_rank(legal_actions: Sequence[int], rank: np.ndarray) -> int:
    """
    在当前 legal actions 中，选择全局 rank 最小的 job。
    """
    best_action = None
    best_rank = None
    for a in legal_actions:
        r = int(rank[int(a)])
        if (best_rank is None) or (r < best_rank):
            best_rank = r
            best_action = int(a)

    if best_action is None:
        raise RuntimeError("No legal action available for GA decoder.")
    return best_action


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def sample_test_instances(
    test_instances: Sequence[InstanceData],
    num_samples: int,
    rng: np.random.RandomState,
) -> Tuple[List[InstanceData], List[int]]:
    """
    从当前规模的实例池中随机无放回抽样。
    返回：
      sampled_instances
      sampled_source_indices  # 在原 test_instances 中的下标
    """
    n = len(test_instances)
    if n <= 0:
        return [], []

    k = min(max(1, int(num_samples)), n)
    idxs = rng.choice(n, size=k, replace=False)
    idxs = [int(i) for i in idxs]
    sampled = [test_instances[i] for i in idxs]
    return sampled, idxs

# ============================================================
# 配置
# ============================================================

@dataclass
class GABaselineConfig:
    experiment_name: str
    method_name: str = DEFAULT_METHOD_NAME
    random_seed: int = 0

    pop_size: int = DEFAULT_POP_SIZE
    elite_size: int = DEFAULT_ELITE_SIZE
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE

    crossover_prob: float = DEFAULT_CROSSOVER_PROB
    mutation_prob: float = DEFAULT_MUTATION_PROB
    priority_mut_sigma: float = DEFAULT_PRIORITY_MUT_SIGMA

    buffer_min: int = DEFAULT_BUFFER_MIN
    buffer_max: int = DEFAULT_BUFFER_MAX

    time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC
    max_generations: int = DEFAULT_MAX_GENERATIONS

    buffer_cost_weight: float = DEFAULT_BUFFER_COST_WEIGHT
    deadlock_penalty: float = DEFAULT_DEADLOCK_PENALTY

    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    max_lower_steps: int = DEFAULT_MAX_LOWER_STEPS


# ============================================================
# 染色体
# ============================================================

@dataclass
class GAChromosome:
    buffers: List[int]
    priority_keys: np.ndarray
    fitness: float = math.inf
    makespan: float = math.inf
    deadlock: bool = False
    total_buffer: float = math.inf

    def copy(self) -> "GAChromosome":
        return GAChromosome(
            buffers=clone_list(self.buffers),
            priority_keys=np.asarray(self.priority_keys, dtype=np.float32).copy(),
            fitness=float(self.fitness),
            makespan=float(self.makespan),
            deadlock=bool(self.deadlock),
            total_buffer=float(self.total_buffer),
        )


# ============================================================
# 解码 / 仿真评价
# ============================================================

def decode_and_evaluate_chromosome(
    instance: InstanceData,
    chrom: GAChromosome,
    cfg: GABaselineConfig,
    reward_cfg: ShopRewardConfig,
) -> Dict[str, Any]:
    """
    对单个 instance 解码并评价：
      fitness = makespan + lambda_B * total_buffer + lambda_D * deadlock
    """
    buffers = clone_list(chrom.buffers)
    env = make_shop_env(
        instance=instance,
        buffers=buffers,
        reward_cfg=reward_cfg,
        max_steps=cfg.max_lower_steps,
        obs_dim_target=None,
    )

    _ = env.reset()
    done = False
    steps = 0
    last_info: Dict[str, Any] = {}

    rank = priority_keys_to_rank(np.asarray(chrom.priority_keys, dtype=np.float32))

    while (not done) and (steps < cfg.max_lower_steps):
        legal_actions = env._core_env.get_legal_actions()
        if not legal_actions:
            # 理论上环境里 deadlock / terminal 会体现在 step 后 info 中；
            # 但这里做一层防御。
            break

        action = select_action_by_rank(legal_actions, rank)
        _, _, done, info = env.step(action)
        last_info = info
        steps += 1

    makespan = float(last_info.get("makespan", cfg.max_lower_steps))
    deadlock_flag = bool(last_info.get("deadlock", False))
    total_buf = total_buffer(buffers)

    fitness = (
        makespan
        + float(cfg.buffer_cost_weight) * total_buf
        + float(cfg.deadlock_penalty) * float(deadlock_flag)
    )

    chrom.fitness = float(fitness)
    chrom.makespan = float(makespan)
    chrom.deadlock = bool(deadlock_flag)
    chrom.total_buffer = float(total_buf)

    return {
        "fitness": float(fitness),
        "makespan": float(makespan),
        "deadlock": bool(deadlock_flag),
        "total_buffer": float(total_buf),
        "steps": int(steps),
    }


# ============================================================
# 初始化
# ============================================================

def random_buffers(
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    rng: np.random.RandomState,
) -> List[int]:
    if len(lower_bounds) == 0:
        return []

    vals = []
    for lo, hi in zip(lower_bounds, upper_bounds):
        vals.append(int(rng.randint(int(lo), int(hi) + 1)))
    return vals


def random_priority_keys(num_jobs: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.rand(num_jobs).astype(np.float32)


def make_random_chromosome(
    num_jobs: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cfg: GABaselineConfig,
    rng: np.random.RandomState,
) -> GAChromosome:
    return GAChromosome(
        buffers=random_buffers(lower_bounds, upper_bounds, rng),
        priority_keys=random_priority_keys(num_jobs, rng),
    )


def make_seed_chromosomes(
    num_jobs: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cfg: GABaselineConfig,
) -> List[GAChromosome]:
    """
    少量启发式 warm-start 个体。
    """
    out: List[GAChromosome] = []
    num_buffers = len(lower_bounds)

    if num_buffers > 0:
        out.append(
            GAChromosome(
                buffers=[int(lo) for lo in lower_bounds],
                priority_keys=np.linspace(0.0, 1.0, num_jobs, dtype=np.float32),
            )
        )
        out.append(
            GAChromosome(
                buffers=[int(hi) for hi in upper_bounds],
                priority_keys=np.linspace(0.0, 1.0, num_jobs, dtype=np.float32),
            )
        )
        mid_buf = [
            int(round((int(lo) + int(hi)) / 2.0))
            for lo, hi in zip(lower_bounds, upper_bounds)
        ]
        out.append(
            GAChromosome(
                buffers=mid_buf,
                priority_keys=np.linspace(1.0, 0.0, num_jobs, dtype=np.float32),
            )
        )
    else:
        out.append(
            GAChromosome(
                buffers=[],
                priority_keys=np.linspace(0.0, 1.0, num_jobs, dtype=np.float32),
            )
        )
        out.append(
            GAChromosome(
                buffers=[],
                priority_keys=np.linspace(1.0, 0.0, num_jobs, dtype=np.float32),
            )
        )

    return out


def initialize_population(
    num_jobs: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cfg: GABaselineConfig,
    rng: np.random.RandomState,
) -> List[GAChromosome]:
    population: List[GAChromosome] = []

    seeds = make_seed_chromosomes(num_jobs, lower_bounds, upper_bounds, cfg)
    population.extend([c.copy() for c in seeds])

    while len(population) < cfg.pop_size:
        population.append(
            make_random_chromosome(
                num_jobs=num_jobs,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                cfg=cfg,
                rng=rng,
            )
        )

    return population[: cfg.pop_size]


# ============================================================
# 选择 / 交叉 / 变异
# ============================================================

def tournament_select(
    population: Sequence[GAChromosome],
    tournament_size: int,
    rng: np.random.RandomState,
) -> GAChromosome:
    idxs = rng.choice(len(population), size=tournament_size, replace=False)
    cand = [population[int(i)] for i in idxs]
    cand.sort(key=lambda c: c.fitness)
    return cand[0]


def crossover(
    parent1: GAChromosome,
    parent2: GAChromosome,
    cfg: GABaselineConfig,
    rng: np.random.RandomState,
) -> GAChromosome:
    child = parent1.copy()

    # buffer segment: uniform crossover
    if len(parent1.buffers) > 0:
        new_buffers: List[int] = []
        for b1, b2 in zip(parent1.buffers, parent2.buffers):
            new_buffers.append(int(b1) if rng.rand() < 0.5 else int(b2))
        child.buffers = new_buffers

    # priority segment: uniform crossover on keys
    keys1 = np.asarray(parent1.priority_keys, dtype=np.float32)
    keys2 = np.asarray(parent2.priority_keys, dtype=np.float32)
    mask = rng.rand(keys1.shape[0]) < 0.5
    new_keys = keys1.copy()
    new_keys[mask] = keys2[mask]
    child.priority_keys = new_keys.astype(np.float32)

    # reset evaluated attributes
    child.fitness = math.inf
    child.makespan = math.inf
    child.deadlock = False
    child.total_buffer = math.inf
    return child


def mutate(
    chrom: GAChromosome,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    cfg: GABaselineConfig,
    rng: np.random.RandomState,
) -> None:
    # buffer mutation
    if len(chrom.buffers) > 0 and rng.rand() < cfg.mutation_prob:
        idx = int(rng.randint(len(chrom.buffers)))
        delta = int(rng.choice([-1, 1]))
        val = int(chrom.buffers[idx]) + delta
        lo = int(lower_bounds[idx])
        hi = int(upper_bounds[idx])
        val = max(lo, min(hi, val))
        chrom.buffers[idx] = int(val)

    # priority mutation
    if chrom.priority_keys.shape[0] > 0 and rng.rand() < cfg.mutation_prob:
        num_mut = max(1, int(round(0.05 * chrom.priority_keys.shape[0])))
        idxs = rng.choice(chrom.priority_keys.shape[0], size=num_mut, replace=False)
        noise = rng.normal(loc=0.0, scale=cfg.priority_mut_sigma, size=num_mut).astype(np.float32)
        chrom.priority_keys[idxs] = chrom.priority_keys[idxs] + noise

    chrom.fitness = math.inf
    chrom.makespan = math.inf
    chrom.deadlock = False
    chrom.total_buffer = math.inf


# ============================================================
# 单个 instance 的 GA 搜索
# ============================================================

def run_ga_search_on_instance(
    instance: InstanceData,
    cfg: GABaselineConfig,
    reward_cfg: ShopRewardConfig,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    num_jobs, num_stages, num_buffers = infer_problem_dims(instance)
    lower_bounds, upper_bounds = infer_buffer_bounds(instance, cfg)

    population = initialize_population(
        num_jobs=num_jobs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        cfg=cfg,
        rng=rng,
    )

    trace_rows: List[Dict[str, Any]] = []
    best: Optional[GAChromosome] = None

    t0 = time.perf_counter()
    gen = 0
    stop_reason = "unknown"

    def time_exceeded() -> bool:
        return (time.perf_counter() - t0) >= float(cfg.time_budget_sec)

    # -------------------------
    # gen = 0：初始种群评价
    # -------------------------
    for chrom in population:
        decode_and_evaluate_chromosome(instance, chrom, cfg, reward_cfg)
        if time_exceeded():
            stop_reason = "time_budget_sec"
            break

    population.sort(key=lambda c: c.fitness)
    best = population[0].copy()

    elapsed = time.perf_counter() - t0
    trace_rows.append(
        dict(
            gen=gen,
            elapsed_sec=elapsed,
            best_fitness=best.fitness,
            best_makespan=best.makespan,
            best_deadlock=int(best.deadlock),
            best_total_buffer=best.total_buffer,
        )
    )

    # 如果初始种群评价就已经超时，则直接结束
    if stop_reason == "time_budget_sec":
        return {
            "best_chromosome": best,
            "best_fitness": float(best.fitness),
            "best_makespan": float(best.makespan),
            "best_deadlock": bool(best.deadlock),
            "best_total_buffer": float(best.total_buffer),
            "elapsed_sec": float(elapsed),
            "num_generations": int(gen),
            "trace_rows": trace_rows,
            "num_jobs": int(num_jobs),
            "num_stages": int(num_stages),
            "num_buffers": int(num_buffers),
            "stop_reason": stop_reason,
        }

    # -------------------------
    # 进化阶段：最多 200 代，或 300 秒先到先停
    # -------------------------
    while gen < cfg.max_generations:
        if time_exceeded():
            stop_reason = "time_budget_sec"
            break

        gen += 1

        population.sort(key=lambda c: c.fitness)
        new_pop: List[GAChromosome] = [
            population[i].copy()
            for i in range(min(cfg.elite_size, len(population)))
        ]

        while len(new_pop) < cfg.pop_size:
            if time_exceeded():
                stop_reason = "time_budget_sec"
                break

            p1 = tournament_select(population, cfg.tournament_size, rng)
            p2 = tournament_select(population, cfg.tournament_size, rng)

            if rng.rand() < cfg.crossover_prob:
                child = crossover(p1, p2, cfg, rng)
            else:
                child = p1.copy()

            mutate(child, lower_bounds, upper_bounds, cfg, rng)
            decode_and_evaluate_chromosome(instance, child, cfg, reward_cfg)
            new_pop.append(child)

        population = new_pop
        population.sort(key=lambda c: c.fitness)

        if population[0].fitness < best.fitness:
            best = population[0].copy()

        elapsed = time.perf_counter() - t0
        trace_rows.append(
            dict(
                gen=gen,
                elapsed_sec=elapsed,
                best_fitness=best.fitness,
                best_makespan=best.makespan,
                best_deadlock=int(best.deadlock),
                best_total_buffer=best.total_buffer,
            )
        )

        if stop_reason == "time_budget_sec":
            break

    if stop_reason == "unknown":
        if gen >= cfg.max_generations:
            stop_reason = "max_generations"
        else:
            stop_reason = "time_budget_sec"

    assert best is not None

    return {
        "best_chromosome": best,
        "best_fitness": float(best.fitness),
        "best_makespan": float(best.makespan),
        "best_deadlock": bool(best.deadlock),
        "best_total_buffer": float(best.total_buffer),
        "elapsed_sec": float(elapsed),
        "num_generations": int(gen),
        "trace_rows": trace_rows,
        "num_jobs": int(num_jobs),
        "num_stages": int(num_stages),
        "num_buffers": int(num_buffers),
        "stop_reason": stop_reason,
    }

# ============================================================
# test 评估
# ============================================================

def evaluate_ga_on_test(
    cfg: GABaselineConfig,
    test_instances: List[InstanceData],
    out_dir: Path,
) -> None:
    reward_cfg = ShopRewardConfig(mode="progress")

    # 单个 master seed：
    # 1) 决定该规模抽到哪 10 个实例
    # 2) 决定这 10 次 GA 独立运行中的随机过程
    rng = np.random.RandomState(cfg.random_seed)

    sampled_instances, sampled_source_indices = sample_test_instances(
        test_instances=test_instances,
        num_samples=cfg.num_eval_episodes,
        rng=rng,
    )

    detail_rows: List[Dict[str, Any]] = []
    all_trace_rows: List[Dict[str, Any]] = []

    total_makespan = 0.0
    num_deadlocks = 0
    total_buffers: List[float] = []
    elapsed_list: List[float] = []

    num_eval_episodes = len(sampled_instances)

    # 记录本次被抽到的 10 个实例原始编号
    sampled_json = out_dir / "sampled_instances.json"
    save_json(
        sampled_json,
        {
            "experiment_name": cfg.experiment_name,
            "random_seed": int(cfg.random_seed),
            "num_candidates_total": int(len(test_instances)),
            "num_sampled_instances": int(num_eval_episodes),
            "sampled_source_indices": [int(i) for i in sampled_source_indices],
        },
    )

    for eval_idx, (source_idx, instance) in enumerate(zip(sampled_source_indices, sampled_instances)):
        result = run_ga_search_on_instance(
            instance=instance,
            cfg=cfg,
            reward_cfg=reward_cfg,
            rng=rng,   # 共享同一个 master rng，表示“该规模只跑一个 seed”
        )

        best = result["best_chromosome"]
        buf_str = " ".join(str(b) for b in best.buffers)

        detail_rows.append(
            dict(
                eval_idx=int(eval_idx),
                source_instance_idx=int(source_idx),
                buffers=buf_str,
                total_buffer=float(result["best_total_buffer"]),
                makespan=float(result["best_makespan"]),
                deadlock=int(result["best_deadlock"]),
                elapsed_sec=float(result["elapsed_sec"]),
                best_fitness=float(result["best_fitness"]),
                num_generations=int(result["num_generations"]),
                stop_reason=str(result["stop_reason"]),
            )
        )

        for row in result["trace_rows"]:
            all_trace_rows.append(
                dict(
                    eval_idx=int(eval_idx),
                    source_instance_idx=int(source_idx),
                    **row,
                )
            )

        total_makespan += float(result["best_makespan"])
        num_deadlocks += int(result["best_deadlock"])
        total_buffers.append(float(result["best_total_buffer"]))
        elapsed_list.append(float(result["elapsed_sec"]))

        print(
            f"[GA][{cfg.experiment_name}][eval={eval_idx:02d} | src={source_idx:03d}] "
            f"makespan={result['best_makespan']:.3f}, "
            f"deadlock={int(result['best_deadlock'])}, "
            f"total_buffer={result['best_total_buffer']:.1f}, "
            f"elapsed={result['elapsed_sec']:.3f}s, "
            f"gens={result['num_generations']}, "
            f"stop={result['stop_reason']}"
        )

    avg_makespan = total_makespan / max(1, num_eval_episodes)
    deadlock_rate = num_deadlocks / max(1, num_eval_episodes)
    avg_total_buffer = float(np.mean(total_buffers)) if total_buffers else math.nan
    std_total_buffer = float(np.std(total_buffers)) if total_buffers else math.nan
    avg_elapsed_sec_per_instance = float(np.mean(elapsed_list)) if elapsed_list else math.nan

    trace_csv = out_dir / "ga_trace.csv"
    write_csv(
        trace_csv,
        header=[
            "eval_idx",
            "source_instance_idx",
            "gen",
            "elapsed_sec",
            "best_fitness",
            "best_makespan",
            "best_deadlock",
            "best_total_buffer",
        ],
        rows=all_trace_rows,
    )

    detail_csv = out_dir / "eval_test_detail.csv"
    write_csv(
        detail_csv,
        header=[
            "eval_idx",
            "source_instance_idx",
            "buffers",
            "total_buffer",
            "makespan",
            "deadlock",
            "elapsed_sec",
            "best_fitness",
            "num_generations",
            "stop_reason",
        ],
        rows=detail_rows,
    )

    summary_csv = out_dir / "eval_test_summary.csv"
    summary_rows = [
        dict(
            experiment_name=str(cfg.experiment_name),
            random_seed=int(cfg.random_seed),
            num_candidates_total=int(len(test_instances)),
            num_eval_episodes=int(num_eval_episodes),
            avg_makespan=float(avg_makespan),
            deadlock_rate=float(deadlock_rate),
            num_deadlocks=int(num_deadlocks),
            avg_total_buffer=float(avg_total_buffer),
            std_total_buffer=float(std_total_buffer),
            avg_elapsed_sec_per_instance=float(avg_elapsed_sec_per_instance),
            time_budget_sec=float(cfg.time_budget_sec),
            max_generations=int(cfg.max_generations),
        )
    ]
    write_csv(
        summary_csv,
        header=[
            "experiment_name",
            "random_seed",
            "num_candidates_total",
            "num_eval_episodes",
            "avg_makespan",
            "deadlock_rate",
            "num_deadlocks",
            "avg_total_buffer",
            "std_total_buffer",
            "avg_elapsed_sec_per_instance",
            "time_budget_sec",
            "max_generations",
        ],
        rows=summary_rows,
    )

    print(
        f"[GA][SUMMARY][{cfg.experiment_name}] "
        f"episodes={num_eval_episodes}, "
        f"avg_makespan={avg_makespan:.3f}, "
        f"deadlock_rate={deadlock_rate:.3f}, "
        f"avg_total_buffer={avg_total_buffer:.3f}, "
        f"avg_elapsed_sec_per_instance={avg_elapsed_sec_per_instance:.3f}, "
        f"time_budget_sec={cfg.time_budget_sec:.1f}, "
        f"max_generations={cfg.max_generations}, "
        f"seed={cfg.random_seed}"
    )
    print(f"[SAVE] {sampled_json}")
    print(f"[SAVE] {trace_csv}")
    print(f"[SAVE] {detail_csv}")
    print(f"[SAVE] {summary_csv}")

# ============================================================
# 运行入口
# ============================================================

def build_ga_config(
    experiment_name: str,
    seed: int = 0,
    time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC,
    method_name: str = DEFAULT_METHOD_NAME,
) -> GABaselineConfig:
    return GABaselineConfig(
        experiment_name=experiment_name,
        method_name=method_name,
        random_seed=int(seed),
        pop_size=DEFAULT_POP_SIZE,
        elite_size=DEFAULT_ELITE_SIZE,
        tournament_size=DEFAULT_TOURNAMENT_SIZE,
        crossover_prob=DEFAULT_CROSSOVER_PROB,
        mutation_prob=DEFAULT_MUTATION_PROB,
        priority_mut_sigma=DEFAULT_PRIORITY_MUT_SIGMA,
        buffer_min=DEFAULT_BUFFER_MIN,
        buffer_max=DEFAULT_BUFFER_MAX,
        time_budget_sec=float(time_budget_sec),
        max_generations=DEFAULT_MAX_GENERATIONS,
        buffer_cost_weight=DEFAULT_BUFFER_COST_WEIGHT,
        deadlock_penalty=DEFAULT_DEADLOCK_PENALTY,
        num_eval_episodes=DEFAULT_NUM_EVAL_EPISODES,
        max_lower_steps=DEFAULT_MAX_LOWER_STEPS,
    )


def run_ga_for_experiment(
    experiment_name: str,
    seed: int = 0,
    time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC,
    method_name: str = DEFAULT_METHOD_NAME,
    run_id: Optional[str] = None,
) -> None:
    _, _, test_instances = load_instances_for_experiment(experiment_name)

    cfg = build_ga_config(
        experiment_name=experiment_name,
        seed=seed,
        time_budget_sec=time_budget_sec,
        method_name=method_name,
    )

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / experiment_name / method_name / f"seed{seed}_{run_id}"
    ensure_dir(out_dir)

    save_json(
        out_dir / "cfg.json",
        {
            **asdict(cfg),
            "project_root": str(PROJECT_ROOT),
            "out_dir": str(out_dir),
        },
    )

    print(
        f"[GA-RUN] experiment={experiment_name}, seed={seed}, "
        f"time_budget_sec={time_budget_sec}, out_dir={out_dir}"
    )

    t0 = time.perf_counter()
    evaluate_ga_on_test(cfg=cfg, test_instances=test_instances, out_dir=out_dir)
    t1 = time.perf_counter()

    save_json(
        out_dir / "runtime_train.json",
        {
            # GA 无训练阶段，这里保留统一字段，便于后续汇总
            "train_wall_clock_sec": 0.0,
            "note": "GA has no training phase; optimization is performed per test instance.",
            "total_wall_clock_sec_end_to_end": float(t1 - t0),
        },
    )

def run_ga_for_all_experiments(
    seed: int = 0,
    time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC,
    method_name: str = DEFAULT_METHOD_NAME,
    experiment_names: Optional[Sequence[str]] = None,
) -> None:
    if experiment_names is None:
        experiment_names = DEFAULT_EXPERIMENT_NAMES

    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_batch_0 = time.perf_counter()

    print(
        f"[GA-BATCH] start | seed={seed}, time_budget_sec={time_budget_sec}, "
        f"num_experiments={len(experiment_names)}, batch_run_id={batch_run_id}"
    )

    for exp_name in experiment_names:
        print(f"\n{'=' * 80}")
        print(f"[GA-BATCH] running experiment: {exp_name}")
        print(f"{'=' * 80}")

        run_ga_for_experiment(
            experiment_name=str(exp_name),
            seed=seed,
            time_budget_sec=time_budget_sec,
            method_name=method_name,
            run_id=batch_run_id,
        )

    t_batch_1 = time.perf_counter()
    print(
        f"\n[GA-BATCH] finished | total_wall_clock_sec={t_batch_1 - t_batch_0:.3f}"
    )

if __name__ == "__main__":
    run_ga_for_all_experiments(
        seed=0,
        time_budget_sec=300.0,
        method_name="ga_static",
    )
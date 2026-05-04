# examples/plot_upper_arch_ablation_dashboard.py
"""
批量版 upper-arch-ablation 可视化仪表盘

功能：
    - 给定 algo_type 和 replay_type，
      自动在 results/j50s3m3/upper_arch_ablation 下找到
      algo_<algo>_buffer_<replay>_seedX 这些子目录；
    - 每个 seed 目录中读取：
        - train_log.csv（如果存在）
        - offline_buffer_eval_val.csv（必须存在）
    - 生成 5 张“大图”，每张图里是 2x5 的 10 个 seed 子图：
        1) episode cost (-reward) + epsilon_U
        2) makespan & total_buffer
        3) deadlock rate
        4) offline buffer evaluation
        5) last-N episode buffer distribution
    - 图片会保存到 ROOT_DIR / "plots" 目录下。

使用方式：
    1. 修改 ROOT_DIR / ALGO / REPLAY。
    2. python plot_upper_arch_ablation_dashboard.py
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 配置区域 =========
# 批量结果根目录
ROOT_DIR = Path("results/j50s3m3/upper_arch_ablation") #C:/Users/Geist/Dropbox/1. 科研/2. 调度/7. Dynamic buffers/dynamic_buffers/

# 选择要画图的算法 & replay 类型
ALGO = "d3qn"      # 例如: "dqn" / "ddqn" / "d3qn"
REPLAY = "nstep"   # 例如: "uniform" / "per" / "nstep"

# 训练曲线的滑动平均窗口
MA_WINDOW = 50

# 统计“终局 buffer 分布”时使用的最后多少个 outer episode
LAST_N_EPISODES_FOR_BUFFERS = 200
# ===========================


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """简单滑动平均，忽略 NaN。"""
    return series.rolling(window, min_periods=1).mean()


def find_seed_dirs(root: Path, algo: str, replay: str) -> List[Path]:
    """
    在 root 下寻找所有 algo_<algo>_buffer_<replay>_seedX 形式的子目录。
    返回按 seed 升序排序的 Path 列表。
    """
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {root}")

    prefix = f"algo_{algo}_buffer_{replay}_seed"
    dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            dirs.append(p)

    if not dirs:
        raise FileNotFoundError(f"在 {root} 下没有找到前缀为 {prefix} 的子目录。")

    # 按 seed 数字排序
    def parse_seed(name: str) -> int:
        # 形如 algo_d3qn_buffer_nstep_seed7
        if "seed" in name:
            try:
                return int(name.split("seed")[-1])
            except ValueError:
                return 0
        return 0

    dirs.sort(key=lambda x: parse_seed(x.name))
    return dirs


def load_csvs(log_dir: Path) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    """
    加载某个 seed 目录下的 train_log.csv（如果存在）
    和 offline_buffer_eval_val.csv（必须存在）。
    """
    train_path = log_dir / "train_log.csv"
    offline_path = log_dir / "offline_buffer_eval_val.csv"

    train = None
    if train_path.exists():
        train = pd.read_csv(train_path)

    if not offline_path.exists():
        raise FileNotFoundError(f"{offline_path} 不存在（需要 offline_buffer_eval_val.csv）")

    offline = pd.read_csv(offline_path)
    return train, offline


def load_all_runs(seed_dirs: List[Path]) -> List[Dict[str, Any]]:
    """对每个 seed 目录读取 train / offline 数据。"""
    runs: List[Dict[str, Any]] = []
    for d in seed_dirs:
        train, offline = load_csvs(d)
        # 解析 seed id
        name = d.name
        seed = 0
        if "seed" in name:
            try:
                seed = int(name.split("seed")[-1])
            except ValueError:
                seed = 0
        runs.append(dict(seed=seed, dir=d, train=train, offline=offline))
    return runs


def make_grid_fig(title: str) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建一个 2x5 的 subplot 网格，并返回 (fig, axes_flat)。
    即便 seed 数小于 10，也保持这个布局，多余的 axes 会隐藏。
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    axes_flat = axes.flatten()
    fig.suptitle(title)
    return fig, axes_flat


# ===== 各类大图的绘制函数 =====

def plot_grid_cost(runs: List[Dict[str, Any]], save_path: Path):
    """大图 1：episode cost (-reward) + epsilon_U（每个 seed 一个子图）"""
    fig, axes = make_grid_fig(f"{ALGO} + {REPLAY}: episode cost (-reward)")

    for idx, run in enumerate(runs):
        ax = axes[idx]
        train = run["train"]
        seed = run["seed"]

        if train is None:
            ax.set_title(f"seed={seed} (no train_log)")
            ax.axis("off")
            continue

        outer_ep = train["outer_ep"]

        if "upper_ep_reward" in train.columns:
            ep_reward = train["upper_ep_reward"]
        elif "ep_reward_U" in train.columns:
            ep_reward = train["ep_reward_U"]
        else:
            ax.set_title(f"seed={seed} (no ep_reward)")
            ax.axis("off")
            continue

        ep_cost = -ep_reward
        ep_cost_ma = moving_average(ep_cost, MA_WINDOW)

        ax.plot(outer_ep, ep_cost, alpha=0.2, label="cost (raw)")
        ax.plot(outer_ep, ep_cost_ma, label=f"cost (MA={MA_WINDOW})")
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("outer episode")
        ax.set_ylabel("ep_cost")

        if "epsilon_U" in train.columns:
            ax2 = ax.twinx()
            ax2.plot(outer_ep, train["epsilon_U"], linestyle="--", alpha=0.6, label="epsilon_U")
            ax2.set_ylabel("epsilon_U")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend(loc="best")

    # 多余的 subplot 清空
    for j in range(len(runs), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[INFO] saved: {save_path}")


def plot_grid_makespan_and_buffer(runs: List[Dict[str, Any]], save_path: Path):
    """大图 2：makespan & total_buffer（每个 seed 一个子图）"""
    fig, axes = make_grid_fig(f"{ALGO} + {REPLAY}: makespan & total buffer")

    for idx, run in enumerate(runs):
        ax = axes[idx]
        train = run["train"]
        seed = run["seed"]

        if train is None:
            ax.set_title(f"seed={seed} (no train_log)")
            ax.axis("off")
            continue

        outer_ep = train["outer_ep"]
        makespan = train["final_makespan"].copy()

        if "deadlock" in train.columns:
            deadlock_flag = train["deadlock"].astype(int)
            makespan = makespan.where(deadlock_flag == 0, np.nan)

        makespan_ma = moving_average(makespan, MA_WINDOW)
        total_buffer_ma = moving_average(train["total_buffer"], MA_WINDOW)

        ax.plot(outer_ep, makespan, alpha=0.2, label="makespan (raw)")
        ax.plot(outer_ep, makespan_ma, label=f"makespan (MA={MA_WINDOW})")
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("outer episode")
        ax.set_ylabel("final_makespan")

        ax2 = ax.twinx()
        ax2.plot(outer_ep, total_buffer_ma, linestyle="--", label="total_buffer (MA)")
        ax2.set_ylabel("total_buffer (MA)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    for j in range(len(runs), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[INFO] saved: {save_path}")


def plot_grid_deadlock(runs: List[Dict[str, Any]], save_path: Path):
    """大图 3：deadlock rate（每个 seed 一个子图）"""
    fig, axes = make_grid_fig(f"{ALGO} + {REPLAY}: deadlock rate")

    for idx, run in enumerate(runs):
        ax = axes[idx]
        train = run["train"]
        seed = run["seed"]

        if train is None or "deadlock" not in train.columns:
            ax.set_title(f"seed={seed} (no deadlock info)")
            ax.axis("off")
            continue

        outer_ep = train["outer_ep"]
        deadlock = train["deadlock"].astype(int)
        deadlock_ma = moving_average(deadlock, MA_WINDOW)

        ax.plot(outer_ep, deadlock_ma, label="deadlock rate (MA)")
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("outer episode")
        ax.set_ylabel("deadlock rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best")

    for j in range(len(runs), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[INFO] saved: {save_path}")


def plot_grid_offline_eval(runs: List[Dict[str, Any]], save_path: Path):
    """大图 4：offline buffer evaluation（每个 seed 一个子图）"""
    fig, axes = make_grid_fig(f"{ALGO} + {REPLAY}: offline buffer eval (validation)")

    for idx, run in enumerate(runs):
        ax = axes[idx]
        offline = run["offline"]
        seed = run["seed"]

        if "avg_makespan" not in offline.columns:
            ax.set_title(f"seed={seed} (no avg_makespan)")
            ax.axis("off")
            continue

        df = offline.copy()
        df["buffers_str"] = df["buffers"].astype(str)
        df = df.sort_values("avg_makespan").reset_index(drop=True)

        x = np.arange(len(df))
        ax.bar(x, df["avg_makespan"], alpha=0.7, label="avg_makespan")
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("buffer vector")
        ax.set_ylabel("avg_makespan")
        ax.set_xticks(x)
        ax.set_xticklabels(df["buffers_str"], rotation=45, ha="right", fontsize=8)

        if "deadlock_rate" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(
                x,
                df["deadlock_rate"],
                linestyle="--",
                marker="o",
                label="deadlock_rate",
            )
            ax2.set_ylabel("deadlock_rate")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend(loc="best")

    for j in range(len(runs), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[INFO] saved: {save_path}")


def plot_grid_buffer_distribution(runs: List[Dict[str, Any]], save_path: Path):
    """大图 5：最后 N 集合的 buffer 分布（每个 seed 一个子图）"""
    fig, axes = make_grid_fig(
        f"{ALGO} + {REPLAY}: buffer distribution in last {LAST_N_EPISODES_FOR_BUFFERS} episodes"
    )

    for idx, run in enumerate(runs):
        ax = axes[idx]
        train = run["train"]
        seed = run["seed"]

        if train is None or "buffers" not in train.columns:
            ax.set_title(f"seed={seed} (no buffers)")
            ax.axis("off")
            continue

        max_outer = train["outer_ep"].max()
        threshold = max_outer - LAST_N_EPISODES_FOR_BUFFERS
        last = train[train["outer_ep"] >= threshold].copy()

        last["buffer_key"] = last["buffers"].astype(str)
        counts = last["buffer_key"].value_counts().sort_values(ascending=False)

        x = np.arange(len(counts))
        ax.bar(x, counts.values)
        ax.set_title(f"seed={seed}")
        ax.set_xticks(x)
        ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("frequency")

    for j in range(len(runs), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[INFO] saved: {save_path}")


# ========== 主入口 ==========

def main():
    seed_dirs = find_seed_dirs(ROOT_DIR, ALGO, REPLAY)
    print("[INFO] found seed dirs:")
    for d in seed_dirs:
        print("   ", d)

    runs = load_all_runs(seed_dirs)

    plot_dir = ROOT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 依次画 5 张大图
    plot_grid_cost(
        runs, plot_dir / f"{ALGO}_{REPLAY}_cost_grid.png"
    )
    plot_grid_makespan_and_buffer(
        runs, plot_dir / f"{ALGO}_{REPLAY}_makespan_buffer_grid.png"
    )
    plot_grid_deadlock(
        runs, plot_dir / f"{ALGO}_{REPLAY}_deadlock_grid.png"
    )
    plot_grid_offline_eval(
        runs, plot_dir / f"{ALGO}_{REPLAY}_offline_eval_grid.png"
    )
    plot_grid_buffer_distribution(
        runs, plot_dir / f"{ALGO}_{REPLAY}_buffer_dist_grid.png"
    )

    # 如需交互查看可取消注释
    # plt.show()


if __name__ == "__main__":
    main()

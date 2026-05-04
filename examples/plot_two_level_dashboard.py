# examples/plot_two_level_dashboard.py

"""
Two-level RL: 训练过程 + offline 缓冲评估 可视化仪表盘

使用方式：
    1. 修改 LOG_DIR 为你某次实验的输出目录
       目录里需要至少有：
         - offline_buffer_eval_val.csv
       如果目录中还有：
         - train_log.csv
       则会额外画训练曲线和 buffer 分布。

    2. python plot_two_level_dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======== 配置这里 ========
# 实验输出目录（包含 offline_buffer_eval_val.csv，若有 train_log.csv 则会画训练曲线）
LOG_DIR = Path(r"./results/j50s3m3/two_level_dqn/20251123_171118")  # TODO: 换成你自己的目录

# 滑动平均窗口大小（outer episode 维度）
MA_WINDOW = 50

# 统计“终局 buffer 分布”时使用的最后多少个 outer episode
LAST_N_EPISODES_FOR_BUFFERS = 200
# ========================


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """简单滑动平均，忽略 NaN。"""
    return series.rolling(window, min_periods=1).mean()


def load_csvs(log_dir: Path):
    """
    加载 train_log.csv（如果存在）和 offline_buffer_eval_val.csv（必须存在）。
    返回 (train_df 或 None, offline_df)。
    """
    train_path = log_dir / "train_log.csv"
    offline_path = log_dir / "offline_buffer_eval_val.csv"

    if train_path.exists():
        print(f"[INFO] loading train_log from: {train_path}")
        train = pd.read_csv(train_path)
        print("[INFO] train_log columns:", list(train.columns))
    else:
        print(f"[WARN] train_log.csv not found at: {train_path}, 训练曲线将被跳过。")
        train = None

    if not offline_path.exists():
        raise FileNotFoundError(f"offline_buffer_eval_val.csv not found at: {offline_path}")

    print(f"[INFO] loading offline eval from: {offline_path}")
    offline = pd.read_csv(offline_path)
    print("[INFO] offline_eval columns:", list(offline.columns))

    return train, offline


def plot_train_reward_and_epsilon(train: pd.DataFrame):
    """图 1：训练“成本”（-reward）+ epsilon_U"""

    outer_ep = train["outer_ep"]

    # 兼容不同列名：优先用 upper_ep_reward，其次 ep_reward_U
    if "upper_ep_reward" in train.columns:
        ep_reward = train["upper_ep_reward"]
    elif "ep_reward_U" in train.columns:
        ep_reward = train["ep_reward_U"]
    else:
        print("[WARN] train_log 既没有 'upper_ep_reward' 也没有 'ep_reward_U'，跳过回报曲线。")
        return

    # ★ 在画图阶段对 reward 取负，视作“episode cost”
    ep_cost = -ep_reward
    ep_cost_ma = moving_average(ep_cost, MA_WINDOW)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title("Upper-level training: episode cost (-episode_reward)")
    ax1.plot(outer_ep, ep_cost, alpha=0.2, label="ep_cost (raw)")
    ax1.plot(
        outer_ep,
        ep_cost_ma,
        label=f"ep_cost (MA, window={MA_WINDOW})",
    )
    ax1.set_xlabel("outer episode")
    ax1.set_ylabel("ep_cost")

    if "epsilon_U" in train.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            outer_ep,
            train["epsilon_U"],
            linestyle="--",
            alpha=0.7,
            label="epsilon_U",
        )
        ax2.set_ylabel("epsilon_U")

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")



def plot_train_makespan_and_buffer(train: pd.DataFrame):
    """图 2：训练 makespan（可选过滤 deadlock）+ total_buffer"""
    outer_ep = train["outer_ep"]

    makespan = train["final_makespan"].copy()
    # 如果有 deadlock 列，把 deadlock 的 makespan 设为 NaN，这样不会影响 MA
    if "deadlock" in train.columns:
        deadlock_flag = train["deadlock"].astype(int)
        makespan = makespan.where(deadlock_flag == 0, np.nan)

    makespan_ma = moving_average(makespan, MA_WINDOW)
    total_buffer_ma = moving_average(train["total_buffer"], MA_WINDOW)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title("Upper-level training: makespan & total buffer")
    ax1.plot(outer_ep, makespan, alpha=0.2, label="final_makespan (raw)")
    ax1.plot(
        outer_ep,
        makespan_ma,
        label=f"final_makespan (MA, window={MA_WINDOW})",
    )
    ax1.set_xlabel("outer episode")
    ax1.set_ylabel("final_makespan")

    ax2 = ax1.twinx()
    ax2.plot(
        outer_ep,
        total_buffer_ma,
        linestyle="--",
        label="total_buffer (MA)",
    )
    ax2.set_ylabel("total_buffer (MA)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()


def plot_train_deadlock_rate(train: pd.DataFrame):
    """图 3：训练 deadlock 率（如果有 deadlock 列）"""
    if "deadlock" not in train.columns:
        print("[WARN] train_log 没有 'deadlock' 列，跳过 deadlock 率图。")
        return

    outer_ep = train["outer_ep"]
    deadlock = train["deadlock"].astype(int)
    deadlock_ma = moving_average(deadlock, MA_WINDOW)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title("Upper-level training: deadlock rate (moving average)")
    ax.plot(outer_ep, deadlock_ma, label="deadlock rate (MA)")
    ax.set_xlabel("outer episode")
    ax.set_ylabel("deadlock rate (0~1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()


def plot_offline_eval(offline: pd.DataFrame):
    """
    图 4：offline buffer 枚举评估结果
    - x 轴为不同的 buffer 向量
    - 主轴为 avg_makespan
    - 副轴为 deadlock_rate（如果有）
    """
    if "avg_makespan" not in offline.columns:
        print("[WARN] offline_eval 没有 'avg_makespan' 列，跳过 offline 评估图。")
        return

    # 按 avg_makespan 升序排序，方便查看最优方案
    df = offline.copy()
    df["buffers_str"] = df["buffers"].astype(str)
    df = df.sort_values("avg_makespan").reset_index(drop=True)

    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title("Offline buffer evaluation on validation set")

    ax1.bar(x, df["avg_makespan"], alpha=0.7, label="avg_makespan")
    ax1.set_xlabel("buffer vector")
    ax1.set_ylabel("avg_makespan")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["buffers_str"], rotation=45, ha="right")

    # deadlock_rate 作为副轴（如果有的话）
    if "deadlock_rate" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            df["deadlock_rate"],
            linestyle="--",
            marker="o",
            label="deadlock_rate",
        )
        ax2.set_ylabel("deadlock_rate")

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    fig.tight_layout()

    # 在终端打印一下最优无死锁方案
    if "deadlock_rate" in df.columns:
        safe = df[df["deadlock_rate"] == 0]
        if not safe.empty:
            best = safe.loc[safe["avg_makespan"].idxmin()]
            print(
                f"[INFO] Offline best (deadlock-free): buffers={best['buffers_str']}, "
                f"avg_makespan={best['avg_makespan']:.3f}, "
                f"avg_total_buffer={best.get('avg_total_buffer', np.nan)}"
            )


def plot_final_buffer_distribution(train: pd.DataFrame):
    """图 5：最后 N 集中的 buffer 分布柱状图（依赖 train_log）"""
    if "buffers" not in train.columns:
        print("[WARN] train_log 没有 'buffers' 列，跳过 buffer 分布图。")
        return

    max_outer = train["outer_ep"].max()
    threshold = max_outer - LAST_N_EPISODES_FOR_BUFFERS
    last = train[train["outer_ep"] >= threshold].copy()

    # 将 buffers 转成字符串 key，例如 "1 1" / "3 2"
    last["buffer_key"] = last["buffers"].astype(str)
    counts = last["buffer_key"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(
        f"Buffer distribution in last {LAST_N_EPISODES_FOR_BUFFERS} episodes"
    )
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel("frequency")
    fig.tight_layout()

    print("=== Buffer distribution in last episodes ===")
    for buf, c in counts.items():
        print(f"{buf} : {c}")


def main():
    log_dir = LOG_DIR
    if not log_dir.exists():
        raise FileNotFoundError(f"LOG_DIR not found: {log_dir}")

    train, offline = load_csvs(log_dir)

    # 如果有 train_log.csv，就画训练相关的曲线
    if train is not None:
        plot_train_reward_and_epsilon(train)
        plot_train_makespan_and_buffer(train)
        plot_train_deadlock_rate(train)
        plot_final_buffer_distribution(train)
    else:
        print("[INFO] no train_log.csv, 只绘制 offline 评估图。")

    # 无论有没有 train_log，都画 offline buffer 评估
    plot_offline_eval(offline)

    plt.show()


if __name__ == "__main__":
    main()

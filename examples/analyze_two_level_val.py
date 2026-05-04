#examples/analyze_two_level_val.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out_dir = Path("results/j50s3m3/two_level_dqn/20251123_152431")  # 根据你的 out_dir 实际路径改
    val_log_path = out_dir / "val_log.csv"

    df = pd.read_csv(val_log_path)
    print(df.head())

    plt.figure(figsize=(10, 4))
    plt.plot(df["outer_ep"], df["avg_makespan"], label="val avg makespan")
    plt.xlabel("outer episode")
    plt.ylabel("avg makespan (validation)")
    plt.title("Two-level RL: validation performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

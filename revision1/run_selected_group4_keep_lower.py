# revision1/run_selected_group4_keep_lower.py
# 作用：
# 1) 一次性串行补跑 3 个选中的 group4 two-level 模型
# 2) 使用新的 method_name，避免覆盖旧结果
# 3) 前提：examples/group4_two_level.py 已经打好 lower_q_last.pth 的保存补丁

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# 把项目根目录加入 sys.path，保证可导入 examples/ 与 revision1/
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.group4_two_level import run_group4_for_experiment


# ============================================================
# 固定配置区（按当前确定内容封版）
# ============================================================

DEVICE_STR = "cuda"
METHOD_NAME = "group4_two_level_keep_lower"
NUM_OUTER_EPISODES = 400

# 已选代表 seed
SELECTED_RUNS = [
    ("j200s3", 3),
    ("j200s4", 0),
    ("j200s5", 0),
]


# ============================================================
# 主执行逻辑
# ============================================================

def main() -> None:
    print("=" * 200)
    print("[START] run_selected_group4_keep_lower")
    print(f"[TIME ] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[CONF ] DEVICE_STR={DEVICE_STR}")
    print(f"[CONF ] METHOD_NAME={METHOD_NAME}")
    print(f"[CONF ] NUM_OUTER_EPISODES={NUM_OUTER_EPISODES}")
    print(f"[CONF ] SELECTED_RUNS={SELECTED_RUNS}")
    print("=" * 200)

    success = []
    failed = []

    for idx, (experiment_name, seed) in enumerate(SELECTED_RUNS, start=1):
        print("\n" + "-" * 200)
        print(f"[TASK {idx}/{len(SELECTED_RUNS)}] experiment={experiment_name}, seed={seed}")
        print("-" * 200)

        try:
            run_group4_for_experiment(
                experiment_name=experiment_name,
                seeds=[seed],
                device_str=DEVICE_STR,
                method_name=METHOD_NAME,
                num_outer_episodes=NUM_OUTER_EPISODES,
            )
            success.append((experiment_name, seed))
            print(f"[DONE ] experiment={experiment_name}, seed={seed}")

        except Exception as e:
            failed.append((experiment_name, seed, str(e)))
            print(f"[ERROR] experiment={experiment_name}, seed={seed}")
            traceback.print_exc()

    print("\n" + "=" * 200)
    print("[FINISH] run_selected_group4_keep_lower")
    print(f"[TIME  ] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[OK    ] {success}")
    print(f"[FAIL  ] {failed}")
    print("=" * 200)


if __name__ == "__main__":
    main()
# revision1/run_selected_group4_keep_lower_and_infer.py
# 作用：
# 1) 串行补跑 3 个已选中的 group4 two-level 模型
# 2) 自动生成 selection 文件
# 3) 训练完成后直接对全部 12 种算例做推理
#
# 前提：
# - examples/group4_two_level.py 已打好补丁，会保存 lower_q_last.pth
# - revision1/select_group4_representative_seeds.py 已放到项目中
# - revision1/infer_group4_selected_j80_models.py 已放到项目中

from __future__ import annotations

import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------
# 把项目根目录加入 sys.path，保证可导入 examples/ 与 revision1/
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.group4_two_level import run_group4_for_experiment


# ============================================================
# 固定配置区
# ============================================================

DEVICE_STR = "cuda"
METHOD_NAME = "group4_two_level_keep_lower"
NUM_OUTER_EPISODES = 400
NUM_EVAL_EPISODES = 100

# 已选代表 seed
SELECTED_RUNS = [
    ("j80s3", 9),
    ("j80s4", 4),
    ("j80s5", 3),
]

# 输出目录
SELECTION_OUT_DIR = "results/revision1_j80_selection_keep_lower"
INFER_OUT_DIR = "results/revision1_j80_generalization_eval"

# 脚本路径
SELECT_SCRIPT = "revision1/select_group4_representative_seeds.py"
INFER_SCRIPT = "revision1/infer_group4_selected_j80_models.py"


# ============================================================
# 工具函数
# ============================================================

def run_cmd(cmd: list[str]) -> None:
    print("\n[CMD  ]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ============================================================
# 主逻辑
# ============================================================

def main() -> None:
    print("=" * 100)
    print("[START] run_selected_group4_keep_lower_and_infer")
    print(f"[TIME ] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[CONF ] DEVICE_STR={DEVICE_STR}")
    print(f"[CONF ] METHOD_NAME={METHOD_NAME}")
    print(f"[CONF ] NUM_OUTER_EPISODES={NUM_OUTER_EPISODES}")
    print(f"[CONF ] NUM_EVAL_EPISODES={NUM_EVAL_EPISODES}")
    print(f"[CONF ] SELECTED_RUNS={SELECTED_RUNS}")
    print("=" * 100)

    success = []
    failed = []

    # --------------------------------------------------------
    # Part A：串行补跑 3 个模型
    # --------------------------------------------------------
    for idx, (experiment_name, seed) in enumerate(SELECTED_RUNS, start=1):
        print("\n" + "-" * 100)
        print(f"[TRAIN {idx}/{len(SELECTED_RUNS)}] experiment={experiment_name}, seed={seed}")
        print("-" * 100)

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

    # 若有失败，则不进入后续推理
    if failed:
        print("\n" + "=" * 100)
        print("[STOP ] Some training runs failed. Skip selection and inference.")
        print(f"[OK   ] {success}")
        print(f"[FAIL ] {failed}")
        print("=" * 100)
        return

    # --------------------------------------------------------
    # Part B：自动生成 selection 文件
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("[STEP ] Build selection json from newly rerun models")
    print("=" * 100)

    run_cmd([
        sys.executable,
        SELECT_SCRIPT,
        "--results_root", "results",
        "--method_name", METHOD_NAME,
        "--experiments", "j80s3", "j80s4", "j80s5",
        "--out_dir", SELECTION_OUT_DIR,
    ])

    selection_json = str(Path(SELECTION_OUT_DIR) / "selected_j80_stage_models.json")

    # --------------------------------------------------------
    # Part C：直接推理到全部 12 种算例
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("[STEP ] Start inference on all 12 experiments")
    print("=" * 100)

    run_cmd([
        sys.executable,
        INFER_SCRIPT,
        "--selection_json", selection_json,
        "--device", DEVICE_STR,
        "--num_eval_episodes", str(NUM_EVAL_EPISODES),
        "--out_dir", INFER_OUT_DIR,
    ])

    print("\n" + "=" * 100)
    print("[FINISH] run_selected_group4_keep_lower_and_infer")
    print(f"[TIME  ] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[OK    ] {success}")
    print(f"[SELOUT] {selection_json}")
    print(f"[INFOUT] {INFER_OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
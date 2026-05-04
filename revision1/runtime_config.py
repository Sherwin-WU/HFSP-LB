# revision1/runtime_config.py
from __future__ import annotations

EXPERIMENTS = [
    "j50s3",
    "j50s4",
    "j50s5",
    "j80s3",
    "j80s4",
    "j80s5",
    "j160s3",
    "j160s4",
    "j160s5",
    "j200s3",
    "j200s4",
    "j200s5",
]

SEED = 0
DEVICE_STR = "cuda"

TRAIN_EPISODES = 200
INFER_REPEATS = 10
NUM_EVAL_EPISODES = 100

# 这一步只统计“旧对比方法”的 runtime
TRAINABLE_METHODS = [
    "UB-FIFO",
    "UB-LPT",
    "UB-SPT",
    "UB-SRPT",
    "LD-Agent",
    "UB+LD",
]

INFER_METHODS = [
    "FB-FIFO",
    "FB-LPT",
    "FB-SPT",
    "FB-SRPT",
    "UB-FIFO",
    "UB-LPT",
    "UB-SPT",
    "UB-SRPT",
    "LD-Agent",
    "UB+LD",
]

# Group2 的 rule 列表
GROUP2_RULES = ["fifo", "lpt", "spt", "srpt"]

# Group3 封版固定 buffers（直接冻结）
GROUP3_FIXED_BUFFERS = {
    "j50s3": [5, 4],
    "j50s4": [5, 4, 4],
    "j50s5": [5, 4, 4, 5],
    "j80s3": [5, 4],
    "j80s4": [5, 5, 5],
    "j80s5": [5, 5, 4, 5],
    "j160s3": [5, 5],
    "j160s4": [5, 3, 5],
    "j160s5": [5, 5, 5, 5],
    "j200s3": [4, 5],
    "j200s4": [5, 5, 4],
    "j200s5": [5, 4, 4, 5],
}

RESULT_ROOT_NAME = "results/revision1_runtime"

# 第一版先统计端到端 runtime；更细 forward latency 后续再加
ENABLE_FINE_GRAINED_FORWARD_TIMING = False
USE_CUDA_SYNC = True
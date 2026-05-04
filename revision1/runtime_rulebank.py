# revision1/runtime_rulebank.py
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_GROUP1_SUMMARY = (
    ROOT_DIR / "results" / "summary" / "group1_rule_baseline_all_rules.csv"
)

RULE_NAME_MAP = {
    "fifo": "FB-FIFO",
    "lpt": "FB-LPT",
    "spt": "FB-SPT",
    "srpt": "FB-SRPT",
}


def _parse_buffers(text: str) -> List[int]:
    """
    兼容多种 best_buffers 存储格式，例如：
      "[5, 5]"
      "(5, 5)"
      "5,5"
      "5 5"
      "5;5"
      "5\t5"

    优先尝试 ast.literal_eval；
    若失败，则退化为正则提取整数。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty best_buffers string.")

    # 1) 先尝试 Python 字面量格式
    try:
        val = ast.literal_eval(text)
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        if isinstance(val, int):
            return [int(val)]
    except Exception:
        pass

    # 2) 再兼容空格/逗号/分号等分隔格式
    nums = re.findall(r"-?\d+", text)
    if nums:
        return [int(x) for x in nums]

    raise ValueError(f"Failed to parse best_buffers='{text}'")


def load_rule_baseline_buffers(
    summary_csv: str | Path | None = None,
) -> Dict[str, Dict[str, List[int]]]:
    """
    从 results/summary/group1_rule_baseline_all_rules.csv 读取：
      experiment -> { "FB-FIFO": [...], "FB-LPT": [...], "FB-SPT": [...], "FB-SRPT": [...] }
    """
    csv_path = Path(summary_csv) if summary_csv is not None else DEFAULT_GROUP1_SUMMARY
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Group1 summary csv not found: {csv_path}\n"
            f"请先确保你已有 results/summary/group1_rule_baseline_all_rules.csv"
        )

    out: Dict[str, Dict[str, List[int]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = (row.get("experiment") or "").strip()
            rule = (row.get("rule") or "").strip().lower()
            best_buffers = row.get("best_buffers") or ""

            if not exp or not rule:
                continue
            if rule not in RULE_NAME_MAP:
                continue

            method_name = RULE_NAME_MAP[rule]
            buffers = _parse_buffers(best_buffers)
            if not buffers:
                raise RuntimeError(
                    f"Experiment '{exp}', rule '{rule}' parsed empty buffers from best_buffers='{best_buffers}'"
                )

            out.setdefault(exp, {})
            out[exp][method_name] = buffers

    # 基本完整性检查
    for exp, dd in out.items():
        missing = [m for m in RULE_NAME_MAP.values() if m not in dd]
        if missing:
            raise RuntimeError(
                f"Experiment '{exp}' missing Group1 rule buffers for: {missing}"
            )

    return out
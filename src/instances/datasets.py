# ============================
# 文件：src/instances/datasets.py
# 说明：
#   - 提供 InstanceData 列表的 train/val/test 划分
#   - 提供从目录批量加载 csv 算例的工具函数
#   - 若安装了 PyTorch，则提供 InstanceDataset 以便配合 DataLoader 使用
# ============================

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np

from .types import InstanceData
from .io import from_matrix_csv


# ============================
#  划分工具
# ============================

@dataclass
class InstanceSplit:
    """简单的 train/val/test 划分结果容器。"""

    train: List[InstanceData]
    val: List[InstanceData]
    test: List[InstanceData]


def split_instances(
    instances: Sequence[InstanceData],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> InstanceSplit:
    """
    将一组 InstanceData 按比例划分为 train/val/test。

    参数：
        instances:   原始实例列表
        train_ratio: 训练集比例（0~1）
        val_ratio:   验证集比例（0~1）
        shuffle:     是否在划分前打乱
        seed:        若 shuffle=True，可指定随机种子保证可复现性

    返回：
        InstanceSplit(train, val, test)
    """
    n = len(instances)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)  # 防止越界
    n_val = min(n_val, max(0, n - n_train))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [instances[i] for i in train_idx]
    val = [instances[i] for i in val_idx]
    test = [instances[i] for i in test_idx]

    return InstanceSplit(train=train, val=val, test=test)


# ============================
#  从目录加载算例
# ============================

def load_instances_from_dir(
    dir_path: str,
    pattern: str = "*.csv",
    devices_csv: Optional[str] = None,
    machines_per_stage: Optional[Sequence[int]] = None,
    sort: bool = True,
) -> List[InstanceData]:
    """
    从目录中批量读取矩阵 csv 算例，并转成 InstanceData 列表。

    参数：
        dir_path:          目录路径，例如 "src/instances/raw/train"
        pattern:           匹配模式，默认 "*.csv"
        devices_csv:       若 machines_per_stage 为 None，则可提供 devices_csv 用于推断
        machines_per_stage:若不为 None，则所有 csv 使用相同的设备数配置
        sort:              是否对匹配到的文件名排序（便于稳定复现）

    返回：
        List[InstanceData]
    """
    glob_pattern = os.path.join(dir_path, pattern)
    paths = glob.glob(glob_pattern)

    if sort:
        paths = sorted(paths)

    instances: List[InstanceData] = []
    for p in paths:
        inst = from_matrix_csv(
            matrix_csv=p,
            machines_per_stage=list(machines_per_stage) if machines_per_stage is not None else None,
            devices_csv=devices_csv,
        )
        instances.append(inst)

    return instances


# ============================
#  PyTorch Dataset（可选）
# ============================

try:
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - 如果环境没有装 torch，就让 Dataset 退化为 object
    class Dataset:  # type: ignore
        pass
    TORCH_AVAILABLE = False


class InstanceDataset(Dataset):
    """
    一个简单的 PyTorch Dataset 包装器。

    - 内部持有一组 InstanceData；
    - 可选 feature_fn: InstanceData -> 任意特征（例如上层 f(I)）。
    - __getitem__ 返回：
        若 feature_fn 为 None：返回 InstanceData；
        否则：返回 (InstanceData, features) 二元组。
    """

    def __init__(
        self,
        instances: Sequence[InstanceData],
        feature_fn: Optional[Callable[[InstanceData], Any]] = None,
    ) -> None:
        super().__init__()
        self._instances: List[InstanceData] = list(instances)
        self._feature_fn = feature_fn

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._instances)

    def __getitem__(self, idx: int) -> Any:  # type: ignore[override]
        inst = self._instances[idx]
        if self._feature_fn is None:
            return inst
        feats = self._feature_fn(inst)
        return inst, feats


# ============================
#  一些便捷构造函数（可选）
# ============================

def build_splitted_datasets(
    instances: Sequence[InstanceData],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
    feature_fn: Optional[Callable[[InstanceData], Any]] = None,
) -> Tuple[InstanceDataset, InstanceDataset, InstanceDataset]:
    """
    便捷函数：先划分，再包装成三个 InstanceDataset。

    返回：
        (train_dataset, val_dataset, test_dataset)
    """
    split = split_instances(
        instances=instances,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        shuffle=shuffle,
        seed=seed,
    )

    train_ds = InstanceDataset(split.train, feature_fn=feature_fn)
    val_ds = InstanceDataset(split.val, feature_fn=feature_fn)
    test_ds = InstanceDataset(split.test, feature_fn=feature_fn)

    return train_ds, val_ds, test_ds

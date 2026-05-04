# src/models/q_networks.py

from typing import List, Sequence, Optional

import torch
from torch import nn


class DQNNet(nn.Module):
    """
    Simple MLP-based DQN network used for both upper and lower agents.
    This is migrated from examples/train_two_level.py so the behaviour
    should remain identical.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (128, 128)

        layers: List[nn.Module] = []
        last_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingDQNNet(nn.Module):
    """
    Dueling-architecture DQN network.

    暂时还没有在训练脚本里使用，但后面做 D3QN / dueling 的时候会用到。
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (128, 128)

        layers: List[nn.Module] = []
        last_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.feature = nn.Sequential(*layers)

        # Value and advantage streams
        self.value_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_head(feat)            # [B, 1]
        advantage = self.adv_head(feat)         # [B, A]
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values

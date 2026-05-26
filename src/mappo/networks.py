from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """共享参数 Actor。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNetwork(nn.Module):
    """集中式 Critic。"""

    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked = masked.masked_fill(action_mask <= 0, -1e9)
    return masked


def sample_masked_actions(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = apply_action_mask(logits, action_mask)
    dist = Categorical(logits=masked_logits)
    if deterministic:
        actions = masked_logits.argmax(dim=-1)
    else:
        actions = dist.sample()
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    return actions, log_probs, entropy


def evaluate_masked_actions(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = apply_action_mask(logits, action_mask)
    dist = Categorical(logits=masked_logits)
    return dist.log_prob(actions), dist.entropy()

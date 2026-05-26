from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .config import MAPPOConfig


def save_checkpoint(
    path: str | Path,
    *,
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: Optional[Optimizer],
    critic_optimizer: Optional[Optimizer],
    config: MAPPOConfig,
    metadata: Dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict() if actor_optimizer is not None else None,
        "critic_optimizer_state_dict": critic_optimizer.state_dict() if critic_optimizer is not None else None,
        "mappo_config": config.to_dict(),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    *,
    actor: Optional[nn.Module] = None,
    critic: Optional[nn.Module] = None,
    actor_optimizer: Optional[Optimizer] = None,
    critic_optimizer: Optional[Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if actor is not None:
        actor.load_state_dict(checkpoint["actor_state_dict"])
    if critic is not None:
        critic.load_state_dict(checkpoint["critic_state_dict"])
    if actor_optimizer is not None and checkpoint.get("actor_optimizer_state_dict") is not None:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    if critic_optimizer is not None and checkpoint.get("critic_optimizer_state_dict") is not None:
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return checkpoint

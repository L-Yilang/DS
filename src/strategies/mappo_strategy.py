from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.mappo.action_space import MacroActionSpace
from src.mappo.checkpoint import load_checkpoint
from src.mappo.config import MAPPOConfig
from src.mappo.features import FeatureEncoder
from src.mappo.networks import ActorNetwork, sample_masked_actions
from .base import SchedulingStrategy, StrategyContext, VehiclePlan


class MAPPOSTrategy(SchedulingStrategy):
    """Inference wrapper for the MAPPO macro-action policy.

    A trained checkpoint is required for meaningful results. Set
    allow_untrained=True only for smoke tests that verify integration.
    """

    name = "mappo"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        deterministic: bool = True,
        allow_untrained: bool = False,
        config: Optional[MAPPOConfig] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.deterministic = deterministic
        self.allow_untrained = allow_untrained

        checkpoint = None
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            checkpoint = load_checkpoint(self.checkpoint_path, map_location="cpu")
            self.config = MAPPOConfig.from_dict(checkpoint.get("mappo_config", {}))
        elif self.checkpoint_path is not None and not allow_untrained:
            raise FileNotFoundError(f"MAPPO checkpoint not found: {self.checkpoint_path}")
        elif not allow_untrained:
            raise ValueError(
                "MAPPOSTrategy requires a checkpoint. Pass allow_untrained=True only for smoke tests."
            )
        else:
            self.config = config or MAPPOConfig()

        device_name = self.config.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.actor = ActorNetwork(
            obs_dim=self.config.actor_obs_dim,
            action_dim=self.config.action_dim,
            hidden_size=self.config.hidden_size,
        ).to(self.device)
        if checkpoint is not None:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.eval()

        self.feature_encoder = FeatureEncoder(self.config)
        self.action_space = MacroActionSpace(self.config)

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        encoded = self.feature_encoder.encode(context)
        action_mask = self.action_space.build_action_mask(context, encoded)
        encoded = encoded.with_action_mask(action_mask)

        if not encoded.agent_mask.any():
            return {}

        actor_obs = torch.as_tensor(
            encoded.actor_obs[encoded.agent_mask],
            dtype=torch.float32,
            device=self.device,
        )
        active_action_mask = torch.as_tensor(
            encoded.action_mask[encoded.agent_mask],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.actor(actor_obs)
            active_actions, _, _ = sample_masked_actions(
                logits,
                active_action_mask,
                deterministic=self.deterministic,
            )

        full_actions = np.zeros((self.config.max_agents,), dtype=np.int64)
        full_actions[encoded.agent_mask] = active_actions.cpu().numpy()
        decoded = self.action_space.decode(context, encoded, full_actions)
        return decoded.plans

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MAPPOConfig:
    """MAPPO 训练与推理配置。"""

    top_k_tasks: int = 5
    top_k_stations: int = 3
    max_agents: int = 16
    max_global_tasks: int = 16
    max_global_stations: int = 8

    hidden_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256

    episodes_per_stage: int = 24
    smoke_episodes_per_stage: int = 2
    eval_interval: int = 4
    eval_seeds: int = 2

    invalid_action_penalty: float = 2.5
    duplicate_task_penalty: float = 1.0
    pending_pressure_penalty: float = 0.02
    extra_distance_penalty: float = 0.01

    checkpoint_dir: str = "outputs/mappo/checkpoints"
    eval_dir: str = "outputs/mappo"
    device: str = "cpu"
    deterministic_eval: bool = True
    seed: int = 20260525

    @property
    def action_dim(self) -> int:
        return 3 + self.top_k_tasks

    @property
    def actor_obs_dim(self) -> int:
        return 8 + self.top_k_tasks * 6 + self.top_k_stations * 4 + 6

    @property
    def critic_obs_dim(self) -> int:
        return 8 + self.max_agents * 5 + self.max_global_tasks * 4 + self.max_global_stations * 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MAPPOConfig":
        known = {field_name: data[field_name] for field_name in cls.__dataclass_fields__ if field_name in data}
        return cls(**known)


def build_preset_config(preset: str, *, device: str | None = None) -> MAPPOConfig:
    """构建训练预设。"""

    normalized = preset.lower().strip()
    config = MAPPOConfig(device=device or "cpu")
    if normalized == "smoke":
        return MAPPOConfig(
            device=device or "cpu",
            hidden_size=96,
            update_epochs=2,
            minibatch_size=64,
            episodes_per_stage=4,
            smoke_episodes_per_stage=2,
            eval_interval=1,
            eval_seeds=1,
        )
    if normalized == "default":
        return config
    raise ValueError(f"未知 MAPPO preset: {preset}")

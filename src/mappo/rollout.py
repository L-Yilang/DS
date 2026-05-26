from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List

import numpy as np
import torch

from .features import EncodedMAPPOState


@dataclass
class RolloutRecord:
    actor_obs: np.ndarray
    critic_obs: np.ndarray
    action_mask: np.ndarray
    agent_mask: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    reward: float
    done: bool
    value: float


class RolloutBuffer:
    """按时间步存储多智能体轨迹。"""

    def __init__(self) -> None:
        self.records: List[RolloutRecord] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def add(
        self,
        state: EncodedMAPPOState,
        *,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.records.append(
            RolloutRecord(
                actor_obs=state.actor_obs.copy(),
                critic_obs=state.critic_obs.copy(),
                action_mask=state.action_mask.copy(),
                agent_mask=state.agent_mask.copy(),
                actions=actions.copy(),
                log_probs=log_probs.copy(),
                reward=reward,
                done=done,
                value=value,
            )
        )

    def compute_returns_and_advantages(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        last_value: float,
    ) -> None:
        self.advantages = [0.0] * len(self.records)
        self.returns = [0.0] * len(self.records)
        gae = 0.0
        next_value = last_value

        for index in reversed(range(len(self.records))):
            record = self.records[index]
            non_terminal = 0.0 if record.done else 1.0
            delta = record.reward + gamma * next_value * non_terminal - record.value
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[index] = gae
            self.returns[index] = gae + record.value
            next_value = record.value

    def build_training_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        actor_obs_rows: List[np.ndarray] = []
        critic_obs_rows: List[np.ndarray] = []
        action_rows: List[np.ndarray] = []
        action_mask_rows: List[np.ndarray] = []
        old_log_prob_rows: List[np.ndarray] = []
        advantage_rows: List[np.ndarray] = []
        return_rows: List[np.ndarray] = []

        for index, record in enumerate(self.records):
            active_indices = np.where(record.agent_mask)[0]
            if len(active_indices) == 0:
                continue

            actor_obs_rows.append(record.actor_obs[active_indices])
            critic_obs_rows.append(np.repeat(record.critic_obs[None, :], len(active_indices), axis=0))
            action_rows.append(record.actions[active_indices][:, None])
            action_mask_rows.append(record.action_mask[active_indices])
            old_log_prob_rows.append(record.log_probs[active_indices][:, None])
            advantage_rows.append(np.full((len(active_indices), 1), self.advantages[index], dtype=np.float32))
            return_rows.append(np.full((len(active_indices), 1), self.returns[index], dtype=np.float32))

        if not actor_obs_rows:
            raise ValueError("RolloutBuffer 为空，无法构造训练张量")

        actor_obs = np.concatenate(actor_obs_rows, axis=0).astype(np.float32)
        critic_obs = np.concatenate(critic_obs_rows, axis=0).astype(np.float32)
        actions = np.concatenate(action_rows, axis=0).astype(np.int64).reshape(-1)
        action_masks = np.concatenate(action_mask_rows, axis=0).astype(np.float32)
        old_log_probs = np.concatenate(old_log_prob_rows, axis=0).astype(np.float32).reshape(-1)
        advantages = np.concatenate(advantage_rows, axis=0).astype(np.float32).reshape(-1)
        returns = np.concatenate(return_rows, axis=0).astype(np.float32).reshape(-1)

        return {
            "actor_obs": torch.as_tensor(actor_obs, device=device),
            "critic_obs": torch.as_tensor(critic_obs, device=device),
            "actions": torch.as_tensor(actions, device=device),
            "action_masks": torch.as_tensor(action_masks, device=device),
            "old_log_probs": torch.as_tensor(old_log_probs, device=device),
            "advantages": torch.as_tensor(advantages, device=device),
            "returns": torch.as_tensor(returns, device=device),
        }


def iterate_minibatches(
    batch: Dict[str, torch.Tensor],
    minibatch_size: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    total = batch["actions"].shape[0]
    indices = torch.randperm(total, device=batch["actions"].device)
    for start in range(0, total, minibatch_size):
        chosen = indices[start : start + minibatch_size]
        yield {key: value[chosen] for key, value in batch.items()}

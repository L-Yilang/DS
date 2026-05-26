from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ..config import ScaleConfig, SimulationConfig
from .checkpoint import save_checkpoint
from .config import MAPPOConfig
from .env_adapter import MAPPOEnvAdapter
from .networks import ActorNetwork, CriticNetwork, evaluate_masked_actions, sample_masked_actions
from .rollout import RolloutBuffer, iterate_minibatches


class MAPPOTrainer:
    """单机版 MAPPO 训练器。"""

    def __init__(
        self,
        *,
        scales: Dict[str, ScaleConfig],
        sim_config: SimulationConfig,
        mappo_config: MAPPOConfig,
    ) -> None:
        self.scales = scales
        self.sim_config = sim_config
        self.config = mappo_config

        device_name = mappo_config.device
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
        self.critic = CriticNetwork(
            obs_dim=self.config.critic_obs_dim,
            hidden_size=self.config.hidden_size,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    def train(
        self,
        *,
        preset: str = "default",
        stage_names: Optional[List[str]] = None,
    ) -> List[dict]:
        output_root = Path(self.config.eval_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        history: List[dict] = []
        best_score = float("-inf")
        selected_stages = stage_names or ["small", "medium", "large"]

        for stage_name in selected_stages:
            if stage_name not in self.scales:
                continue
            scale = self.scales[stage_name]
            episode_count = (
                self.config.smoke_episodes_per_stage
                if preset == "smoke"
                else self.config.episodes_per_stage
            )

            stage_best_score = float("-inf")
            for episode_index in range(1, episode_count + 1):
                episode_seed = scale.seed + episode_index
                env = MAPPOEnvAdapter(
                    scale=scale,
                    sim_config=self.sim_config,
                    mappo_config=self.config,
                )
                state = env.reset(seed=episode_seed)
                buffer = RolloutBuffer()
                episode_reward = 0.0
                episode_step_count = 0
                done = False

                while not done:
                    active_actor_obs = torch.as_tensor(
                        state.actor_obs[state.agent_mask],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    active_action_mask = torch.as_tensor(
                        state.action_mask[state.agent_mask],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    critic_obs = torch.as_tensor(
                        state.critic_obs,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                    with torch.no_grad():
                        logits = self.actor(active_actor_obs)
                        actions_tensor, log_probs_tensor, _ = sample_masked_actions(
                            logits,
                            active_action_mask,
                            deterministic=False,
                        )
                        value = self.critic(critic_obs).item()

                    full_actions = np.zeros((self.config.max_agents,), dtype=np.int64)
                    full_log_probs = np.zeros((self.config.max_agents,), dtype=np.float32)
                    full_actions[state.agent_mask] = actions_tensor.cpu().numpy()
                    full_log_probs[state.agent_mask] = log_probs_tensor.cpu().numpy()

                    next_state, reward, done, info = env.step(full_actions)
                    buffer.add(
                        state,
                        actions=full_actions,
                        log_probs=full_log_probs,
                        reward=reward,
                        done=done,
                        value=value,
                    )
                    episode_reward += reward
                    episode_step_count += 1
                    state = next_state

                last_value = 0.0
                buffer.compute_returns_and_advantages(
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    last_value=last_value,
                )
                update_metrics = self._update_from_buffer(buffer)
                result = env.episode_result()

                history_row = {
                    "stage": stage_name,
                    "episode": episode_index,
                    "seed": episode_seed,
                    "episode_reward": round(episode_reward, 4),
                    "total_score": result.total_score,
                    "completed_tasks": result.completed_tasks,
                    "overdue_tasks": result.overdue_tasks,
                    "timeout_rate": result.timeout_rate,
                    "total_distance": result.total_distance,
                    "simulation_failed": result.simulation_failed,
                    "steps": episode_step_count,
                    **update_metrics,
                }
                history.append(history_row)

                if result.total_score > stage_best_score:
                    stage_best_score = result.total_score
                    save_checkpoint(
                        Path(self.config.checkpoint_dir) / f"best_{stage_name}.pt",
                        actor=self.actor,
                        critic=self.critic,
                        actor_optimizer=self.actor_optimizer,
                        critic_optimizer=self.critic_optimizer,
                        config=self.config,
                        metadata=history_row,
                    )

                if result.total_score > best_score:
                    best_score = result.total_score
                    save_checkpoint(
                        Path(self.config.checkpoint_dir) / "best.pt",
                        actor=self.actor,
                        critic=self.critic,
                        actor_optimizer=self.actor_optimizer,
                        critic_optimizer=self.critic_optimizer,
                        config=self.config,
                        metadata=history_row,
                    )

                save_checkpoint(
                    Path(self.config.checkpoint_dir) / "latest.pt",
                    actor=self.actor,
                    critic=self.critic,
                    actor_optimizer=self.actor_optimizer,
                    critic_optimizer=self.critic_optimizer,
                    config=self.config,
                    metadata=history_row,
                )

        self._write_history(history)
        return history

    def _update_from_buffer(self, buffer: RolloutBuffer) -> dict:
        batch = buffer.build_training_tensors(self.device)
        advantages = batch["advantages"]
        batch["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0
        update_steps = 0

        for _ in range(self.config.update_epochs):
            for mini_batch in iterate_minibatches(batch, self.config.minibatch_size):
                logits = self.actor(mini_batch["actor_obs"])
                new_log_probs, entropy = evaluate_masked_actions(
                    logits,
                    mini_batch["action_masks"],
                    mini_batch["actions"],
                )
                ratio = torch.exp(new_log_probs - mini_batch["old_log_probs"])
                unclipped = ratio * mini_batch["advantages"]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * mini_batch["advantages"]
                actor_loss = -torch.min(unclipped, clipped).mean()

                values = self.critic(mini_batch["critic_obs"]).squeeze(-1)
                critic_loss = F.mse_loss(values, mini_batch["returns"])
                entropy_bonus = entropy.mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss = (
                    actor_loss
                    + self.config.value_coef * critic_loss
                    - self.config.entropy_coef * entropy_bonus
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_total += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())
                entropy_total += float(entropy_bonus.item())
                update_steps += 1

        denom = max(1, update_steps)
        return {
            "actor_loss": round(actor_loss_total / denom, 6),
            "critic_loss": round(critic_loss_total / denom, 6),
            "entropy": round(entropy_total / denom, 6),
        }

    def _write_history(self, history: List[dict]) -> None:
        if not history:
            return

        output_path = Path(self.config.eval_dir) / "train_log.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

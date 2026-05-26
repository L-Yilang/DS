from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from ..config import ScaleConfig, SimulationConfig
from ..strategies.base import SchedulingStrategy, StrategyContext, VehiclePlan
from ..world import WorldManager
from .action_space import MacroActionSpace
from .config import MAPPOConfig
from .features import EncodedMAPPOState, FeatureEncoder


class _ExternalPlanStrategy(SchedulingStrategy):
    """占位策略；训练时由外部直接提供 plans。"""

    name = "mappo_external"

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        return {}


class MAPPOEnvAdapter:
    """把 WorldManager 包装成 MAPPO 可消费的训练环境。"""

    def __init__(
        self,
        *,
        scale: ScaleConfig,
        sim_config: SimulationConfig,
        mappo_config: MAPPOConfig,
    ) -> None:
        self.base_scale = scale
        self.sim_config = sim_config
        self.mappo_config = mappo_config
        self.feature_encoder = FeatureEncoder(mappo_config)
        self.action_space = MacroActionSpace(mappo_config)
        self.world: Optional[WorldManager] = None
        self.encoded_state: Optional[EncodedMAPPOState] = None

    def reset(self, *, seed: Optional[int] = None) -> EncodedMAPPOState:
        scale = replace(self.base_scale, seed=seed) if seed is not None else self.base_scale
        self.world = WorldManager(
            scale=scale,
            config=self.sim_config,
            strategy=_ExternalPlanStrategy(),
        )
        self.world.advance_to_next_decision()
        self.encoded_state = self._observe()
        return self.encoded_state

    def step(self, actions: np.ndarray) -> tuple[EncodedMAPPOState, float, bool, dict]:
        if self.world is None or self.encoded_state is None:
            raise RuntimeError("MAPPOEnvAdapter 尚未 reset")

        context = self.world.build_context()
        total_distance_before = sum(vehicle.distance_travelled for vehicle in self.world.vehicles.values())
        score_before = self.world.score
        decoded = self.action_space.decode(context, self.encoded_state, actions)

        self.world.step_with_plans(decoded.plans)
        if not self.world.is_done():
            self.world.advance_to_next_decision()

        next_state = self._observe()
        total_distance_after = sum(vehicle.distance_travelled for vehicle in self.world.vehicles.values())
        score_after = self.world.score
        pending_after = sum(
            1
            for task in self.world.tasks.values()
            if task.release_time <= max(0, self.world.current_tick - 1)
            and task.status.value == "pending"
        )
        vehicle_count = max(1, len(self.world.vehicles))

        reward = (
            (score_after - score_before)
            - self.mappo_config.invalid_action_penalty * decoded.invalid_count
            - self.mappo_config.duplicate_task_penalty * decoded.duplicate_task_count
            - self.mappo_config.pending_pressure_penalty * (pending_after / vehicle_count)
            - self.mappo_config.extra_distance_penalty * (total_distance_after - total_distance_before)
        )

        info = {
            "score": score_after,
            "invalid_action_count": decoded.invalid_count,
            "duplicate_task_count": decoded.duplicate_task_count,
            "pending_task_count": pending_after,
            "simulation_failed": self.world.simulation_failed,
        }

        self.encoded_state = next_state
        return next_state, float(reward), self.world.is_done(), info

    def episode_result(self):
        if self.world is None:
            raise RuntimeError("MAPPOEnvAdapter 尚未 reset")
        return self.world.episode_result()

    def _observe(self) -> EncodedMAPPOState:
        if self.world is None or self.world.is_done():
            return self.feature_encoder.empty_state()

        context = self.world.build_context()
        encoded = self.feature_encoder.encode(context)
        action_mask = self.action_space.build_action_mask(context, encoded)
        return encoded.with_action_mask(action_mask)

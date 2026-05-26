from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..models import Task, TaskStatus, Vehicle, VehicleState
from ..strategies.base import StrategyContext
from .config import MAPPOConfig


@dataclass
class EncodedMAPPOState:
    """MAPPO 编码后的固定维度状态。"""

    actor_obs: np.ndarray
    critic_obs: np.ndarray
    action_mask: np.ndarray
    agent_mask: np.ndarray
    vehicle_ids: List[Optional[int]]
    candidate_task_ids: List[List[Optional[int]]]

    def with_action_mask(self, action_mask: np.ndarray) -> "EncodedMAPPOState":
        return EncodedMAPPOState(
            actor_obs=self.actor_obs,
            critic_obs=self.critic_obs,
            action_mask=action_mask,
            agent_mask=self.agent_mask,
            vehicle_ids=self.vehicle_ids,
            candidate_task_ids=self.candidate_task_ids,
        )


class FeatureEncoder:
    """把 StrategyContext 编码为 Actor/Critic 可消费的定长向量。"""

    def __init__(self, config: MAPPOConfig) -> None:
        self.config = config

    def empty_state(self) -> EncodedMAPPOState:
        return EncodedMAPPOState(
            actor_obs=np.zeros((self.config.max_agents, self.config.actor_obs_dim), dtype=np.float32),
            critic_obs=np.zeros((self.config.critic_obs_dim,), dtype=np.float32),
            action_mask=np.zeros((self.config.max_agents, self.config.action_dim), dtype=np.float32),
            agent_mask=np.zeros((self.config.max_agents,), dtype=bool),
            vehicle_ids=[None] * self.config.max_agents,
            candidate_task_ids=[
                [None] * self.config.top_k_tasks
                for _ in range(self.config.max_agents)
            ],
        )

    def encode(self, context: StrategyContext) -> EncodedMAPPOState:
        vehicles = sorted(context.vehicles.values(), key=lambda item: item.vehicle_id)
        if len(vehicles) > self.config.max_agents:
            raise ValueError(
                f"MAPPO max_agents={self.config.max_agents} 小于当前车辆数 {len(vehicles)}"
            )

        actor_obs = np.zeros((self.config.max_agents, self.config.actor_obs_dim), dtype=np.float32)
        agent_mask = np.zeros((self.config.max_agents,), dtype=bool)
        vehicle_ids: List[Optional[int]] = [None] * self.config.max_agents
        candidate_task_ids: List[List[Optional[int]]] = [
            [None] * self.config.top_k_tasks
            for _ in range(self.config.max_agents)
        ]

        global_summary = self._global_summary(context)
        for slot, vehicle in enumerate(vehicles):
            agent_mask[slot] = True
            vehicle_ids[slot] = vehicle.vehicle_id
            candidate_tasks = self._rank_tasks_for_vehicle(vehicle, context)
            candidate_task_ids[slot] = [
                task.task_id if task is not None else None
                for task in candidate_tasks
            ]
            actor_obs[slot] = self._build_actor_obs(
                vehicle=vehicle,
                tasks=candidate_tasks,
                context=context,
                global_summary=global_summary,
            )

        critic_obs = self._build_critic_obs(context)
        action_mask = np.zeros((self.config.max_agents, self.config.action_dim), dtype=np.float32)
        return EncodedMAPPOState(
            actor_obs=actor_obs,
            critic_obs=critic_obs,
            action_mask=action_mask,
            agent_mask=agent_mask,
            vehicle_ids=vehicle_ids,
            candidate_task_ids=candidate_task_ids,
        )

    def _build_actor_obs(
        self,
        *,
        vehicle: Vehicle,
        tasks: List[Optional[Task]],
        context: StrategyContext,
        global_summary: np.ndarray,
    ) -> np.ndarray:
        pieces: List[float] = []
        node_denom = max(1.0, float(len(context.graph.nodes) - 1))
        load_ratio = vehicle.carried_weight / max(1.0, vehicle.load_capacity)
        pieces.extend(
            [
                vehicle.current_node / node_denom,
                self._battery_ratio(vehicle),
                load_ratio,
                1.0 if vehicle.state == VehicleState.IDLE else 0.0,
                1.0 if vehicle.state == VehicleState.MOVING else 0.0,
                1.0 if vehicle.state == VehicleState.CHARGING else 0.0,
                1.0 if vehicle.state == VehicleState.WAITING_CHARGE else 0.0,
                1.0 if vehicle.current_node == context.depot_node else 0.0,
            ]
        )

        distance_scale = self._distance_scale(context)
        max_deadline_span = max(1.0, float(context.config.deadline_range[1]))
        for task in tasks:
            if task is None:
                pieces.extend([0.0] * 6)
                continue

            travel_distance = (
                context.oracle.shortest_distance(vehicle.current_node, context.depot_node)
                + context.oracle.shortest_distance(context.depot_node, task.destination_node)
            )
            pieces.extend(
                [
                    1.0,
                    min(1.0, travel_distance / distance_scale),
                    min(
                        1.0,
                        context.oracle.shortest_distance(context.depot_node, task.destination_node) / distance_scale,
                    ),
                    min(1.0, task.weight / max(1.0, vehicle.load_capacity)),
                    float(np.clip((task.deadline - context.tick) / max_deadline_span, -1.0, 1.0)),
                    1.0 if task.overdue_penalized else 0.0,
                ]
            )

        stations = self._rank_stations_for_vehicle(vehicle, context)
        for station_id in stations:
            if station_id is None:
                pieces.extend([0.0] * 4)
                continue

            station = context.stations[station_id]
            distance = context.oracle.shortest_distance(vehicle.current_node, station.node_id)
            reachable = 1.0 if distance * vehicle.energy_per_distance <= vehicle.battery + 1e-6 else 0.0
            pieces.extend(
                [
                    1.0,
                    min(1.0, distance / distance_scale),
                    min(1.0, station.pressure_index() / 3.0),
                    reachable,
                ]
            )

        pieces.extend(global_summary.tolist())
        return np.asarray(pieces, dtype=np.float32)

    def _build_critic_obs(self, context: StrategyContext) -> np.ndarray:
        pieces: List[float] = self._critic_summary(context)
        node_denom = max(1.0, float(len(context.graph.nodes) - 1))

        vehicles = sorted(context.vehicles.values(), key=lambda item: item.vehicle_id)
        for slot in range(self.config.max_agents):
            if slot >= len(vehicles):
                pieces.extend([0.0] * 5)
                continue

            vehicle = vehicles[slot]
            state_code = {
                VehicleState.IDLE: 0,
                VehicleState.MOVING: 1,
                VehicleState.LOADING: 2,
                VehicleState.UNLOADING: 3,
                VehicleState.WAITING_CHARGE: 4,
                VehicleState.CHARGING: 5,
            }[vehicle.state]
            pieces.extend(
                [
                    self._battery_ratio(vehicle),
                    vehicle.carried_weight / max(1.0, vehicle.load_capacity),
                    state_code / 5.0,
                    vehicle.current_node / node_denom,
                    min(1.0, len(vehicle.planned_task_ids) / max(1, self.config.top_k_tasks)),
                ]
            )

        pending_tasks = [
            task
            for task in context.tasks.values()
            if task.status == TaskStatus.PENDING and task.release_time <= context.tick
        ]
        pending_tasks.sort(
            key=lambda task: (
                task.deadline - context.tick,
                context.oracle.shortest_distance(context.depot_node, task.destination_node),
                task.task_id,
            )
        )
        distance_scale = self._distance_scale(context)
        max_deadline_span = max(1.0, float(context.config.deadline_range[1]))
        for slot in range(self.config.max_global_tasks):
            if slot >= len(pending_tasks):
                pieces.extend([0.0] * 4)
                continue

            task = pending_tasks[slot]
            pieces.extend(
                [
                    float(np.clip((task.deadline - context.tick) / max_deadline_span, -1.0, 1.0)),
                    min(1.0, task.weight / max(1.0, context.config.task_weight_range[1])),
                    1.0 if task.overdue_penalized else 0.0,
                    min(
                        1.0,
                        context.oracle.shortest_distance(context.depot_node, task.destination_node) / distance_scale,
                    ),
                ]
            )

        stations = sorted(
            context.stations.values(),
            key=lambda station: (-station.pressure_index(), station.station_id),
        )
        for slot in range(self.config.max_global_stations):
            if slot >= len(stations):
                pieces.extend([0.0] * 3)
                continue
            station = stations[slot]
            pieces.extend(
                [
                    min(1.0, station.pressure_index() / 3.0),
                    min(1.0, station.charge_rate / 5.0),
                    station.node_id / node_denom,
                ]
            )

        return np.asarray(pieces, dtype=np.float32)

    def _rank_tasks_for_vehicle(
        self,
        vehicle: Vehicle,
        context: StrategyContext,
    ) -> List[Optional[Task]]:
        pending_tasks = [
            task
            for task in context.tasks.values()
            if task.status == TaskStatus.PENDING and task.release_time <= context.tick
        ]
        pending_tasks.sort(
            key=lambda task: (
                task.deadline - context.tick,
                context.oracle.shortest_distance(vehicle.current_node, context.depot_node)
                + context.oracle.shortest_distance(context.depot_node, task.destination_node),
                -task.weight,
                task.task_id,
            )
        )
        ranked = pending_tasks[: self.config.top_k_tasks]
        while len(ranked) < self.config.top_k_tasks:
            ranked.append(None)
        return ranked

    def _rank_stations_for_vehicle(
        self,
        vehicle: Vehicle,
        context: StrategyContext,
    ) -> List[Optional[int]]:
        stations = sorted(
            context.stations.values(),
            key=lambda station: (
                context.oracle.shortest_distance(vehicle.current_node, station.node_id),
                station.pressure_index(),
                station.station_id,
            )
        )
        ranked = [station.station_id for station in stations[: self.config.top_k_stations]]
        while len(ranked) < self.config.top_k_stations:
            ranked.append(None)
        return ranked

    def _global_summary(self, context: StrategyContext) -> np.ndarray:
        vehicles = list(context.vehicles.values())
        tasks = list(context.tasks.values())
        released_tasks = [task for task in tasks if task.release_time <= context.tick]
        released_total = max(1, len(released_tasks))
        tick_scale = max(
            context.config.deadline_range[1] * 4.0,
            float(max((task.deadline for task in tasks), default=context.config.deadline_range[1])),
        )

        pending = sum(1 for task in released_tasks if task.status == TaskStatus.PENDING)
        in_progress = sum(1 for task in released_tasks if task.status == TaskStatus.IN_PROGRESS)
        completed = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        overdue = sum(1 for task in tasks if task.overdue_penalized)
        avg_battery = sum(self._battery_ratio(vehicle) for vehicle in vehicles) / max(1, len(vehicles))
        idle_ratio = sum(1 for vehicle in vehicles if vehicle.state == VehicleState.IDLE) / max(1, len(vehicles))

        return np.asarray(
            [
                min(1.0, context.tick / max(1.0, tick_scale)),
                pending / released_total,
                in_progress / released_total,
                completed / max(1, len(tasks)),
                overdue / max(1, len(tasks)),
                idle_ratio if len(vehicles) else avg_battery,
            ],
            dtype=np.float32,
        )

    def _critic_summary(self, context: StrategyContext) -> List[float]:
        vehicles = list(context.vehicles.values())
        tasks = list(context.tasks.values())
        released_tasks = [task for task in tasks if task.release_time <= context.tick]
        released_total = max(1, len(released_tasks))
        tick_scale = max(
            context.config.deadline_range[1] * 4.0,
            float(max((task.deadline for task in tasks), default=context.config.deadline_range[1])),
        )

        pending = sum(1 for task in released_tasks if task.status == TaskStatus.PENDING)
        assigned = sum(1 for task in released_tasks if task.status == TaskStatus.ASSIGNED)
        in_progress = sum(1 for task in released_tasks if task.status == TaskStatus.IN_PROGRESS)
        completed = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        overdue = sum(1 for task in tasks if task.overdue_penalized)
        avg_battery = sum(self._battery_ratio(vehicle) for vehicle in vehicles) / max(1, len(vehicles))
        avg_station_pressure = (
            sum(station.pressure_index() for station in context.stations.values()) / max(1, len(context.stations))
        )

        return [
            min(1.0, context.tick / max(1.0, tick_scale)),
            pending / released_total,
            assigned / released_total,
            in_progress / released_total,
            completed / max(1, len(tasks)),
            overdue / max(1, len(tasks)),
            avg_battery,
            min(1.0, avg_station_pressure / 3.0),
        ]

    @staticmethod
    def _battery_ratio(vehicle: Vehicle) -> float:
        return vehicle.battery / max(1.0, vehicle.battery_capacity)

    @staticmethod
    def _distance_scale(context: StrategyContext) -> float:
        return max(10.0, context.config.max_edge_distance * max(1, len(context.graph.nodes)))

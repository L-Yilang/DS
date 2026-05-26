from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..models import ChargingStation, Task, TaskStatus, Vehicle, VehicleState
from ..strategies.base import StrategyContext, VehiclePlan
from .config import MAPPOConfig
from .features import EncodedMAPPOState


ACTION_KEEP = 0
ACTION_RETURN_DEPOT = 1
ACTION_CHARGE_BEST = 2


@dataclass
class DecodeResult:
    plans: Dict[int, VehiclePlan]
    invalid_count: int = 0
    duplicate_task_count: int = 0


class MacroActionSpace:
    """MAPPO 宏动作空间与 VehiclePlan 解码器。"""

    def __init__(self, config: MAPPOConfig) -> None:
        self.config = config
        self._plan_builder = _MacroPlanBuilder()

    def build_action_mask(
        self,
        context: StrategyContext,
        encoded: EncodedMAPPOState,
    ) -> np.ndarray:
        mask = np.zeros((self.config.max_agents, self.config.action_dim), dtype=np.float32)

        for agent_index, vehicle_id in enumerate(encoded.vehicle_ids):
            if vehicle_id is None:
                continue

            vehicle = context.vehicles[vehicle_id]
            mask[agent_index, ACTION_KEEP] = 1.0

            if vehicle.state not in (VehicleState.IDLE, VehicleState.CHARGING, VehicleState.WAITING_CHARGE):
                continue

            if vehicle.state == VehicleState.IDLE and vehicle.current_node != context.depot_node:
                mask[agent_index, ACTION_RETURN_DEPOT] = 1.0

            if vehicle.state == VehicleState.IDLE:
                if self._plan_builder.build_charge_plan(vehicle, context) is not None:
                    mask[agent_index, ACTION_CHARGE_BEST] = 1.0

            for task_slot, task_id in enumerate(encoded.candidate_task_ids[agent_index]):
                action_index = 3 + task_slot
                if task_id is None:
                    continue
                task = context.tasks.get(task_id)
                if task is None:
                    continue
                if self._plan_builder.can_dispatch_single_task(vehicle, task, context):
                    mask[agent_index, action_index] = 1.0

        return mask

    def decode(
        self,
        context: StrategyContext,
        encoded: EncodedMAPPOState,
        actions: np.ndarray,
    ) -> DecodeResult:
        plans: Dict[int, VehiclePlan] = {}
        invalid_count = 0
        duplicate_task_count = 0
        reserved_task_ids: set[int] = set()

        for agent_index, vehicle_id in enumerate(encoded.vehicle_ids):
            if vehicle_id is None:
                continue

            vehicle = context.vehicles[vehicle_id]
            action = int(actions[agent_index])
            if action < 0 or action >= self.config.action_dim:
                invalid_count += 1
                plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                continue

            if encoded.action_mask[agent_index, action] <= 0.0:
                invalid_count += 1
                plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                continue

            if action == ACTION_KEEP:
                plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                continue

            if action == ACTION_RETURN_DEPOT:
                plans[vehicle_id] = self._plan_builder.build_return_plan(vehicle, context)
                continue

            if action == ACTION_CHARGE_BEST:
                charge_plan = self._plan_builder.build_charge_plan(vehicle, context)
                if charge_plan is None:
                    invalid_count += 1
                    plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                else:
                    plans[vehicle_id] = charge_plan
                continue

            task_slot = action - 3
            task_id = encoded.candidate_task_ids[agent_index][task_slot]
            if task_id is None:
                invalid_count += 1
                plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                continue

            if task_id in reserved_task_ids:
                duplicate_task_count += 1
                plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                continue

            task = context.tasks.get(task_id)
            if task is None or not self._plan_builder.can_dispatch_single_task(vehicle, task, context):
                invalid_count += 1
                plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
                continue

            plans[vehicle_id] = self._plan_builder.build_single_task_plan(vehicle, task, context)
            reserved_task_ids.add(task_id)

        return DecodeResult(
            plans=plans,
            invalid_count=invalid_count,
            duplicate_task_count=duplicate_task_count,
        )


class _MacroPlanBuilder:
    """只负责把宏动作翻译为安全的单车 VehiclePlan。"""

    def __init__(self) -> None:
        self.reserve_energy = 2.0

    def can_dispatch_single_task(
        self,
        vehicle: Vehicle,
        task: Task,
        context: StrategyContext,
    ) -> bool:
        if task.status != TaskStatus.PENDING or task.release_time > context.tick:
            return False
        if task.weight > vehicle.load_capacity + 1e-6:
            return False

        path = [context.depot_node, task.destination_node, context.depot_node]
        actions = ["load", "unload", "keep"]
        if not self._plan_energy_safe(vehicle, path, actions, context):
            return False
        if not self._can_interrupt_charge_for_plan(vehicle, path, actions, context):
            return False
        return True

    def build_single_task_plan(
        self,
        vehicle: Vehicle,
        task: Task,
        context: StrategyContext,
    ) -> VehiclePlan:
        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[task.task_id],
            action=["load", "unload", "keep"],
            planned_path=[context.depot_node, task.destination_node, context.depot_node],
        )

    def build_return_plan(
        self,
        vehicle: Vehicle,
        context: StrategyContext,
    ) -> VehiclePlan:
        if vehicle.current_node == context.depot_node:
            return VehiclePlan(vehicle_id=vehicle.vehicle_id)
        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[],
            action=["keep"],
            planned_path=[context.depot_node],
        )

    def build_charge_plan(
        self,
        vehicle: Vehicle,
        context: StrategyContext,
    ) -> Optional[VehiclePlan]:
        station = self._select_best_station(
            from_node=vehicle.current_node,
            available_energy=vehicle.battery,
            energy_per_distance=vehicle.energy_per_distance,
            context=context,
        )
        if station is None:
            return None
        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[],
            action=["charge"],
            planned_path=[station.node_id],
        )

    def _select_best_station(
        self,
        from_node: int,
        available_energy: float,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> Optional[ChargingStation]:
        best_station: Optional[ChargingStation] = None
        best_score = float("inf")
        for station in context.stations.values():
            distance = context.oracle.shortest_distance(from_node, station.node_id)
            required = distance * energy_per_distance + self.reserve_energy
            if required > available_energy + 1e-6:
                continue

            queue_cost = station.pressure_index() * context.config.queue_penalty_factor
            score = distance + queue_cost
            if score < best_score:
                best_score = score
                best_station = station
        return best_station

    def _plan_energy_safe(
        self,
        vehicle: Vehicle,
        key_nodes: list[int],
        actions: list[str],
        context: StrategyContext,
    ) -> bool:
        if len(key_nodes) != len(actions):
            return False

        battery = vehicle.battery
        cursor = vehicle.current_node
        for destination, action in zip(key_nodes, actions):
            ok, battery = self._can_traverse_segment_safely(
                start_node=cursor,
                target_node=destination,
                battery=battery,
                energy_per_distance=vehicle.energy_per_distance,
                context=context,
            )
            if not ok:
                return False
            cursor = destination
            if action == "charge":
                battery = vehicle.battery_capacity
        return True

    def _can_traverse_segment_safely(
        self,
        *,
        start_node: int,
        target_node: int,
        battery: float,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> tuple[bool, float]:
        if start_node == target_node:
            return True, battery

        path = context.oracle.shortest_path(start_node, target_node)
        if len(path) < 2:
            return False, battery

        current_battery = battery
        for index in range(1, len(path)):
            a = path[index - 1]
            b = path[index]
            edge_distance = context.graph.edge_distance(a, b)
            energy_for_edge = edge_distance * energy_per_distance
            if current_battery + 1e-6 < energy_for_edge:
                return False, battery

            energy_to_depot = context.oracle.shortest_distance(b, context.depot_node) * energy_per_distance
            energy_to_station = self._nearest_station_energy(b, energy_per_distance, context)
            required = energy_for_edge + min(energy_to_depot, energy_to_station) + self.reserve_energy
            if current_battery + 1e-6 < required:
                return False, battery

            current_battery -= energy_for_edge

        return True, current_battery

    def _nearest_station_energy(
        self,
        node_id: int,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> float:
        return min(
            context.oracle.shortest_distance(node_id, station.node_id) * energy_per_distance
            for station in context.stations.values()
        )

    def _can_interrupt_charge_for_plan(
        self,
        vehicle: Vehicle,
        planned_path: list[int],
        action: list[str],
        context: StrategyContext,
    ) -> bool:
        if vehicle.state not in (VehicleState.CHARGING, VehicleState.WAITING_CHARGE):
            return True

        battery = vehicle.battery
        cursor = vehicle.current_node
        for destination, current_action in zip(planned_path, action):
            if cursor == destination:
                if current_action == "charge":
                    battery = vehicle.battery_capacity
                continue

            path = context.oracle.shortest_path(cursor, destination)
            if len(path) < 2:
                return False

            next_node = path[1]
            edge_distance = context.graph.edge_distance(cursor, next_node)
            energy_for_edge = edge_distance * vehicle.energy_per_distance
            if battery + 1e-6 < energy_for_edge:
                return False

            energy_to_depot = context.oracle.shortest_distance(next_node, context.depot_node) * vehicle.energy_per_distance
            energy_to_station = min(
                context.oracle.shortest_distance(next_node, station.node_id) * vehicle.energy_per_distance
                for station in context.stations.values()
            )
            required = energy_for_edge + min(energy_to_depot, energy_to_station) + self.reserve_energy
            return battery + 1e-6 >= required

        return True

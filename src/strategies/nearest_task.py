from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..models import ChargingStation, Task, TaskStatus, Vehicle, VehicleState
from .base import SchedulingStrategy, StrategyContext, VehiclePlan


class NearestTaskStrategy(SchedulingStrategy):
    """最近任务优先策略（子类负责完整编排：匹配 + 充电 + 任务链）。"""

    name = "nearest_task"

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        plans: Dict[int, VehiclePlan] = {}

        available_tasks: Dict[int, Task] = {
            task.task_id: task
            for task in context.tasks.values()
            if task.status == TaskStatus.PENDING and task.release_time <= context.tick
        }

        idle_vehicle_ids = [
            vehicle_id
            for vehicle_id, vehicle in sorted(context.vehicles.items())
            if vehicle.state in (
                VehicleState.IDLE,
                VehicleState.CHARGING,
                VehicleState.WAITING_CHARGE,
            )
        ]

        # 1) 低电量车辆优先补能。
        for vehicle_id in idle_vehicle_ids:
            if vehicle_id in plans:
                continue

            vehicle = context.vehicles[vehicle_id]
            if self._battery_ratio(vehicle) >= context.config.low_battery_ratio:
                continue

            charge_plan = self._build_charge_plan(vehicle, context)
            if charge_plan is not None:
                plans[vehicle_id] = charge_plan

        # 2) 全局任务-车辆匹配（贪心）。
        candidate_vehicle_ids = [
            vehicle_id
            for vehicle_id in idle_vehicle_ids
            if vehicle_id not in plans
        ]

        assignments = self._match_tasks_to_vehicles(
            candidate_vehicle_ids=candidate_vehicle_ids,
            available_tasks=available_tasks,
            context=context,
        )

        for vehicle_id, task in assignments.items():
            vehicle = context.vehicles[vehicle_id]
            plans[vehicle_id] = self._build_single_task_chain(vehicle, task, context)
            available_tasks.pop(task.task_id, None)

        # 3) 其余空闲车辆兜底（尽量充电，否则回仓或保持）。
        for vehicle_id in idle_vehicle_ids:
            if vehicle_id in plans:
                continue
            vehicle = context.vehicles[vehicle_id]
            plans[vehicle_id] = self._build_idle_fallback_plan(vehicle, context)

        return plans

    def _match_tasks_to_vehicles(
        self,
        candidate_vehicle_ids: Sequence[int],
        available_tasks: Dict[int, Task],
        context: StrategyContext,
    ) -> Dict[int, Task]:
        """构建全部可行 pair，按策略优先级做全局贪心匹配。"""

        pair_candidates: List[Tuple[Tuple[float, ...], int, int]] = []

        for vehicle_id in candidate_vehicle_ids:
            vehicle = context.vehicles[vehicle_id]
            to_depot = context.oracle.shortest_distance(vehicle.current_node, context.depot_node)

            for task in available_tasks.values():
                if task.weight > vehicle.load_capacity:
                    continue

                required = self._estimate_required_energy(vehicle, task, context)
                if vehicle.battery + 1e-6 < required:
                    continue

                plan = self._build_single_task_chain(vehicle, task, context)
                if not self._can_interrupt_charge_for_plan(
                    vehicle,
                    plan.planned_path,
                    plan.action,
                    context,
                ):
                    continue

                to_destination = context.oracle.shortest_distance(context.depot_node, task.destination_node)
                priority = (to_depot + to_destination, task.deadline, -task.weight)
                pair_candidates.append((priority, vehicle_id, task.task_id))

        pair_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        assignments: Dict[int, Task] = {}
        used_vehicle_ids: set[int] = set()
        used_task_ids: set[int] = set()

        for _, vehicle_id, task_id in pair_candidates:
            if vehicle_id in used_vehicle_ids or task_id in used_task_ids:
                continue
            assignments[vehicle_id] = available_tasks[task_id]
            used_vehicle_ids.add(vehicle_id)
            used_task_ids.add(task_id)

        return assignments

    def _build_single_task_chain(
        self,
        vehicle: Vehicle,
        task: Task,
        context: StrategyContext,
    ) -> VehiclePlan:
        """单任务链：仓库装货 -> 任务点卸货，并按需追加充电。"""

        planned_path = [context.depot_node, task.destination_node]
        action = ["load", "unload"]

        to_depot = context.oracle.shortest_distance(vehicle.current_node, context.depot_node)
        to_task = context.oracle.shortest_distance(context.depot_node, task.destination_node)
        used = (to_depot + to_task) * vehicle.energy_per_distance
        expected_after_unload = max(0.0, vehicle.battery - used)

        self._append_post_unload_charge_if_needed(
            planned_path=planned_path,
            action=action,
            from_node=task.destination_node,
            expected_battery=expected_after_unload,
            battery_capacity=vehicle.battery_capacity,
            energy_per_distance=vehicle.energy_per_distance,
            context=context,
        )

        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[task.task_id],
            action=action,
            planned_path=planned_path,
        )

    def _build_idle_fallback_plan(self, vehicle: Vehicle, context: StrategyContext) -> VehiclePlan:
        """空闲兜底：优先补能，否则回仓或保持。"""

        if self._battery_ratio(vehicle) < 0.98:
            charge_plan = self._build_charge_plan(vehicle, context)
            if charge_plan is not None:
                return charge_plan

        if vehicle.current_node != context.depot_node:
            return VehiclePlan(
                vehicle_id=vehicle.vehicle_id,
                task_id=[],
                action=["keep"],
                planned_path=[context.depot_node],
            )

        return VehiclePlan(vehicle_id=vehicle.vehicle_id)

    def _build_charge_plan(self, vehicle: Vehicle, context: StrategyContext) -> Optional[VehiclePlan]:
        """为车辆构建充电计划。"""

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

    def _append_post_unload_charge_if_needed(
        self,
        planned_path: List[int],
        action: List[str],
        from_node: int,
        expected_battery: float,
        battery_capacity: float,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> None:
        """卸货后若电量偏低，则在链尾追加充电动作。"""

        threshold = max(context.config.low_battery_ratio * 1.2, 0.35)
        if expected_battery / max(1.0, battery_capacity) >= threshold:
            return

        station = self._select_best_station(
            from_node=from_node,
            available_energy=expected_battery,
            energy_per_distance=energy_per_distance,
            context=context,
        )
        if station is None:
            return

        planned_path.append(station.node_id)
        action.append("charge")

    def _select_best_station(
        self,
        from_node: int,
        available_energy: float,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> Optional[ChargingStation]:
        """在可达站点中按“距离 + 排队压力”选择最优充电站。"""

        best_station: Optional[ChargingStation] = None
        best_score = float("inf")

        for station in context.stations.values():
            distance = context.oracle.shortest_distance(from_node, station.node_id)
            required = distance * energy_per_distance
            if required > available_energy + 1e-6:
                continue

            score = distance + context.config.queue_penalty_factor * station.pressure_index()
            if score < best_score:
                best_score = score
                best_station = station

        return best_station

    def _estimate_required_energy(self, vehicle: Vehicle, task: Task, context: StrategyContext) -> float:
        """估算单任务安全电量：当前点->仓库->任务点->仓库 + 冗余。"""

        d1 = context.oracle.shortest_distance(vehicle.current_node, context.depot_node)
        d2 = context.oracle.shortest_distance(context.depot_node, task.destination_node)
        d3 = context.oracle.shortest_distance(task.destination_node, context.depot_node)
        reserve = 2.0
        return (d1 + d2 + d3) * vehicle.energy_per_distance + reserve

    @staticmethod
    def _battery_ratio(vehicle: Vehicle) -> float:
        return vehicle.battery / max(1.0, vehicle.battery_capacity)








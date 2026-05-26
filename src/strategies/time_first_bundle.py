from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from ..models import ChargingStation, Task, TaskStatus, Vehicle, VehicleState
from .base import SchedulingStrategy, StrategyContext, VehiclePlan


class TimeFirstBundleStrategy(SchedulingStrategy):
    """时间优先 + 拼单 + 充电再平衡策略。"""

    name = "time_first_bundle"

    def __init__(self) -> None:
        self._task_wait_ticks: Dict[int, int] = {}

        # 紧急判断阈值
        self.emergency_margin_ticks = 12

        # 拼单外部充电绕路比例阈值
        self.max_detour_ratio = 0.25

        # 三方案代价函数权重
        self.w_travel = 1.0
        self.w_queue = 1.2
        self.w_energy = 0.8
        self.cost_epsilon = 0.35

        # 再平衡规则阈值
        self.keep_idle_ratio = 0.30
        self.all_low_battery_ratio = 0.30
        self.depot_charge_ratio_no_pressure = 0.90

        self.reserve_energy = 2.0

        # 拼单调度开关
        self.enable_bundle_dispatch = True

        # 单任务优先级分层：优先保护未超时任务，其次处理可救回的轻度超时任务。
        self.rescue_lateness_ticks = 10
        self.hopeless_lateness_ticks = 28
        self.bundle_wait_escape_ticks = 2

        # 紧急阈值随任务积压动态抬升，避免在中高负载下触发过晚。
        self.backlog_margin_gain = 3.0
        self.backlog_margin_cap = 16

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        plans: Dict[int, VehiclePlan] = {}

        pending_tasks = self._pending_tasks(context)
        self._refresh_wait_counters(pending_tasks)

        idle_vehicle_ids = [
            vehicle_id
            for vehicle_id, vehicle in sorted(context.vehicles.items())
            if vehicle.state in (
                VehicleState.IDLE,
                VehicleState.CHARGING,
                VehicleState.WAITING_CHARGE,
            )
            and not self._vehicle_has_unfinished_work(vehicle, context)
        ]

        # 阶段A：先处理非仓库车辆（回仓/低电充电），仓库空闲车先参与任务分配。
        depot_idle_ids: List[int] = []
        for vehicle_id in idle_vehicle_ids:
            vehicle = context.vehicles[vehicle_id]

            if vehicle.current_node != context.depot_node:
                if self._battery_ratio(vehicle) < context.config.low_battery_ratio:
                    charge_plan = self._build_charge_plan(vehicle, context)
                    if charge_plan is not None:
                        plans[vehicle_id] = charge_plan
                        continue

                plans[vehicle_id] = VehiclePlan(
                    vehicle_id=vehicle.vehicle_id,
                    action=["keep"],
                    planned_path=[context.depot_node],
                )
                continue

            depot_idle_ids.append(vehicle_id)

        if not pending_tasks:
            self._apply_idle_rebalance(
                idle_vehicle_ids=depot_idle_ids,
                plans=plans,
                context=context,
                pending_task_count=0,
                urgent_task_count=0,
            )
            return plans

        available_vehicle_ids = [
            vehicle_id
            for vehicle_id in depot_idle_ids
            if vehicle_id not in plans
        ]

        remaining_tasks: Dict[int, Task] = {task.task_id: task for task in pending_tasks}

        # 阶段B-1：先派发紧急任务（单任务）。
        urgent_task_ids = [
            task.task_id
            for task in remaining_tasks.values()
            if self._is_emergency_task(task, context)
        ]
        urgent_task_ids.sort(
            key=lambda task_id: self._single_task_sort_key(remaining_tasks[task_id], context)
        )

        for task_id in urgent_task_ids:
            if task_id not in remaining_tasks:
                continue
            if not available_vehicle_ids:
                break

            selected_vehicle_id = self._select_vehicle_for_single_task(
                task=remaining_tasks[task_id],
                candidate_vehicle_ids=available_vehicle_ids,
                context=context,
            )
            if selected_vehicle_id is None:
                continue

            vehicle = context.vehicles[selected_vehicle_id]
            plan = self._build_single_task_plan(vehicle, remaining_tasks[task_id], context)
            plans[selected_vehicle_id] = plan

            available_vehicle_ids.remove(selected_vehicle_id)
            remaining_tasks.pop(task_id, None)

        # 阶段B-2：非紧急任务进入拼单候选池，每个 tick 全量尝试拼单组合。
        if self.enable_bundle_dispatch and available_vehicle_ids:
            bundle_pool: Dict[int, Task] = {
                task_id: task
                for task_id, task in remaining_tasks.items()
                if not self._is_emergency_task(task, context)
            }
            if len(bundle_pool) >= 2:
                bundle_task_ids_before = set(bundle_pool.keys())
                self._dispatch_bundle_tasks(
                    available_vehicle_ids=available_vehicle_ids,
                    remaining_tasks=bundle_pool,
                    plans=plans,
                    context=context,
                )
                assigned_bundle_task_ids = bundle_task_ids_before - set(bundle_pool.keys())
                for assigned_task_id in assigned_bundle_task_ids:
                    remaining_tasks.pop(assigned_task_id, None)

        # 阶段B-3：非紧急任务等待拼单超阈值后，允许单任务兜底派发。
        waited_non_urgent_tasks = [
            task
            for task in remaining_tasks.values()
            if not self._is_emergency_task(task, context)
            and self._task_wait_ticks.get(task.task_id, 0) >= self.bundle_wait_escape_ticks
        ]
        waited_non_urgent_tasks.sort(key=lambda task: self._single_task_sort_key(task, context))

        for task in waited_non_urgent_tasks:
            if task.task_id not in remaining_tasks or not available_vehicle_ids:
                continue

            selected_vehicle_id = self._select_vehicle_for_single_task(
                task=task,
                candidate_vehicle_ids=available_vehicle_ids,
                context=context,
            )
            if selected_vehicle_id is None:
                continue

            vehicle = context.vehicles[selected_vehicle_id]
            plans[selected_vehicle_id] = self._build_single_task_plan(vehicle, task, context)

            available_vehicle_ids.remove(selected_vehicle_id)
            remaining_tasks.pop(task.task_id, None)

        # 阶段C：派发后的空闲车再平衡充电。
        remaining_idle_ids = [
            vehicle_id
            for vehicle_id in depot_idle_ids
            if vehicle_id not in plans
        ]
        urgent_remaining = sum(
            1
            for task in remaining_tasks.values()
            if self._is_emergency_task(task, context)
        )
        self._apply_idle_rebalance(
            idle_vehicle_ids=remaining_idle_ids,
            plans=plans,
            context=context,
            pending_task_count=len(remaining_tasks),
            urgent_task_count=urgent_remaining,
        )

        return plans

    def _dispatch_bundle_tasks(
        self,
        available_vehicle_ids: List[int],
        remaining_tasks: Dict[int, Task],
        plans: Dict[int, VehiclePlan],
        context: StrategyContext,
    ) -> None:
        while len(available_vehicle_ids) > 0 and len(remaining_tasks) >= 2:
            pair_candidates = self._build_pair_candidates(
                candidate_vehicle_ids=available_vehicle_ids,
                tasks=list(remaining_tasks.values()),
                context=context,
            )
            if not pair_candidates:
                break

            # 可行方案数少的任务对优先；其次代价低优先。
            pair_candidates.sort(
                key=lambda item: (
                    item["mode_count"],
                    item["cost"],
                    item["slack_key"],
                    item["vehicle_id"],
                    item["task_ids"],
                )
            )

            chosen = pair_candidates[0]
            vehicle_id = chosen["vehicle_id"]
            task_ids = chosen["task_ids"]

            if vehicle_id not in available_vehicle_ids:
                break
            if any(task_id not in remaining_tasks for task_id in task_ids):
                # 动态重算期间任务已被其它分配占用。
                continue

            vehicle = context.vehicles[vehicle_id]
            plan = self._build_bundle_plan(
                vehicle=vehicle,
                first_task=remaining_tasks[task_ids[0]],
                second_task=remaining_tasks[task_ids[1]],
                mode=chosen["mode"],
                station_id=chosen.get("station_id"),
                context=context,
            )

            plans[vehicle_id] = plan
            available_vehicle_ids.remove(vehicle_id)
            for task_id in task_ids:
                remaining_tasks.pop(task_id, None)



    def _build_pair_candidates(
        self,
        candidate_vehicle_ids: Sequence[int],
        tasks: List[Task],
        context: StrategyContext,
    ) -> List[dict]:
        candidates: List[dict] = []

        for vehicle_id in candidate_vehicle_ids:
            vehicle = context.vehicles[vehicle_id]
            for task_a, task_b in combinations(tasks, 2):
                if not self._pair_weight_feasible(vehicle, task_a, task_b):
                    continue

                mode_infos = self._feasible_modes_for_pair(vehicle, task_a, task_b, context)
                if not mode_infos:
                    continue

                mode_infos.sort(key=lambda info: (info["cost"], info["mode_rank"]))
                best = mode_infos[0]

                slack_key = min(
                    self._pair_task_slack(task_a, context),
                    self._pair_task_slack(task_b, context),
                )

                task_ids = (task_a.task_id, task_b.task_id)
                plan = self._build_bundle_plan(
                    vehicle=vehicle,
                    first_task=task_a,
                    second_task=task_b,
                    mode=best["mode"],
                    station_id=best.get("station_id"),
                    context=context,
                )
                if not self._can_interrupt_charge_for_plan(
                    vehicle,
                    plan.planned_path,
                    plan.action,
                    context,
                ):
                    continue
                candidates.append(
                    {
                        "vehicle_id": vehicle_id,
                        "task_ids": task_ids,
                        "mode": best["mode"],
                        "station_id": best.get("station_id"),
                        "cost": best["cost"],
                        "mode_count": len(mode_infos),
                        "slack_key": slack_key,
                    }
                )

        return candidates

    def _feasible_modes_for_pair(
        self,
        vehicle: Vehicle,
        first_task: Task,
        second_task: Task,
        context: StrategyContext,
    ) -> List[dict]:
        depot = context.depot_node

        d_depot_1 = context.oracle.shortest_distance(depot, first_task.destination_node)
        d_depot_2 = context.oracle.shortest_distance(depot, second_task.destination_node)
        d_1_2 = context.oracle.shortest_distance(
            first_task.destination_node,
            second_task.destination_node,
        )
        d_2_depot = context.oracle.shortest_distance(second_task.destination_node, depot)

        # 方案1：仓库连续装两单，再串行配送后回仓。
        mode1_path = [
            depot,
            depot,
            first_task.destination_node,
            second_task.destination_node,
            depot,
        ]
        mode1_actions = ["load", "load", "unload", "unload", "keep"]
        mode1_distance = self._key_path_distance(vehicle.current_node, mode1_path, context)

        modes: List[dict] = []
        if self._plan_energy_safe(vehicle, mode1_path, mode1_actions, context):
            mode1_energy = mode1_distance * vehicle.energy_per_distance + self.reserve_energy
            energy_risk = max(0.0, mode1_energy - vehicle.battery) / max(1.0, vehicle.battery_capacity)
            modes.append(
                {
                    "mode": "mode1",
                    "cost": self.w_energy * energy_risk,
                    "mode_rank": 1,
                    "station_id": None,
                }
            )

        # 方案2：双载出发，第一单卸完后补电，再送第二单。
        station_after_first = self._nearest_reachable_station(
            from_node=first_task.destination_node,
            available_energy=vehicle.battery_capacity,
            energy_per_distance=vehicle.energy_per_distance,
            context=context,
        )
        if station_after_first is not None:
            d_1_s = context.oracle.shortest_distance(first_task.destination_node, station_after_first.node_id)
            d_s_2 = context.oracle.shortest_distance(station_after_first.node_id, second_task.destination_node)
            mode2_distance = d_depot_1 + d_1_s + d_s_2 + d_2_depot
            detour_ratio = max(0.0, mode2_distance - mode1_distance) / max(1.0, mode1_distance)

            mode2_path = [
                depot,
                depot,
                first_task.destination_node,
                station_after_first.node_id,
                second_task.destination_node,
                depot,
            ]
            mode2_actions = ["load", "load", "unload", "charge", "unload", "keep"]

            if detour_ratio <= self.max_detour_ratio and self._plan_energy_safe(
                vehicle,
                mode2_path,
                mode2_actions,
                context,
            ):
                extra_travel = max(0.0, mode2_distance - mode1_distance)
                queue_wait = station_after_first.pressure_index() * 5.0
                required_to_station = (d_depot_1 + d_1_s) * vehicle.energy_per_distance + self.reserve_energy
                energy_risk = max(0.0, required_to_station - vehicle.battery) / max(
                    1.0, vehicle.battery_capacity
                )
                modes.append(
                    {
                        "mode": "mode2",
                        "cost": (
                            self.w_travel * extra_travel
                            + self.w_queue * queue_wait
                            + self.w_energy * energy_risk
                        ),
                        "mode_rank": 2,
                        "station_id": station_after_first.station_id,
                    }
                )

        # 方案3：双载串行送完后补电，再回仓。
        station_after_second = self._nearest_station(
            from_node=second_task.destination_node,
            context=context,
        )
        if station_after_second is not None:
            d_2_s = context.oracle.shortest_distance(second_task.destination_node, station_after_second.node_id)
            d_s_depot_after = context.oracle.shortest_distance(station_after_second.node_id, depot)
            mode3_distance = d_depot_1 + d_1_2 + d_2_s + d_s_depot_after
            detour_ratio = max(0.0, mode3_distance - mode1_distance) / max(1.0, mode1_distance)

            mode3_path = [
                depot,
                depot,
                first_task.destination_node,
                second_task.destination_node,
                station_after_second.node_id,
                depot,
            ]
            mode3_actions = ["load", "load", "unload", "unload", "charge", "keep"]

            if detour_ratio <= self.max_detour_ratio and self._plan_energy_safe(
                vehicle,
                mode3_path,
                mode3_actions,
                context,
            ):
                extra_travel = max(0.0, mode3_distance - mode1_distance)
                queue_wait = station_after_second.pressure_index() * 5.0
                required_to_station = (
                    d_depot_1 + d_1_2 + d_2_s
                ) * vehicle.energy_per_distance + self.reserve_energy
                energy_risk = max(0.0, required_to_station - vehicle.battery) / max(
                    1.0, vehicle.battery_capacity
                )
                modes.append(
                    {
                        "mode": "mode3",
                        "cost": (
                            self.w_travel * extra_travel
                            + self.w_queue * queue_wait
                            + self.w_energy * energy_risk
                        ),
                        "mode_rank": 3,
                        "station_id": station_after_second.station_id,
                    }
                )

        if len(modes) >= 2:
            modes.sort(key=lambda info: (info["cost"], info["mode_rank"]))
            if abs(modes[0]["cost"] - modes[1]["cost"]) <= self.cost_epsilon:
                modes.sort(key=lambda info: (info["mode_rank"], info["cost"]))

        return modes

    def _build_bundle_plan(
        self,
        vehicle: Vehicle,
        first_task: Task,
        second_task: Task,
        mode: str,
        station_id: Optional[int],
        context: StrategyContext,
    ) -> VehiclePlan:
        depot = context.depot_node
        station = context.stations.get(station_id) if station_id is not None else None
        
        # 根据 mode 动态构造路径和动作
        if mode == "mode1":
            path = [depot, depot, first_task.destination_node, second_task.destination_node, depot]
            actions = ["load", "load", "unload", "unload", "keep"]
        elif mode == "mode2" and station is not None:
            path = [depot, depot, first_task.destination_node, station.node_id, second_task.destination_node, depot]
            actions = ["load", "load", "unload", "charge", "unload", "keep"]
        elif mode == "mode3" and station is not None:
            path = [depot, depot, first_task.destination_node, second_task.destination_node, station.node_id, depot]
            actions = ["load", "load", "unload", "unload", "charge", "keep"]
        else:
            # 降级为基础方案（如 station 为 None）
            path = [depot, depot, first_task.destination_node, second_task.destination_node, depot]
            actions = ["load", "load", "unload", "unload", "keep"]

        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[first_task.task_id, second_task.task_id],
            action=actions,
            planned_path=path,
        )

    def _build_single_task_plan(
        self,
        vehicle: Vehicle,
        task: Task,
        context: StrategyContext,
    ) -> VehiclePlan:
        depot = context.depot_node
        path = [depot, task.destination_node, depot]
        actions = ["load", "unload", "keep"]
        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[task.task_id],
            action=actions,
            planned_path=path,
        )

    def _select_vehicle_for_single_task(
        self,
        task: Task,
        candidate_vehicle_ids: Sequence[int],
        context: StrategyContext,
    ) -> Optional[int]:
        options: List[Tuple[Tuple[float, float, int], int]] = []

        for vehicle_id in candidate_vehicle_ids:
            vehicle = context.vehicles[vehicle_id]
            if task.weight > vehicle.load_capacity + 1e-6:
                continue

            required_energy = self._required_energy_for_single_task(vehicle, task, context)
            single_path = [context.depot_node, task.destination_node, context.depot_node]
            single_actions = ["load", "unload", "keep"]
            if not self._plan_energy_safe(vehicle, single_path, single_actions, context):
                continue
            if not self._can_interrupt_charge_for_plan(
                vehicle,
                single_path,
                single_actions,
                context,
            ):
                continue

            sort_key = (
                self._direct_eta(task, context),
                required_energy,
                vehicle_id,
            )
            options.append((sort_key, vehicle_id))

        if not options:
            return None

        options.sort(key=lambda item: item[0])
        return options[0][1]

    def _single_task_sort_key(self, task: Task, context: StrategyContext) -> Tuple[int, int, float, int]:
        slack = self._pair_task_slack(task, context)
        wait_ticks = self._task_wait_ticks.get(task.task_id, 0)

        if slack >= 0:
            # 未超时任务优先，且越靠近截止时间越优先。
            tier = 0 if slack <= self.emergency_margin_ticks else 1
        elif slack >= -self.rescue_lateness_ticks:
            # 轻度超时：仍可能通过后续调度止损。
            tier = 2
        elif slack >= -self.hopeless_lateness_ticks:
            tier = 3
        else:
            # 重度超时：优先级最低，避免持续挤占可救任务吞吐。
            tier = 4

        # 同层内：先照顾等待更久的任务，再按 slack、task_id 稳定排序。
        return (tier, -wait_ticks, slack, task.task_id)

    def _pair_task_slack(self, task: Task, context: StrategyContext) -> float:
        eta = self._direct_eta(task, context)
        return task.deadline - (context.tick + eta)

    def _required_energy_for_single_task(
        self,
        vehicle: Vehicle,
        task: Task,
        context: StrategyContext,
    ) -> float:
        depot = context.depot_node
        d1 = context.oracle.shortest_distance(vehicle.current_node, depot)
        d2 = context.oracle.shortest_distance(depot, task.destination_node)
        d3 = context.oracle.shortest_distance(task.destination_node, depot)
        return (d1 + d2 + d3) * vehicle.energy_per_distance + self.reserve_energy

    def _plan_energy_safe(
        self,
        vehicle: Vehicle,
        key_nodes: Sequence[int],
        actions: Sequence[str],
        context: StrategyContext,
    ) -> bool:
        if len(key_nodes) != len(actions):
            return False

        battery = vehicle.battery
        cursor = vehicle.current_node
        energy_per_distance = vehicle.energy_per_distance

        for destination, action in zip(key_nodes, actions):
            ok, battery = self._can_traverse_segment_safely(
                start_node=cursor,
                target_node=destination,
                battery=battery,
                energy_per_distance=energy_per_distance,
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
        start_node: int,
        target_node: int,
        battery: float,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> Tuple[bool, float]:
        if start_node == target_node:
            return True, battery

        path = context.oracle.shortest_path(start_node, target_node)
        if len(path) < 2:
            return False, battery

        current_battery = battery
        for idx in range(1, len(path)):
            a = path[idx - 1]
            b = path[idx]
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

    def _direct_eta(self, task: Task, context: StrategyContext) -> int:
        depot = context.depot_node
        distance = (
            context.oracle.shortest_distance(depot, task.destination_node)
            + context.oracle.shortest_distance(task.destination_node, depot)
        )
        travel_ticks = math.ceil(distance / max(0.1, context.config.vehicle_speed))
        return travel_ticks + context.config.loading_duration + context.config.unloading_duration

    def _is_emergency_task(self, task: Task, context: StrategyContext) -> bool:
        remaining = task.deadline - context.tick
        pending_count = sum(
            1
            for t in context.tasks.values()
            if t.status == TaskStatus.PENDING and t.release_time <= context.tick
        )
        backlog_ratio = pending_count / max(1, len(context.vehicles))
        dynamic_margin = self.emergency_margin_ticks + min(
            self.backlog_margin_cap,
            int(math.ceil(backlog_ratio * self.backlog_margin_gain)),
        )
        return remaining <= self._direct_eta(task, context) + dynamic_margin

    def _pending_tasks(self, context: StrategyContext) -> List[Task]:
        return [
            task
            for task in context.tasks.values()
            if task.status == TaskStatus.PENDING and task.release_time <= context.tick
        ]

    def _vehicle_has_unfinished_work(self, vehicle: Vehicle, context: StrategyContext) -> bool:
        """车上或任务链里还有未完成任务时，不允许策略重新分配新任务。"""

        if vehicle.carried_weight > 1e-6:
            return True
        if vehicle.loaded_task_ids or vehicle.planned_task_ids:
            return True
        if vehicle.assigned_task_id is None:
            return False

        task = context.tasks.get(vehicle.assigned_task_id)
        return task is not None and task.status != TaskStatus.COMPLETED

    def _refresh_wait_counters(self, pending_tasks: Sequence[Task]) -> None:
        pending_ids = {task.task_id for task in pending_tasks}

        for task in pending_tasks:
            self._task_wait_ticks[task.task_id] = self._task_wait_ticks.get(task.task_id, 0) + 1

        removed_ids = [task_id for task_id in self._task_wait_ticks if task_id not in pending_ids]
        for task_id in removed_ids:
            self._task_wait_ticks.pop(task_id, None)

    def _pair_weight_feasible(self, vehicle: Vehicle, task_a: Task, task_b: Task) -> bool:
        return (task_a.weight + task_b.weight) <= vehicle.load_capacity + 1e-6

    def _key_path_distance(
        self,
        start_node: int,
        key_nodes: Sequence[int],
        context: StrategyContext,
    ) -> float:
        if not key_nodes:
            return 0.0

        total = 0.0
        cursor = start_node
        for node_id in key_nodes:
            total += context.oracle.shortest_distance(cursor, node_id)
            cursor = node_id
        return total

    def _apply_idle_rebalance(
        self,
        idle_vehicle_ids: Sequence[int],
        plans: Dict[int, VehiclePlan],
        context: StrategyContext,
        pending_task_count: int,
        urgent_task_count: int,
    ) -> None:
        if not idle_vehicle_ids:
            return

        idle_ids = list(idle_vehicle_ids)
        total_vehicles = len(context.vehicles)

        # 有任务压力时，回仓空闲车优先保持可派发，避免被过度拉去充电导致吞吐下降。
        if pending_task_count > 0 or urgent_task_count > 0:
            charge_threshold = context.config.low_battery_ratio
            for vehicle_id in idle_ids:
                vehicle = context.vehicles[vehicle_id]
                if self._battery_ratio(vehicle) >= charge_threshold:
                    continue
                charge_plan = self._build_charge_plan(vehicle, context)
                if charge_plan is not None:
                    plans[vehicle_id] = charge_plan
            return

        all_low_battery = all(
            self._battery_ratio(vehicle) < self.all_low_battery_ratio
            for vehicle in context.vehicles.values()
        )

        if all_low_battery:
            keep_count = 1
        elif urgent_task_count > 0:
            # 有紧急任务积压时，不主动压缩可派车数量。
            keep_count = len(idle_ids)
        elif pending_task_count > 0:
            # 任务压力越高，保留越多可派车，避免中规模超时堆积。
            demand_ratio = min(1.0, pending_task_count / max(1, total_vehicles))
            dynamic_keep_ratio = max(self.keep_idle_ratio, 0.45 + 0.45 * demand_ratio)
            keep_count = max(1, math.ceil(len(idle_ids) * dynamic_keep_ratio))
        else:
            if len(idle_ids) > total_vehicles * self.keep_idle_ratio:
                keep_count = max(1, math.ceil(total_vehicles * self.keep_idle_ratio))
            else:
                keep_count = len(idle_ids)

        idle_ids.sort(
            key=lambda vehicle_id: (
                -self._battery_ratio(context.vehicles[vehicle_id]),
                vehicle_id,
            )
        )

        keep_ids = set(idle_ids[:keep_count])
        for vehicle_id in idle_ids:
            if vehicle_id in keep_ids:
                vehicle = context.vehicles[vehicle_id]
                # 无任务压力时，对高电空闲车保留，偏低电则主动补能，避免下轮出车受限。
                if self._battery_ratio(vehicle) >= self.depot_charge_ratio_no_pressure:
                    continue
                charge_plan = self._build_charge_plan(vehicle, context)
                if charge_plan is not None:
                    plans[vehicle_id] = charge_plan
                continue
            vehicle = context.vehicles[vehicle_id]
            charge_plan = self._build_charge_plan(vehicle, context)
            if charge_plan is not None:
                plans[vehicle_id] = charge_plan

    def _build_charge_plan(self, vehicle: Vehicle, context: StrategyContext) -> Optional[VehiclePlan]:
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

    def _nearest_station(self, from_node: int, context: StrategyContext) -> Optional[ChargingStation]:
        best_station: Optional[ChargingStation] = None
        best_distance = float("inf")
        for station in context.stations.values():
            distance = context.oracle.shortest_distance(from_node, station.node_id)
            if distance < best_distance:
                best_distance = distance
                best_station = station
        return best_station

    def _nearest_reachable_station(
        self,
        from_node: int,
        available_energy: float,
        energy_per_distance: float,
        context: StrategyContext,
    ) -> Optional[ChargingStation]:
        stations: List[Tuple[float, ChargingStation]] = []
        for station in context.stations.values():
            distance = context.oracle.shortest_distance(from_node, station.node_id)
            stations.append((distance, station))
        stations.sort(key=lambda item: item[0])

        for distance, station in stations:
            required = distance * energy_per_distance + self.reserve_energy
            if required <= available_energy + 1e-6:
                return station
        return None

    @staticmethod
    def _battery_ratio(vehicle: Vehicle) -> float:
        return vehicle.battery / max(1.0, vehicle.battery_capacity)

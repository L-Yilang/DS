from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..models import ChargingStation, Task, TaskStatus, Vehicle, VehicleState
from .base import SchedulingStrategy, StrategyContext, VehiclePlan

_TRACE_PATH = Path("outputs/alns_trace.csv")
_TRACE_HEADER = [
    "tick",
    "scale_name",
    "candidate_task_count",
    "candidate_vehicle_count",
    "iteration_count",
    "planning_time_ms",
    "initial_score",
    "best_score",
    "score_improvement",
    "initial_distance",
    "best_distance",
    "initial_lateness",
    "best_lateness",
    "initial_unassigned",
    "best_unassigned",
    "accepted_count",
    "best_destroy",
    "best_repair",
    "destroy_weights",
    "repair_weights",
]

_DESTROY_NAMES = ["random", "lateness", "worst_distance", "related"]
_REPAIR_NAMES = ["greedy", "urgent", "weight", "energy_safe", "regret_k"]


@dataclass
class _RouteSolution:
    vehicle_routes: Dict[int, List[int]] = field(default_factory=dict)
    unassigned_tasks: Set[int] = field(default_factory=set)

    def copy(self) -> "_RouteSolution":
        return _RouteSolution(
            vehicle_routes={v: list(r) for v, r in self.vehicle_routes.items()},
            unassigned_tasks=set(self.unassigned_tasks),
        )


class EnergyAwareALNSStrategy(SchedulingStrategy):
    """Energy-aware Adaptive Large Neighborhood Search scheduling strategy."""

    name = "energy_aware_alns"

    # ── candidate pruning ──────────────────────────────────────────
    max_candidate_tasks: int = 30
    top_k_per_dim: int = 15

    # ── ALNS iterations ────────────────────────────────────────────
    max_iterations_small: int = 50
    max_iterations_medium: int = 80
    max_iterations_large: int = 120

    # ── simulated annealing ─────────────────────────────────────────
    initial_temperature: float = 5.0
    cooling_rate: float = 0.96
    min_temperature: float = 0.01

    # ── scoring weights ─────────────────────────────────────────────
    completed_reward_per_task: float = 10.0
    weight_reward_factor: float = 0.5
    distance_penalty_per_unit: float = 0.2
    lateness_penalty_per_tick: float = 2.0
    timeout_penalty_per_task: float = 20.0
    energy_infeasible_penalty: float = 1000.0
    charging_queue_penalty_per_pressure: float = 1.0
    unassigned_penalty_per_task: float = 8.0

    # ── energy ──────────────────────────────────────────────────────
    reserve_energy: float = 2.0
    low_battery_ratio: float = 0.22

    # ── destroy quantity ────────────────────────────────────────────
    destroy_fraction_min: float = 0.10
    destroy_fraction_max: float = 0.40

    # ── regret-k ─────────────────────────────────────────────────────
    regret_k: int = 3

    # ══════════════════════════════════════════════════════════════════
    #  Ablation study switches
    # ══════════════════════════════════════════════════════════════════

    # ALNS 核心机制
    enable_simulated_annealing: bool = True       # 关闭则只看改进解（爬山）
    enable_adaptive_operator_weight: bool = True  # 关闭则算子均匀随机选择

    # 破坏算子开关
    enable_related_removal: bool = True
    enable_lateness_removal: bool = True
    enable_worst_distance_removal: bool = True

    # 惩罚项开关
    enable_energy_penalty: bool = True
    enable_queue_penalty: bool = True

    # 优化增强
    enable_route_order_optimization: bool = True   # 路线内任务顺序优化（2-opt 启发式）
    enable_mid_route_charging: bool = False        # 路线中途充电站插入（实验中）

    # 直接关闭 ALNS 迭代 = 只用贪心初始解
    alns_iterations_enabled: bool = True

    # ─────────────────────────────────────────────────────────────────

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._destroy_weights: List[float] = []
        self._repair_weights: List[float] = []
        self._rng = random.Random()
        self._trace_initialized = False
        self._scale_name: str = ""
        self._best_destroy_name: str = ""
        self._best_repair_name: str = ""

    # ══════════════════════════════════════════════════════════════════
    #  Public entry
    # ══════════════════════════════════════════════════════════════════

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        t_start = time.perf_counter()

        avail_vehicles = self._available_vehicles(context)
        avail_tasks = self._available_tasks(context)

        # 检测规模名称
        self._scale_name = self._detect_scale_name(context)

        # 处理非仓库车辆
        plans: Dict[int, VehiclePlan] = {}
        depot_vehicle_ids: List[int] = []
        for vehicle_id in avail_vehicles:
            vehicle = context.vehicles[vehicle_id]
            if vehicle.current_node != context.depot_node:
                if self._battery_ratio(vehicle) < self.low_battery_ratio:
                    charge_plan = self._build_charge_plan(vehicle, context)
                    if charge_plan is not None:
                        plans[vehicle_id] = self._validate_plan_output(charge_plan)
                        continue
                plans[vehicle_id] = self._validate_plan_output(VehiclePlan(
                    vehicle_id=vehicle_id,
                    action=["keep"],
                    planned_path=[context.depot_node],
                ))
                continue
            depot_vehicle_ids.append(vehicle_id)

        if not avail_tasks:
            plans.update(self._idle_rebalance(depot_vehicle_ids, context))
            return plans

        # 任务极少 → 贪心
        if len(avail_tasks) <= 2:
            solution = self._build_initial_solution(depot_vehicle_ids, avail_tasks, context)
            plans.update(self._solution_to_plans(solution, context))
            return plans

        # 关闭 ALNS → 只用初始解
        if not self.alns_iterations_enabled:
            solution = self._build_initial_solution(depot_vehicle_ids, avail_tasks, context)
            plans.update(self._solution_to_plans(solution, context))
            return plans

        # ALNS 搜索
        initial = self._build_initial_solution(depot_vehicle_ids, avail_tasks, context)
        if not initial.vehicle_routes:
            plans.update(self._solution_to_plans(initial, context))
            return plans

        best_solution, best_score, trace_info = self._alns_search(
            initial, avail_tasks, context, t_start
        )
        trace_info["candidate_task_count"] = len(avail_tasks)
        trace_info["candidate_vehicle_count"] = len(depot_vehicle_ids)
        self._write_trace_row(context.tick, trace_info)

        plans.update(self._solution_to_plans(best_solution, context))

        # 确保所有候选车辆都有计划
        for vehicle_id in depot_vehicle_ids:
            if vehicle_id not in plans:
                vehicle = context.vehicles[vehicle_id]
                if self._battery_ratio(vehicle) < self.low_battery_ratio:
                    charge_plan = self._build_charge_plan(vehicle, context)
                    if charge_plan is not None:
                        plans[vehicle_id] = self._validate_plan_output(charge_plan)
                        continue
                if vehicle.current_node != context.depot_node:
                    plans[vehicle_id] = self._validate_plan_output(VehiclePlan(
                        vehicle_id=vehicle_id,
                        action=["keep"],
                        planned_path=[context.depot_node],
                    ))

        return plans

    def _detect_scale_name(self, context: StrategyContext) -> str:
        """从 horizon 推断规模名称。"""
        horizon = self._horizon_from_context(context)
        if horizon <= 550:
            return "small"
        elif horizon <= 900:
            return "medium"
        else:
            return "large"

    @staticmethod
    def _horizon_from_context(context: StrategyContext) -> int:
        return context.horizon

    # ══════════════════════════════════════════════════════════════════
    #  Idle rebalance
    # ══════════════════════════════════════════════════════════════════

    def _idle_rebalance(
        self, vehicle_ids: List[int], context: StrategyContext
    ) -> Dict[int, VehiclePlan]:
        plans: Dict[int, VehiclePlan] = {}
        for vehicle_id in vehicle_ids:
            vehicle = context.vehicles[vehicle_id]
            if self._battery_ratio(vehicle) < self.low_battery_ratio:
                charge_plan = self._build_charge_plan(vehicle, context)
                if charge_plan is not None:
                    plans[vehicle_id] = self._validate_plan_output(charge_plan)
                    continue
            if vehicle.current_node != context.depot_node:
                plans[vehicle_id] = self._validate_plan_output(VehiclePlan(
                    vehicle_id=vehicle_id,
                    action=["keep"],
                    planned_path=[context.depot_node],
                ))
        return plans

    # ══════════════════════════════════════════════════════════════════
    #  Candidate helpers
    # ══════════════════════════════════════════════════════════════════

    def _available_vehicles(self, context: StrategyContext) -> List[int]:
        ids: List[int] = []
        for vehicle_id, vehicle in sorted(context.vehicles.items()):
            if vehicle.state not in (
                VehicleState.IDLE,
                VehicleState.CHARGING,
                VehicleState.WAITING_CHARGE,
            ):
                continue
            if self._vehicle_has_unfinished_work(vehicle, context):
                continue
            ids.append(vehicle_id)
        return ids

    def _available_tasks(self, context: StrategyContext) -> List[Task]:
        pending = [
            t
            for t in context.tasks.values()
            if t.status == TaskStatus.PENDING and t.release_time <= context.tick
        ]
        if len(pending) <= self.max_candidate_tasks:
            return pending

        depot = context.depot_node
        selected: Dict[int, Task] = {}

        by_deadline = sorted(pending, key=lambda t: t.deadline)
        for t in by_deadline[: self.top_k_per_dim]:
            selected[t.task_id] = t

        by_wait = sorted(
            pending, key=lambda t: context.tick - t.release_time, reverse=True
        )
        for t in by_wait[: self.top_k_per_dim]:
            selected[t.task_id] = t

        by_weight = sorted(pending, key=lambda t: t.weight, reverse=True)
        for t in by_weight[: self.top_k_per_dim]:
            selected[t.task_id] = t

        by_dist = sorted(
            pending,
            key=lambda t: context.oracle.shortest_distance(depot, t.destination_node),
        )
        for t in by_dist[: self.top_k_per_dim]:
            selected[t.task_id] = t

        result = list(selected.values())
        if len(result) > self.max_candidate_tasks:
            result = result[: self.max_candidate_tasks]
        return result

    # ══════════════════════════════════════════════════════════════════
    #  Initial solution
    # ══════════════════════════════════════════════════════════════════

    def _build_initial_solution(
        self,
        vehicle_ids: List[int],
        tasks: List[Task],
        context: StrategyContext,
    ) -> _RouteSolution:
        solution = _RouteSolution()
        for vehicle_id in vehicle_ids:
            solution.vehicle_routes[vehicle_id] = []

        sorted_tasks = sorted(tasks, key=lambda t: self._task_slack(t, context))
        for task in sorted_tasks:
            best_vid, best_pos = self._insert_task_best_position(solution, task, context)
            if best_vid is not None and best_pos is not None:
                solution.vehicle_routes[best_vid].insert(best_pos, task.task_id)
            else:
                solution.unassigned_tasks.add(task.task_id)

        # 路线内顺序优化
        if self.enable_route_order_optimization:
            solution = self._optimize_all_routes(solution, context)

        return solution

    def _optimize_all_routes(
        self, solution: _RouteSolution, context: StrategyContext
    ) -> _RouteSolution:
        """对所有非空路线应用 2-opt 优化。"""
        for vehicle_id, route in list(solution.vehicle_routes.items()):
            if len(route) < 2:
                continue
            solution.vehicle_routes[vehicle_id] = self._two_opt_route(
                vehicle_id, route, context
            )
        return solution

    def _two_opt_route(
        self, vehicle_id: int, route: List[int], context: StrategyContext
    ) -> List[int]:
        """简单 2-opt 局部搜索：翻转子序列，接受首次改进。"""
        best_route = list(route)
        best_dist = self._route_distance(vehicle_id, best_route, context)

        improved = True
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route) + 1):
                    new_route = best_route[:i] + list(reversed(best_route[i:j])) + best_route[j:]
                    if not self._is_route_feasible(vehicle_id, new_route, context):
                        continue
                    new_dist = self._route_distance(vehicle_id, new_route, context)
                    if new_dist < best_dist - 1e-6:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def _route_distance(
        self, vehicle_id: int, task_ids: List[int], context: StrategyContext
    ) -> float:
        metrics = self._route_metrics(vehicle_id, task_ids, context)
        return metrics[0] if metrics else float("inf")

    def _task_slack(self, task: Task, context: StrategyContext) -> float:
        depot = context.depot_node
        dist = context.oracle.shortest_distance(depot, task.destination_node)
        travel_ticks = (2.0 * dist) / max(0.1, 1.0)
        service_ticks = context.config.loading_duration + context.config.unloading_duration
        est_finish = context.tick + travel_ticks + service_ticks
        return task.deadline - est_finish

    def _insertion_cost(
        self,
        vehicle_id: int,
        route: List[int],
        pos: int,
        task: Task,
        context: StrategyContext,
    ) -> Optional[float]:
        new_route = list(route)
        new_route.insert(pos, task.task_id)
        if not self._is_route_feasible(vehicle_id, new_route, context):
            return None

        old_metrics = self._route_metrics(vehicle_id, route, context)
        new_metrics = self._route_metrics(vehicle_id, new_route, context)
        old_dist = old_metrics[0] if old_metrics else 0.0
        new_dist = new_metrics[0] if new_metrics else float("inf")
        delta_dist = new_dist - old_dist
        delta_time = (new_metrics[2] if new_metrics else 0.0) - (
            old_metrics[2] if old_metrics else 0.0
        )
        return delta_dist + 0.5 * max(0.0, delta_time) + 0.01 * pos

    # ══════════════════════════════════════════════════════════════════
    #  Route metrics & feasibility
    # ══════════════════════════════════════════════════════════════════

    def _route_metrics(
        self,
        vehicle_id: int,
        task_ids: List[int],
        context: StrategyContext,
    ) -> Optional[Tuple[float, float, float, List[int]]]:
        vehicle = context.vehicles[vehicle_id]
        depot = context.depot_node
        path_nodes = [vehicle.current_node, depot]

        for task_id in task_ids:
            task = context.tasks.get(task_id)
            if task is None:
                return None
            path_nodes.append(task.destination_node)
        path_nodes.append(depot)

        total_distance = 0.0
        for i in range(len(path_nodes) - 1):
            d = context.oracle.shortest_distance(path_nodes[i], path_nodes[i + 1])
            total_distance += d

        speed = max(0.1, vehicle.speed)
        load_time = context.config.loading_duration
        unload_time = context.config.unloading_duration
        total_ticks = total_distance / speed + load_time + len(task_ids) * unload_time
        total_energy = total_distance * vehicle.energy_per_distance

        return total_distance, total_energy, total_ticks, path_nodes

    def _is_route_feasible(
        self,
        vehicle_id: int,
        task_ids: List[int],
        context: StrategyContext,
    ) -> bool:
        vehicle = context.vehicles[vehicle_id]
        depot = context.depot_node

        if not task_ids:
            return True

        # weight check
        total_weight = 0.0
        for task_id in task_ids:
            task = context.tasks.get(task_id)
            if task is None:
                return False
            if task.weight > vehicle.load_capacity + 1e-6:
                return False
            total_weight += task.weight
        if total_weight > vehicle.load_capacity + 1e-6:
            return False

        # energy check
        metrics = self._route_metrics(vehicle_id, task_ids, context)
        if metrics is None:
            return False
        total_energy = metrics[1]

        if vehicle.battery + 1e-6 < total_energy:
            return False

        battery_after = vehicle.battery - total_energy
        final_node = depot

        dist_to_nearest = min(
            (context.oracle.shortest_distance(final_node, station.node_id)
             for station in context.stations.values()),
            default=float("inf"),
        )
        energy_to_station = dist_to_nearest * vehicle.energy_per_distance
        energy_to_depot = (
            context.oracle.shortest_distance(final_node, depot)
            * vehicle.energy_per_distance
        )
        min_safety = min(energy_to_depot, energy_to_station) + self.reserve_energy

        if battery_after + 1e-6 < min_safety:
            best_station = self._select_best_station(
                final_node, battery_after,
                vehicle.energy_per_distance, context,
            )
            if best_station is None:
                return False

        # Per-segment check
        path_nodes = metrics[3]
        remaining_battery = vehicle.battery
        for i in range(len(path_nodes) - 1):
            a, b = path_nodes[i], path_nodes[i + 1]
            seg_dist = context.oracle.shortest_distance(a, b)
            seg_energy = seg_dist * vehicle.energy_per_distance
            if remaining_battery + 1e-6 < seg_energy:
                return False
            remaining_battery -= seg_energy
            dist_next_to_station = min(
                (context.oracle.shortest_distance(b, st.node_id)
                 for st in context.stations.values()),
                default=float("inf"),
            )
            dist_next_to_depot = context.oracle.shortest_distance(b, depot)
            safety_energy = (
                min(dist_next_to_depot, dist_next_to_station)
                * vehicle.energy_per_distance
                + self.reserve_energy
            )
            if remaining_battery + 1e-6 < safety_energy:
                return False

        return True

    # ══════════════════════════════════════════════════════════════════
    #  Station selection
    # ══════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════
    #  Solution evaluation
    # ══════════════════════════════════════════════════════════════════

    def _evaluate_solution(
        self,
        solution: _RouteSolution,
        context: StrategyContext,
    ) -> float:
        score = 0.0
        total_distance = 0.0
        lateness_sum = 0.0
        timeout_count = 0
        infeasible_count = 0
        charging_pressure_sum = 0.0
        completed_count = 0
        completed_weight = 0.0

        for vehicle_id, task_ids in solution.vehicle_routes.items():
            if not task_ids:
                continue

            metrics = self._route_metrics(vehicle_id, task_ids, context)
            if metrics is None:
                infeasible_count += 1
                continue

            dist, energy, total_ticks, path_nodes = metrics
            total_distance += dist
            battery_after = context.vehicles[vehicle_id].battery - energy

            # Check energy infeasibility
            final_node = path_nodes[-1]
            dist_nearest_station = min(
                context.oracle.shortest_distance(final_node, st.node_id)
                for st in context.stations.values()
            )
            energy_to_safe = (
                dist_nearest_station
                * context.vehicles[vehicle_id].energy_per_distance
            )
            if battery_after < energy_to_safe + self.reserve_energy:
                best_station = self._select_best_station(
                    final_node, battery_after,
                    context.vehicles[vehicle_id].energy_per_distance, context,
                )
                if best_station is None:
                    infeasible_count += 1
                    continue
                charging_pressure_sum += best_station.pressure_index()

            for idx, task_id in enumerate(task_ids):
                task = context.tasks.get(task_id)
                if task is None:
                    continue
                fraction = (idx + 1) / len(task_ids)
                task_finish = context.tick + total_ticks * fraction
                lateness = task_finish - task.deadline
                if lateness > 0:
                    lateness_sum += lateness
                    timeout_count += 1
                completed_count += 1
                completed_weight += task.weight

        completed_reward = (
            self.completed_reward_per_task * completed_count
            + self.weight_reward_factor * completed_weight
        )
        distance_penalty = self.distance_penalty_per_unit * total_distance
        lateness_penalty = self.lateness_penalty_per_tick * lateness_sum
        timeout_penalty = self.timeout_penalty_per_task * timeout_count
        energy_risk_penalty = (
            self.energy_infeasible_penalty * infeasible_count
            if self.enable_energy_penalty
            else 0.0
        )
        charging_queue_penalty = (
            self.charging_queue_penalty_per_pressure * charging_pressure_sum
            if self.enable_queue_penalty
            else 0.0
        )
        unassigned_penalty = (
            self.unassigned_penalty_per_task * len(solution.unassigned_tasks)
        )

        score = (
            completed_reward
            - distance_penalty
            - lateness_penalty
            - timeout_penalty
            - energy_risk_penalty
            - charging_queue_penalty
            - unassigned_penalty
        )
        return score

    def _evaluate_solution_detail(
        self, solution: _RouteSolution, context: StrategyContext
    ) -> Dict[str, float]:
        """返回评分明细字典，用于 trace 日志。"""
        total_distance = 0.0
        lateness_sum = 0.0
        completed_count = 0
        unassigned = len(solution.unassigned_tasks)

        for vehicle_id, task_ids in solution.vehicle_routes.items():
            if not task_ids:
                continue
            metrics = self._route_metrics(vehicle_id, task_ids, context)
            if metrics is None:
                continue
            total_distance += metrics[0]
            completed_count += len(task_ids)
            for idx, task_id in enumerate(task_ids):
                task = context.tasks.get(task_id)
                if task is None:
                    continue
                fraction = (idx + 1) / max(1, len(task_ids))
                task_finish = context.tick + metrics[2] * fraction
                lateness_sum += max(0.0, task_finish - task.deadline)

        return {
            "distance": total_distance,
            "lateness": lateness_sum,
            "completed": completed_count,
            "unassigned": unassigned,
        }

    # ══════════════════════════════════════════════════════════════════
    #  Destroy operators
    # ══════════════════════════════════════════════════════════════════

    def _build_destroy_ops(self) -> List[Tuple[str, Callable]]:
        ops: List[Tuple[str, Callable]] = [("random", self._destroy_random)]
        if self.enable_lateness_removal:
            ops.append(("lateness", self._destroy_lateness))
        if self.enable_worst_distance_removal:
            ops.append(("worst_distance", self._destroy_worst_distance))
        if self.enable_related_removal:
            ops.append(("related", self._destroy_related))
        return ops

    def _build_repair_ops(self) -> List[Tuple[str, Callable]]:
        """构建修复算子列表（可通过子类覆盖调整算子组合）。"""
        return [
            ("greedy", self._repair_greedy),
            ("urgent", self._repair_urgent),
            ("weight", self._repair_weight),
            ("energy_safe", self._repair_energy_safe),
            ("regret_k", self._repair_regret_k),
        ]

    def _destroy_random(
        self,
        solution: _RouteSolution,
        remove_count: int,
        context: StrategyContext,
    ) -> Tuple[_RouteSolution, Set[int]]:
        all_tasks = [
            tid for route in solution.vehicle_routes.values() for tid in route
        ]
        if not all_tasks:
            return solution, set()
        count = min(remove_count, len(all_tasks))
        removed = set(self._rng.sample(all_tasks, count))
        return self._remove_tasks(solution, removed), removed

    def _destroy_lateness(
        self,
        solution: _RouteSolution,
        remove_count: int,
        context: StrategyContext,
    ) -> Tuple[_RouteSolution, Set[int]]:
        scored = []
        for _vid, route in solution.vehicle_routes.items():
            for task_id in route:
                task = context.tasks.get(task_id)
                if task is None:
                    continue
                scored.append((task.deadline - context.tick, task_id))
        scored.sort(key=lambda x: x[0])
        count = min(remove_count, len(scored))
        removed = {task_id for _, task_id in scored[:count]}
        return self._remove_tasks(solution, removed), removed

    def _destroy_worst_distance(
        self,
        solution: _RouteSolution,
        remove_count: int,
        context: StrategyContext,
    ) -> Tuple[_RouteSolution, Set[int]]:
        contributions = []
        for vehicle_id, route in solution.vehicle_routes.items():
            full_metrics = self._route_metrics(vehicle_id, route, context)
            full_dist = full_metrics[0] if full_metrics else 0.0
            for i, task_id in enumerate(route):
                route_without = route[:i] + route[i + 1 :]
                without_metrics = self._route_metrics(vehicle_id, route_without, context)
                without_dist = without_metrics[0] if without_metrics else 0.0
                contributions.append((full_dist - without_dist, task_id))
        contributions.sort(key=lambda x: x[0], reverse=True)
        count = min(remove_count, len(contributions))
        removed = {task_id for _, task_id in contributions[:count]}
        return self._remove_tasks(solution, removed), removed

    def _destroy_related(
        self,
        solution: _RouteSolution,
        remove_count: int,
        context: StrategyContext,
    ) -> Tuple[_RouteSolution, Set[int]]:
        all_tasks = [
            tid for route in solution.vehicle_routes.values() for tid in route
        ]
        if not all_tasks:
            return solution, set()

        seed = self._rng.choice(all_tasks)
        seed_task = context.tasks.get(seed)
        seed_node = seed_task.destination_node if seed_task else context.depot_node

        distances = []
        for task_id in all_tasks:
            if task_id == seed:
                continue
            task = context.tasks.get(task_id)
            if task is None:
                continue
            d = context.oracle.shortest_distance(seed_node, task.destination_node)
            distances.append((d, task_id))
        distances.sort(key=lambda x: x[0])
        count = min(remove_count, len(distances))
        removed = {seed}
        for _, task_id in distances[:count]:
            removed.add(task_id)
        return self._remove_tasks(solution, removed), removed

    def _remove_tasks(
        self, solution: _RouteSolution, task_ids: Set[int]
    ) -> _RouteSolution:
        new_sol = _RouteSolution(
            unassigned_tasks=set(solution.unassigned_tasks),
        )
        for vid, route in solution.vehicle_routes.items():
            new_sol.vehicle_routes[vid] = [tid for tid in route if tid not in task_ids]
        return new_sol

    # ══════════════════════════════════════════════════════════════════
    #  Repair operators
    # ══════════════════════════════════════════════════════════════════

    _REPAIR_OPS: List[Tuple[str, Optional[str]]] = [
        ("greedy", None),
        ("urgent", "deadline"),
        ("weight", "-weight"),
        ("energy_safe", "energy"),
        ("regret_k", "regret"),
    ]

    def _repair_greedy(
        self, solution: _RouteSolution, removed_tasks: Set[int], context: StrategyContext
    ) -> _RouteSolution:
        return self._repair_generic(solution, removed_tasks, context, sort_key=None)

    def _repair_urgent(
        self, solution: _RouteSolution, removed_tasks: Set[int], context: StrategyContext
    ) -> _RouteSolution:
        return self._repair_generic(
            solution, removed_tasks, context, sort_key=lambda t: float(t.deadline)
        )

    def _repair_weight(
        self, solution: _RouteSolution, removed_tasks: Set[int], context: StrategyContext
    ) -> _RouteSolution:
        return self._repair_generic(
            solution, removed_tasks, context, sort_key=lambda t: -t.weight
        )

    def _repair_energy_safe(
        self, solution: _RouteSolution, removed_tasks: Set[int], context: StrategyContext
    ) -> _RouteSolution:
        return self._repair_generic(
            solution, removed_tasks, context,
            sort_key=lambda t: context.oracle.shortest_distance(
                context.depot_node, t.destination_node
            ),
        )

    def _repair_regret_k(
        self, solution: _RouteSolution, removed_tasks: Set[int], context: StrategyContext
    ) -> _RouteSolution:
        """Regret-k 插入：优先插入"可选位置少/代价差异大"的任务。"""
        k = max(2, self.regret_k)
        new_sol = solution.copy()
        tasks_to_insert = [
            context.tasks[tid]
            for tid in removed_tasks
            if tid in context.tasks and context.tasks[tid].status == TaskStatus.PENDING
        ]

        while tasks_to_insert:
            best_regret = -float("inf")
            best_idx: Optional[int] = None
            best_vid: Optional[int] = None
            best_pos: Optional[int] = None

            for idx, task in enumerate(tasks_to_insert):
                insertions = self._find_task_insertions(new_sol, task, context)
                if not insertions:
                    continue

                best_cost = insertions[0][0]
                if len(insertions) == 1:
                    regret = float("inf")
                else:
                    regret = sum(c - best_cost for c, _, _ in insertions[1:k])

                if regret > best_regret:
                    best_regret = regret
                    best_idx = idx
                    best_vid = insertions[0][1]
                    best_pos = insertions[0][2]

            if best_idx is None:
                for task in tasks_to_insert:
                    new_sol.unassigned_tasks.add(task.task_id)
                break

            task = tasks_to_insert.pop(best_idx)
            new_sol.vehicle_routes[best_vid].insert(best_pos, task.task_id)

        if self.enable_route_order_optimization:
            new_sol = self._optimize_all_routes(new_sol, context)

        return new_sol

    def _find_task_insertions(
        self,
        solution: _RouteSolution,
        task: Task,
        context: StrategyContext,
    ) -> List[Tuple[float, int, int]]:
        """返回任务所有可行插入方案，按代价升序排列 [(cost, vehicle_id, position), ...]."""
        result: List[Tuple[float, int, int]] = []
        for vehicle_id, route in solution.vehicle_routes.items():
            for pos in range(len(route) + 1):
                cost = self._insertion_cost(vehicle_id, route, pos, task, context)
                if cost is not None:
                    result.append((cost, vehicle_id, pos))
        result.sort(key=lambda x: x[0])
        return result

    def _repair_generic(
        self,
        solution: _RouteSolution,
        removed_tasks: Set[int],
        context: StrategyContext,
        sort_key=None,
    ) -> _RouteSolution:
        new_sol = solution.copy()
        tasks_to_insert = [
            context.tasks[tid]
            for tid in removed_tasks
            if tid in context.tasks and context.tasks[tid].status == TaskStatus.PENDING
        ]
        if sort_key is not None:
            tasks_to_insert.sort(key=sort_key)
        else:
            self._rng.shuffle(tasks_to_insert)

        for task in tasks_to_insert:
            best_vid, best_pos = self._insert_task_best_position(new_sol, task, context)
            if best_vid is not None and best_pos is not None:
                new_sol.vehicle_routes[best_vid].insert(best_pos, task.task_id)
            else:
                new_sol.unassigned_tasks.add(task.task_id)

        if self.enable_route_order_optimization:
            new_sol = self._optimize_all_routes(new_sol, context)

        return new_sol

    def _insert_task_best_position(
        self,
        solution: _RouteSolution,
        task: Task,
        context: StrategyContext,
    ) -> Tuple[Optional[int], Optional[int]]:
        best_vid = None
        best_pos = None
        best_cost = float("inf")

        for vehicle_id, route in solution.vehicle_routes.items():
            for pos in range(len(route) + 1):
                cost = self._insertion_cost(vehicle_id, route, pos, task, context)
                if cost is None:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_vid = vehicle_id
                    best_pos = pos

        return best_vid, best_pos

    # ══════════════════════════════════════════════════════════════════
    #  ALNS main loop
    # ══════════════════════════════════════════════════════════════════

    def _alns_search(
        self,
        initial_solution: _RouteSolution,
        tasks: List[Task],
        context: StrategyContext,
        t_start: float,
    ) -> Tuple[_RouteSolution, float, Dict]:
        # 确定迭代次数
        horizon = self._horizon_from_context(context)
        if horizon <= 550:
            max_iter = self.max_iterations_small
        elif horizon <= 900:
            max_iter = self.max_iterations_medium
        else:
            max_iter = self.max_iterations_large

        # 构建算子列表
        destroy_ops = self._build_destroy_ops()
        destroy_names = [name for name, _ in destroy_ops]
        destroy_funcs = [func for _, func in destroy_ops]

        repair_ops = self._build_repair_ops()
        repair_names = [name for name, _ in repair_ops]
        repair_funcs = [func for _, func in repair_ops]

        n_destroy = len(destroy_funcs)
        n_repair = len(repair_funcs)
        self._destroy_weights = [1.0] * n_destroy
        self._repair_weights = [1.0] * n_repair
        # regret_k 算子探索更充分，给更高初始权重
        if n_repair >= 5:
            self._repair_weights[4] = 3.0

        current = initial_solution.copy()
        initial_score = self._evaluate_solution(current, context)
        current_score = initial_score
        best = current.copy()
        best_score = current_score
        temperature = self.initial_temperature
        accepted_count = 0
        self._best_destroy_name = ""
        self._best_repair_name = ""

        initial_detail = self._evaluate_solution_detail(initial_solution, context)
        best_detail = dict(initial_detail)

        for iteration in range(max_iter):
            # 选择算子
            if self.enable_adaptive_operator_weight and n_destroy > 1:
                d_idx = self._rng.choices(
                    range(n_destroy), weights=self._destroy_weights, k=1
                )[0]
            else:
                d_idx = self._rng.randrange(n_destroy)

            if self.enable_adaptive_operator_weight and n_repair > 1:
                r_idx = self._rng.choices(
                    range(n_repair), weights=self._repair_weights, k=1
                )[0]
            else:
                r_idx = self._rng.randrange(n_repair)

            # 移除数量
            total_tasks = sum(len(r) for r in current.vehicle_routes.values())
            if total_tasks == 0:
                break
            remove_count = max(
                1,
                int(
                    total_tasks
                    * self._rng.uniform(
                        self.destroy_fraction_min, self.destroy_fraction_max
                    )
                ),
            )

            # destroy
            partial, removed = destroy_funcs[d_idx](current, remove_count, context)
            if not removed:
                continue

            # repair
            new_solution = repair_funcs[r_idx](partial, removed, context)

            # evaluate
            new_score = self._evaluate_solution(new_solution, context)

            # accept decision
            accepted = False
            if new_score > current_score:
                accepted = True
            elif self.enable_simulated_annealing and temperature > 1e-9:
                accept_prob = math.exp((new_score - current_score) / temperature)
                if self._rng.random() < accept_prob:
                    accepted = True

            if accepted:
                current = new_solution
                current_score = new_score
                accepted_count += 1

                if new_score > best_score:
                    best = new_solution.copy()
                    best_score = new_score
                    best_detail = self._evaluate_solution_detail(best, context)
                    self._best_destroy_name = destroy_names[d_idx]
                    self._best_repair_name = repair_names[r_idx]
                    if self.enable_adaptive_operator_weight:
                        self._destroy_weights[d_idx] += 0.2
                        self._repair_weights[r_idx] += 0.2
                elif self.enable_adaptive_operator_weight:
                    self._destroy_weights[d_idx] += 0.05
                    self._repair_weights[r_idx] += 0.05

            temperature *= self.cooling_rate
            if temperature < self.min_temperature:
                temperature = self.min_temperature

        planning_time_ms = (time.perf_counter() - t_start) * 1000.0

        trace_info = {
            "scale_name": self._scale_name,
            "iteration_count": max_iter,
            "planning_time_ms": round(planning_time_ms, 2),
            "initial_score": round(initial_score, 2),
            "best_score": round(best_score, 2),
            "score_improvement": round(best_score - initial_score, 2),
            "initial_distance": round(initial_detail.get("distance", 0.0), 2),
            "best_distance": round(best_detail.get("distance", 0.0), 2),
            "initial_lateness": round(initial_detail.get("lateness", 0.0), 2),
            "best_lateness": round(best_detail.get("lateness", 0.0), 2),
            "initial_unassigned": initial_detail.get("unassigned", 0),
            "best_unassigned": best_detail.get("unassigned", 0),
            "accepted_count": accepted_count,
            "best_destroy": self._best_destroy_name,
            "best_repair": self._best_repair_name,
            "destroy_weights": [round(w, 2) for w in self._destroy_weights],
            "repair_weights": [round(w, 2) for w in self._repair_weights],
        }
        return best, best_score, trace_info

    # ══════════════════════════════════════════════════════════════════
    #  Solution → VehiclePlan conversion
    # ══════════════════════════════════════════════════════════════════

    def _solution_to_plans(
        self,
        solution: _RouteSolution,
        context: StrategyContext,
    ) -> Dict[int, VehiclePlan]:
        plans: Dict[int, VehiclePlan] = {}
        depot = context.depot_node

        for vehicle_id, task_ids in solution.vehicle_routes.items():
            vehicle = context.vehicles[vehicle_id]

            if not task_ids:
                continue

            # 验证可行性
            if not self._is_route_feasible(vehicle_id, task_ids, context):
                fallback = self._fallback_plan_for_vehicle(vehicle, task_ids, context)
                if fallback is not None:
                    plans[vehicle_id] = self._validate_plan_output(fallback)
                continue

            # 检查充电中断
            planned_path, actions = self._build_route_path_and_actions(
                vehicle, task_ids, depot, context
            )
            if not self._can_interrupt_charge_for_plan(
                vehicle, planned_path, actions, context
            ):
                continue

            plans[vehicle_id] = self._validate_plan_output(VehiclePlan(
                vehicle_id=vehicle_id,
                task_id=list(task_ids),
                action=actions,
                planned_path=planned_path,
            ))

        return plans

    def _build_route_path_and_actions(
        self,
        vehicle: Vehicle,
        task_ids: List[int],
        depot: int,
        context: StrategyContext,
    ) -> Tuple[List[int], List[str]]:
        """构建单次装货、多点卸货路线，可选中途充电。"""
        planned_path = [depot]
        actions = ["load"]

        for task_id in task_ids:
            task = context.tasks.get(task_id)
            if task is None:
                continue
            planned_path.append(task.destination_node)
            actions.append("unload")

        # 终末充电检查
        metrics = self._route_metrics(vehicle.vehicle_id, task_ids, context)
        if metrics is not None:
            battery_after = vehicle.battery - metrics[1]
            if battery_after < self.reserve_energy:
                best_station = self._select_best_station(
                    depot, battery_after,
                    vehicle.energy_per_distance, context,
                )
                if best_station is not None:
                    planned_path.append(best_station.node_id)
                    actions.append("charge")

        planned_path.append(depot)
        actions.append("keep")

        return planned_path, actions

    def _build_single_task_plan(
        self,
        vehicle: Vehicle,
        task_id: int,
        context: StrategyContext,
    ) -> VehiclePlan:
        depot = context.depot_node
        task = context.tasks.get(task_id)
        if task is None:
            return VehiclePlan(
                vehicle_id=vehicle.vehicle_id,
                action=["keep"],
                planned_path=[depot],
            )
        return VehiclePlan(
            vehicle_id=vehicle.vehicle_id,
            task_id=[task_id],
            action=["load", "unload", "keep"],
            planned_path=[depot, task.destination_node, depot],
        )

    def _fallback_plan_for_vehicle(
        self,
        vehicle: Vehicle,
        task_ids: List[int],
        context: StrategyContext,
    ) -> Optional[VehiclePlan]:
        """兜底策略：低电量充电 → 最近可行单任务 → 回仓/keep。"""
        depot = context.depot_node

        # 1) 低电量优先充电
        if self._battery_ratio(vehicle) < self.low_battery_ratio:
            charge_plan = self._build_charge_plan(vehicle, context)
            if charge_plan is not None:
                return charge_plan

        # 2) 尝试最近可行单任务
        candidates = []
        for task_id in task_ids:
            task = context.tasks.get(task_id)
            if task is None:
                continue
            if task.weight > vehicle.load_capacity + 1e-6:
                continue
            if not self._is_route_feasible(vehicle.vehicle_id, [task_id], context):
                continue
            dist = context.oracle.shortest_distance(depot, task.destination_node)
            candidates.append((dist, task_id))
        candidates.sort(key=lambda x: x[0])
        if candidates:
            return self._build_single_task_plan(vehicle, candidates[0][1], context)

        # 3) 回仓或 keep
        if vehicle.current_node != depot:
            return VehiclePlan(
                vehicle_id=vehicle.vehicle_id,
                action=["keep"],
                planned_path=[depot],
            )
        return None

    def _build_charge_plan(
        self, vehicle: Vehicle, context: StrategyContext
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

    # ══════════════════════════════════════════════════════════════════
    #  Plan output validation
    # ══════════════════════════════════════════════════════════════════

    def _validate_plan_output(self, plan: VehiclePlan) -> VehiclePlan:
        """对输出计划做最终安全检查，非法则降级为安全的 keep/depot 计划。"""
        try:
            if len(plan.action) != len(plan.planned_path):
                return VehiclePlan(
                    vehicle_id=plan.vehicle_id,
                    action=["keep"],
                    planned_path=[0],
                )
            for action in plan.action:
                if action not in ("keep", "load", "unload", "charge"):
                    return VehiclePlan(
                        vehicle_id=plan.vehicle_id,
                        action=["keep"],
                        planned_path=[0],
                    )
        except Exception:
            return VehiclePlan(
                vehicle_id=plan.vehicle_id if hasattr(plan, "vehicle_id") else 0,
                action=["keep"],
                planned_path=[0],
            )
        return plan

    # ══════════════════════════════════════════════════════════════════
    #  Trace CSV
    # ══════════════════════════════════════════════════════════════════

    def _write_trace_row(self, tick: int, info: Dict) -> None:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not self._trace_initialized:
                _TRACE_PATH.write_bytes(b"")
                self._trace_initialized = True

            with open(_TRACE_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                try:
                    size = _TRACE_PATH.stat().st_size
                except Exception:
                    size = 0
                if size == 0:
                    writer.writerow(_TRACE_HEADER)
                writer.writerow([
                    tick,
                    info.get("scale_name", ""),
                    info.get("candidate_task_count", 0),
                    info.get("candidate_vehicle_count", 0),
                    info.get("iteration_count", 0),
                    info.get("planning_time_ms", 0.0),
                    info.get("initial_score", 0.0),
                    info.get("best_score", 0.0),
                    info.get("score_improvement", 0.0),
                    info.get("initial_distance", 0.0),
                    info.get("best_distance", 0.0),
                    info.get("initial_lateness", 0.0),
                    info.get("best_lateness", 0.0),
                    info.get("initial_unassigned", 0),
                    info.get("best_unassigned", 0),
                    info.get("accepted_count", 0),
                    info.get("best_destroy", ""),
                    info.get("best_repair", ""),
                    str(info.get("destroy_weights", [])),
                    str(info.get("repair_weights", [])),
                ])
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    #  Utilities
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _battery_ratio(vehicle: Vehicle) -> float:
        return vehicle.battery / max(1.0, vehicle.battery_capacity)

    @staticmethod
    def _vehicle_has_unfinished_work(
        vehicle: Vehicle, context: StrategyContext
    ) -> bool:
        if vehicle.carried_weight > 1e-6:
            return True
        if vehicle.loaded_task_ids:
            return True
        if vehicle.planned_task_ids:
            return True
        if vehicle.assigned_task_id is not None:
            task = context.tasks.get(vehicle.assigned_task_id)
            if task is not None and task.status != TaskStatus.COMPLETED:
                return True
        return False

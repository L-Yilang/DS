from __future__ import annotations

import math
import random
from collections import deque
from statistics import NormalDist
from typing import Deque, Dict, List, Optional

from .config import ScaleConfig, SimulationConfig
from .graph_utils import ShortestPathOracle, generate_road_graph
from .models import (
    ChargingStation,
    SimulationResult,
    Task,
    TaskStatus,
    Vehicle,
    VehicleState,
)
from .scoring import completion_score, distance_penalty
from .strategies.base import SchedulingStrategy, StrategyContext, VehiclePlan


class WorldManager:
    """
    世界管理器：仿真的唯一状态拥有者。

    职责：
    1. 初始化路网、车辆、充电站与随机环境。
    2. 推进世界真实状态（任务、充电、车辆位置、装卸进度）。
    3. 调用策略获取计划，并将计划安装为可执行路线/动作。
    4. 记录每个时间步快照，支持回放与离线分析。
    5. 在极端异常下触发失败并提前终止仿真。
    """

    def __init__(
        self,
        scale: ScaleConfig,
        config: SimulationConfig,
        strategy: SchedulingStrategy,
    ) -> None:
        self.scale = scale
        self.config = config
        self.strategy = strategy
        self.rng = random.Random(scale.seed)

        self.graph = generate_road_graph(scale.node_count, config, self.rng)
        self.oracle = ShortestPathOracle(self.graph)

        self.stations: Dict[int, ChargingStation] = {}
        self.station_by_node: Dict[int, ChargingStation] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        self.tasks: Dict[int, Task] = {}
        self.precomputed_tasks_by_tick: Dict[int, List[Task]] = {}

        self._task_counter = 1
        self.score = 0.0
        # 分数拆账：用于回放 UI 展示总分构成。
        self.completion_score_total = 0.0
        self.timeout_penalty_total = 0.0
        self.distance_penalty_total = 0.0
        self.failure_penalty_total = 0.0
        self.timeline: List[dict] = []
        self.events: List[str] = []

        # 保存当前 tick 策略输出，供回放和调试使用。
        self.last_strategy_plans: Dict[int, dict] = {}

        # 失败态：一旦触发，主循环提前结束。
        self.simulation_failed = False
        self.failure_reason: Optional[str] = None
        self.failure_tick: Optional[int] = None

        self._init_stations()
        self._init_vehicles()
        self._precompute_tasks()

    def run(self) -> SimulationResult:
        """
        运行完整仿真，若触发失败则提前终止。

        每个时间步包含两个固定阶段：
        - 阶段1 `world_manager_step`：先推进世界状态（物理与事件结算）。
        - 阶段2 `schedule_step`：再调用策略下发新计划。

        这种顺序保证策略始终基于“最新世界状态”做决策。
        """

        # 主循环：逐 tick 推进。
        for tick in range(self.scale.horizon):
            self.events = []

            # 操作 1：更新世界（任务生成、超时、充电、车辆推进等）。
            self.world_manager_step(tick)
            if self.simulation_failed:
                self._save_timestep(tick)
                break

            # 操作 2：策略输出全车计划并安装到车辆。
            self.schedule_step(tick)
            self._save_timestep(tick)

            if self.simulation_failed:
                break

        total_distance = sum(v.distance_travelled for v in self.vehicles.values())
        distance_component = distance_penalty(total_distance, self.config.distance_penalty_factor)
        self.distance_penalty_total += distance_component
        self.score += distance_component

        completed = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        overdue = sum(1 for task in self.tasks.values() if task.overdue_penalized)
        timeout_rate = overdue / max(1, len(self.tasks))

        return SimulationResult(
            scale_name=self.scale.name,
            strategy_name=self.strategy.name,
            total_score=round(self.score, 2),
            completed_tasks=completed,
            total_tasks=len(self.tasks),
            overdue_tasks=overdue,
            timeout_rate=round(timeout_rate, 4),
            total_distance=round(total_distance, 2),
            timeline=self.timeline,
            replay_meta=self._build_replay_meta(),
            simulation_failed=self.simulation_failed,
            failure_reason=self.failure_reason,
            failure_tick=self.failure_tick,
        )

    def world_manager_step(self, tick: int) -> None:
        """
        主循环操作 1：更新世界状态（不做策略选择）。

        该函数按固定顺序执行：
        1. `_spawn_tasks`：生成新任务。
        2. `_apply_timeout_penalties`：对超时未完成任务打罚分（仅一次）。
        3. `_advance_charging`：推进充电车辆电量，并尝试从排队队列补位。
        4. `_advance_vehicle`：逐车推进状态机（装卸计时、移动、到达动作触发）。
        5. `_promote_waiting_queue`：车辆推进后再次补位，减少入队到入桩的等待拍数。
        """

        self._spawn_tasks(tick)
        self._apply_timeout_penalties(tick)
        self._advance_charging()

        for vehicle in self.vehicles.values():
            self._advance_vehicle(vehicle, tick)
            if self.simulation_failed:
                return

        for station in self.stations.values():
            self._promote_waiting_queue(station)

    def schedule_step(self, tick: int) -> None:
        """
        主循环操作 2：调度决策。

        输入：完整世界上下文（图、任务、车辆、充电站、配置、时间）。
        输出：每辆车的 VehiclePlan（可为空计划）。

        本函数只负责“组装输入 + 调用策略 + 安装计划”，不直接实现策略规则。
        """

        context = StrategyContext(
            tick=tick,
            depot_node=self.config.depot_node,
            config=self.config,
            graph=self.graph,
            oracle=self.oracle,
            vehicles=self.vehicles,
            tasks=self.tasks,
            stations=self.stations,
        )

        plans = self.strategy.plan(context)
        self.last_strategy_plans = {
            vehicle_id: plan.to_dict()
            for vehicle_id, plan in plans.items()
        }
        self._apply_strategy_plans(plans, tick)

    def _apply_strategy_plans(self, plans: Dict[int, VehiclePlan], tick: int) -> None:
        """承接策略计划：安装空闲车辆计划，并支持充电态车辆被重新派单。"""

        for vehicle_id in sorted(self.vehicles):
            vehicle = self.vehicles[vehicle_id]
            plan = plans.get(vehicle_id, VehiclePlan(vehicle_id=vehicle_id))

            charging_reassign = (
                vehicle.state in (VehicleState.CHARGING, VehicleState.WAITING_CHARGE)
                and bool(plan.task_id)
            )

            # 非空闲车辆默认由 world_manager_step 推进；仅允许充电态车辆被策略改派任务。
            if vehicle.state != VehicleState.IDLE and not charging_reassign:
                continue

            if len(plan.action) != len(plan.planned_path):
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=(
                        f"策略输出非法：车辆#{vehicle.vehicle_id} action 与 planned_path 长度不一致"
                    ),
                )
                return

            if not plan.action and not plan.planned_path:
                continue

            if charging_reassign:
                self._interrupt_charge_for_new_plan(vehicle, tick)
                if self.simulation_failed:
                    return

            task_chain = self._bind_task_chain(vehicle, plan.task_id, tick)
            if self.simulation_failed:
                return

            self._install_vehicle_plan(
                vehicle=vehicle,
                task_chain=task_chain,
                destination_chain=plan.planned_path,
                action_chain=plan.action,
                tick=tick,
            )
            if self.simulation_failed:
                return

    def _interrupt_charge_for_new_plan(self, vehicle: Vehicle, tick: int) -> None:
        """允许策略中断充电态车辆，并释放旧的预留任务链。"""

        if vehicle.carried_weight > 1e-6:
            self._trigger_simulation_failure(
                tick=tick,
                reason=f"车辆#{vehicle.vehicle_id}载货时不能从充电计划切换到新任务",
            )
            return

        current_task = self.tasks.get(vehicle.assigned_task_id)
        if current_task is not None and current_task.status == TaskStatus.IN_PROGRESS:
            self._trigger_simulation_failure(
                tick=tick,
                reason=f"车辆#{vehicle.vehicle_id}存在进行中任务，不能在充电中改派",
            )
            return

        reserved_task_ids = {
            task_id
            for task_id in [
                vehicle.assigned_task_id,
                *vehicle.loaded_task_ids,
                *vehicle.planned_task_ids,
            ]
            if task_id is not None
        }
        for task_id in reserved_task_ids:
            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                continue
            if task.assigned_vehicle_id == vehicle.vehicle_id:
                task.assigned_vehicle_id = None
            if task.status == TaskStatus.ASSIGNED:
                task.status = TaskStatus.PENDING

        vehicle.assigned_task_id = None
        vehicle.loaded_task_ids.clear()
        vehicle.planned_task_ids.clear()
        self._remove_vehicle_from_station_lists(vehicle.vehicle_id)
        vehicle.visiting_station_id = None
        vehicle.state = VehicleState.IDLE

        self.events.append(f"车辆#{vehicle.vehicle_id}中断充电，改派执行新任务")

    def _bind_task_chain(self, vehicle: Vehicle, task_ids: List[int], tick: int) -> Deque[int]:
        """绑定策略下发的任务链，并检查任务有效性与占用冲突。"""

        bound: Deque[int] = deque()

        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task is None:
                continue
            if task.weight > vehicle.load_capacity:
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=(
                        f"策略输出非法：车辆#{vehicle.vehicle_id}尝试接取超载任务#{task.task_id}"
                    ),
                )
                return deque()
            if task.status == TaskStatus.COMPLETED:
                continue
            if task.assigned_vehicle_id not in (None, vehicle.vehicle_id):
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=(
                        f"策略输出冲突：任务#{task.task_id}已被车辆#{task.assigned_vehicle_id}占用"
                    ),
                )
                return deque()

            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.ASSIGNED
            task.assigned_vehicle_id = vehicle.vehicle_id
            bound.append(task.task_id)

        return bound

    def _install_vehicle_plan(
        self,
        vehicle: Vehicle,
        task_chain: Deque[int],
        destination_chain: List[int],
        action_chain: List[str],
        tick: int,
    ) -> None:
        """
        将关键目的地链 + 动作链展开为可执行状态。

        结果写入车辆：
        - oute`: 逐边节点序列
        - `planned_arrivals`: 每个关键目的地的预计到达时间步
        - `planned_actions`: 与 planned_arrivals 对齐
        - `planned_task_ids`: 任务链
        """

        vehicle.route.clear()
        vehicle.planned_arrivals.clear()
        vehicle.planned_actions.clear()
        vehicle.next_node = None
        vehicle.edge_remaining = 0.0
        vehicle.route_final_action = None
        vehicle.visiting_station_id = None

        vehicle.planned_task_ids = deque(task_chain)
        if vehicle.assigned_task_id is None and vehicle.planned_task_ids:
            vehicle.assigned_task_id = vehicle.planned_task_ids[0]

        cursor = vehicle.current_node
        eta = tick
        route_nodes: List[int] = []

        for destination, action in zip(destination_chain, action_chain):
            path = self.oracle.shortest_path(cursor, destination)
            if not path:
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=f"车辆#{vehicle.vehicle_id}无法规划到节点{destination}的路径",
                )
                return

            if len(path) > 1:
                for idx in range(1, len(path)):
                    a = path[idx - 1]
                    b = path[idx]
                    eta += math.ceil(self.graph.edge_distance(a, b) / max(0.1, vehicle.speed))
                    route_nodes.append(b)

            vehicle.planned_arrivals.append((destination, eta))
            vehicle.planned_actions.append(action)
            cursor = destination

        vehicle.route = deque(route_nodes)
        vehicle.state = VehicleState.MOVING if vehicle.route else VehicleState.IDLE

        # 若第一个目的地就是当前位置，立刻执行当前 tick 动作。
        self._apply_due_arrival_actions(vehicle, tick)

    def _trigger_simulation_failure(self, tick: int, reason: str) -> None:
        """进入失败态并记录原因（只触发一次）。"""

        if self.simulation_failed:
            return
        self.simulation_failed = True
        self.failure_tick = tick
        self.failure_reason = reason
        failure_penalty = -200.0
        self.failure_penalty_total += failure_penalty
        self.score += failure_penalty
        self.events.append(f"仿真失败：{reason}")

    def _init_stations(self) -> None:
        """初始化充电站：包含仓库站点，并使其余站点尽量分散。"""

        extra_count = max(0, self.scale.station_count - 1)
        station_nodes = [self.config.depot_node]
        station_nodes.extend(self._select_spread_station_nodes(extra_count))

        depot_piles = self.scale.depot_station_piles
        if depot_piles is None:
            depot_piles = self.scale.vehicle_count
        depot_piles = max(1, int(depot_piles))

        non_depot_min_piles, non_depot_max_piles = self.scale.non_depot_station_piles_range
        non_depot_min_piles = max(1, int(non_depot_min_piles))
        non_depot_max_piles = max(non_depot_min_piles, int(non_depot_max_piles))

        for station_id, node_id in enumerate(station_nodes, start=1):
            if node_id == self.config.depot_node:
                piles = depot_piles
                charge_rate = 3.2
            else:
                piles = self.rng.randint(non_depot_min_piles, non_depot_max_piles)
                charge_rate = round(self.rng.uniform(2.0, 2.8), 2)

            station = ChargingStation(
                station_id=station_id,
                node_id=node_id,
                piles=piles,
                charge_rate=charge_rate,
            )
            self.stations[station_id] = station
            self.station_by_node[node_id] = station

    def _select_spread_station_nodes(self, count: int) -> List[int]:
        """贪心最远点采样：每次选取与已选站点最小距离最大的节点。"""

        if count <= 0:
            return []

        depot = self.config.depot_node
        candidates = [node_id for node_id in self.graph.nodes if node_id != depot]
        if not candidates:
            return []

        first = max(candidates, key=lambda node_id: self._node_spatial_distance(node_id, depot))
        selected = [first]

        while len(selected) < count and len(selected) < len(candidates):
            remaining = [node_id for node_id in candidates if node_id not in selected]
            best_node: Optional[int] = None
            best_score = -1.0

            for node_id in remaining:
                min_dist = min(
                    self._node_spatial_distance(node_id, chosen)
                    for chosen in [depot, *selected]
                )

                if min_dist > best_score + 1e-9:
                    best_score = min_dist
                    best_node = node_id
                elif abs(min_dist - best_score) <= 1e-9 and self.rng.random() < 0.5:
                    best_node = node_id

            if best_node is None:
                break
            selected.append(best_node)

        return selected

    def _node_spatial_distance(self, a: int, b: int) -> float:
        """按节点坐标计算欧氏距离（用于站点分散采样）。"""
        node_a = self.graph.nodes[a]
        node_b = self.graph.nodes[b]
        return math.hypot(node_a.x - node_b.x, node_a.y - node_b.y)

    def _init_vehicles(self) -> None:
        """初始化车辆池：电池容量和载重能力在配置范围内随机化。"""
        min_load, max_load = self.config.vehicle_load_capacity_range
        for vehicle_id in range(1, self.scale.vehicle_count + 1):
            battery_capacity = round(
                self.config.vehicle_battery_capacity + self.rng.uniform(-0.0, 0.0), #不要随机化
                2,
            )
            load_capacity = round(self.rng.uniform(min_load, max_load), 2)
            vehicle = Vehicle(
                vehicle_id=vehicle_id,
                current_node=self.config.depot_node,
                battery_capacity=battery_capacity,
                load_capacity=load_capacity,
                speed=self.config.vehicle_speed,
                energy_per_distance=self.config.energy_per_distance,
                battery=battery_capacity,
            )
            self.vehicles[vehicle_id] = vehicle

    def _precompute_tasks(self) -> None:
        """在仿真开始前一次性预生成全部任务，并按 release_time 分桶。"""

        spawn_cutoff_tick = int(self.scale.horizon * 0.9)
        if spawn_cutoff_tick <= 0 or self.scale.task_count <= 0:
            return

        max_vehicle_load = max(v.load_capacity for v in self.vehicles.values())
        weight_upper = min(self.config.task_weight_range[1], max_vehicle_load * 0.95)
        weight_lower = min(self.config.task_weight_range[0], weight_upper)
        weight_mean = min(max(self.scale.task_weight_mean, weight_lower), weight_upper)
        weight_std = max(0.1, (weight_upper - weight_lower) / 6.0)

        for _ in range(self.scale.task_count):
            release_time = self._sample_task_release_tick(spawn_cutoff_tick)
            destination = self.rng.randrange(1, self.scale.node_count)
            weight = round(
                self._sample_clipped_normal(weight_mean, weight_std, weight_lower, weight_upper),
                2,
            )
            deadline = release_time + self.rng.randint(*self.config.deadline_range)

            task = Task(
                task_id=self._task_counter,
                release_time=release_time,
                destination_node=destination,
                weight=weight,
                deadline=deadline,
            )
            self.precomputed_tasks_by_tick.setdefault(release_time, []).append(task)
            self._task_counter += 1

        for tick_tasks in self.precomputed_tasks_by_tick.values():
            tick_tasks.sort(key=lambda task: task.task_id)

    def _sample_task_release_tick(self, spawn_cutoff_tick: int) -> int:
        """按配置的时间分布抽样任务 release_time。"""

        distribution = self.scale.task_time_distribution.lower()
        if distribution == "uniform":
            position = self.rng.random()
        elif distribution == "gaussian":
            position = self._sample_truncated_normal(
                mean=self.scale.task_time_mean_ratio,
                std=max(1e-6, self.scale.task_time_std_ratio),
                lower=0.0,
                upper=1.0,
            )
        else:
            raise ValueError(f"未知任务时间分布: {self.scale.task_time_distribution}")

        position = min(max(position, 0.0), 1.0 - 1e-9)
        return min(spawn_cutoff_tick - 1, int(position * spawn_cutoff_tick))

    def _sample_clipped_normal(
        self,
        mean: float,
        std: float,
        lower: float,
        upper: float,
    ) -> float:
        """抽样截断高斯值，用于在范围内控制重量均值。"""

        return self._sample_truncated_normal(
            mean=mean,
            std=max(1e-6, std),
            lower=lower,
            upper=upper,
        )

    def _sample_truncated_normal(
        self,
        mean: float,
        std: float,
        lower: float,
        upper: float,
    ) -> float:
        """
        从 [lower, upper] 上的截断高斯分布采样。

        与简单 clip 不同，落在区间外的概率质量会重新归一化到区间内部，
        避免大量样本堆积在边界点。
        """

        if upper <= lower:
            return lower

        std = max(1e-6, std)
        dist = NormalDist(mu=mean, sigma=std)
        cdf_lower = dist.cdf(lower)
        cdf_upper = dist.cdf(upper)

        if cdf_upper - cdf_lower <= 1e-12:
            return min(max(mean, lower), upper)

        quantile = self.rng.uniform(cdf_lower, cdf_upper)
        value = dist.inv_cdf(quantile)
        return min(max(value, lower), upper)

    def _spawn_tasks(self, tick: int) -> None:
        """释放当前 tick 预生成的任务，并写入事件日志。"""

        for task in self.precomputed_tasks_by_tick.get(tick, []):
            self.tasks[task.task_id] = task
            self.events.append(
                f"生成任务#{task.task_id}: 目的地{task.destination_node}, 重量{task.weight}, 截止{task.deadline}"
            )

    def _apply_timeout_penalties(self, tick: int) -> None:
        """对超时任务施加一次性罚分。"""
        for task in self.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                continue
            if tick > task.deadline and not task.overdue_penalized:
                penalty = -self.config.timeout_penalty
                self.timeout_penalty_total += penalty
                self.score += penalty
                task.overdue_penalized = True
                self.events.append(f"任务#{task.task_id}超时扣分{penalty}")

    def _advance_charging(self) -> None:
        """推进各充电站状态：加电、充满释放、队列补位。"""
        for station in self.stations.values():
            for vehicle_id in list(station.charging_vehicle_ids):
                vehicle = self.vehicles[vehicle_id]
                vehicle.battery = min(vehicle.battery_capacity, vehicle.battery + station.charge_rate)

                if vehicle.battery >= vehicle.battery_capacity - 1e-6:
                    station.charging_vehicle_ids.remove(vehicle_id)
                    vehicle.state = VehicleState.IDLE
                    vehicle.visiting_station_id = None
                    self.events.append(
                        f"车辆#{vehicle.vehicle_id}在充电站{station.station_id}完成充电"
                    )

            self._promote_waiting_queue(station)

    def _promote_waiting_queue(self, station: ChargingStation) -> None:
        """若有空桩则从等待队列补位到充电列表。"""
        while len(station.charging_vehicle_ids) < station.piles and station.queue_vehicle_ids:
            vehicle_id = station.queue_vehicle_ids.popleft()
            vehicle = self.vehicles[vehicle_id]
            if vehicle.current_node != station.node_id:
                continue
            if vehicle.state != VehicleState.WAITING_CHARGE:
                continue

            station.charging_vehicle_ids.append(vehicle_id)
            vehicle.state = VehicleState.CHARGING
            vehicle.visiting_station_id = station.station_id
            self.events.append(f"车辆#{vehicle.vehicle_id}开始在充电站{station.station_id}充电")

    def _advance_vehicle(self, vehicle: Vehicle, tick: int) -> None:
        """推进单车状态机：装卸计时、行驶推进、空闲到达动作检查。"""
        if vehicle.state in (VehicleState.LOADING, VehicleState.UNLOADING):
            vehicle.operation_timer -= 1
            if vehicle.operation_timer <= 0:
                if vehicle.state == VehicleState.LOADING:
                    self._finish_loading(vehicle, tick)
                else:
                    self._finish_unloading(vehicle, tick)
            return

        if vehicle.state == VehicleState.MOVING:
            self._advance_moving_vehicle(vehicle, tick)
            return

        if vehicle.state == VehicleState.IDLE:
            # 空闲态先由世界管理器做“恢复计划”兜底，再执行到站动作。
            self._recover_vehicle_plan_if_needed(vehicle, tick)
            if self.simulation_failed:
                return
            if vehicle.state == VehicleState.IDLE:
                self._apply_due_arrival_actions(vehicle, tick)

    def _advance_moving_vehicle(self, vehicle: Vehicle, tick: int) -> None:
        """推进行驶态车辆：选下一跳、扣电，并在到站时触发动作。"""
        if vehicle.next_node is None:
            if not vehicle.route:
                vehicle.state = VehicleState.IDLE
                self._apply_due_arrival_actions(vehicle, tick)
                return

            next_node = vehicle.route[0]
            edge_distance = self.graph.edge_distance(vehicle.current_node, next_node)
            if not self._can_start_next_edge(vehicle, next_node, edge_distance):
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=(
                        f"车辆#{vehicle.vehicle_id}电量不足，无法从节点{vehicle.current_node}"
                        f"行驶到节点{next_node}"
                    ),
                )
                return

            vehicle.next_node = next_node
            vehicle.edge_remaining = edge_distance
            vehicle.route.popleft()

        move_distance = min(vehicle.speed, vehicle.edge_remaining)
        vehicle.edge_remaining -= move_distance
        vehicle.battery = max(0.0, vehicle.battery - move_distance * vehicle.energy_per_distance)
        vehicle.distance_travelled += move_distance

        if vehicle.edge_remaining > 1e-6:
            return

        vehicle.current_node = int(vehicle.next_node)
        vehicle.next_node = None

        self._apply_due_arrival_actions(vehicle, tick)
        if self.simulation_failed:
            return

        if vehicle.state == VehicleState.MOVING and not vehicle.route and vehicle.next_node is None:
            vehicle.state = VehicleState.IDLE

    def _apply_due_arrival_actions(self, vehicle: Vehicle, tick: int) -> None:
        """按 planned_arrivals/planned_actions 对齐关系，在到站时执行动作。"""

        while vehicle.planned_arrivals and vehicle.planned_actions:
            destination, expected_tick = vehicle.planned_arrivals[0]
            if destination != vehicle.current_node or tick < expected_tick:
                break

            vehicle.planned_arrivals.pop(0)
            action = vehicle.planned_actions.popleft()

            if action == "keep":
                continue

            if action == "load":
                self._start_loading(vehicle, tick)
                return

            if action == "unload":
                self._start_unloading(vehicle, tick)
                return

            if action == "charge":
                station = self.station_by_node.get(vehicle.current_node)
                if station is None:
                    self._trigger_simulation_failure(
                        tick=tick,
                        reason=(
                            f"车辆#{vehicle.vehicle_id}到达节点{vehicle.current_node}执行充电动作失败："
                            "该节点不是充电站"
                        ),
                    )
                    return
                self._enqueue_or_charge(vehicle, station)
                return

            self.events.append(f"未知动作{action}，按 keep 处理")

        if vehicle.state == VehicleState.IDLE and vehicle.route:
            vehicle.state = VehicleState.MOVING

    def _can_start_next_edge(self, vehicle: Vehicle, next_node: int, edge_distance: float) -> bool:
        """边起步安全校验：当前边能耗 + 兜底能耗 + 安全冗余。"""
        energy_for_edge = edge_distance * vehicle.energy_per_distance
        if vehicle.battery + 1e-6 < energy_for_edge:
            return False

        energy_to_depot = self.oracle.shortest_distance(next_node, self.config.depot_node) * vehicle.energy_per_distance
        energy_to_station = self._nearest_station_energy(next_node, vehicle.energy_per_distance)
        safety_reserve = 2.0
        required = energy_for_edge + min(energy_to_depot, energy_to_station) + safety_reserve
        return vehicle.battery + 1e-6 >= required

    def _nearest_station_energy(self, node_id: int, energy_per_distance: float) -> float:
        """从指定节点到最近充电站的理论最小能耗。"""
        return min(
            self.oracle.shortest_distance(node_id, station.node_id) * energy_per_distance
            for station in self.stations.values()
        )

    def _sync_vehicle_task_queues(self, vehicle: Vehicle) -> None:
        """清理车辆任务链中的已完成/失效任务，并保持顺序稳定。"""

        loaded_remaining: List[int] = []
        seen_loaded: set[int] = set()
        for task_id in list(vehicle.loaded_task_ids):
            if task_id in seen_loaded:
                continue
            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                continue
            loaded_remaining.append(task_id)
            seen_loaded.add(task_id)
        vehicle.loaded_task_ids = deque(loaded_remaining)

        planned_remaining: List[int] = []
        seen_planned: set[int] = set()
        for task_id in list(vehicle.planned_task_ids):
            if task_id in seen_planned:
                continue
            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                continue
            planned_remaining.append(task_id)
            seen_planned.add(task_id)
        vehicle.planned_task_ids = deque(planned_remaining)

        if vehicle.assigned_task_id is None:
            return

        assigned_task = self.tasks.get(vehicle.assigned_task_id)
        if assigned_task is None or assigned_task.status == TaskStatus.COMPLETED:
            vehicle.assigned_task_id = None

    def _ensure_next_load_task_from_chain(self, vehicle: Vehicle, tick: int) -> Optional[Task]:
        """从任务链中挑选下一个待装货任务。"""

        self._sync_vehicle_task_queues(vehicle)
        loaded_task_ids = set(vehicle.loaded_task_ids)

        for task_id in vehicle.planned_task_ids:
            if task_id in loaded_task_ids:
                continue

            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                continue

            if task.weight > vehicle.load_capacity:
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=f"车辆#{vehicle.vehicle_id}任务链含超载任务#{task.task_id}",
                )
                return None

            if task.assigned_vehicle_id not in (None, vehicle.vehicle_id):
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=f"任务#{task.task_id}被其它车辆占用，无法执行",
                )
                return None

            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.ASSIGNED
            task.assigned_vehicle_id = vehicle.vehicle_id
            vehicle.assigned_task_id = task.task_id
            return task

        vehicle.assigned_task_id = None
        return None

    def _ensure_next_unload_task(self, vehicle: Vehicle, tick: int) -> Optional[Task]:
        """从车上货物队列中挑选下一个待卸货任务。"""

        self._sync_vehicle_task_queues(vehicle)

        while vehicle.loaded_task_ids:
            task_id = vehicle.loaded_task_ids[0]
            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                vehicle.loaded_task_ids.popleft()
                continue

            if task.assigned_vehicle_id not in (None, vehicle.vehicle_id):
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=f"任务#{task.task_id}被其它车辆占用，无法卸货",
                )
                return None

            task.assigned_vehicle_id = vehicle.vehicle_id
            vehicle.assigned_task_id = task.task_id
            return task

        vehicle.assigned_task_id = None
        return None

    def _collect_remaining_task_chain(self, vehicle: Vehicle) -> List[int]:
        """提取并清理车辆剩余任务链，优先保留车上货物的卸货顺序。"""

        remaining: List[int] = []
        seen: set[int] = set()
        self._sync_vehicle_task_queues(vehicle)
        loaded_task_ids = set(vehicle.loaded_task_ids)

        def append_if_valid(task_id: Optional[int]) -> None:
            if task_id is None or task_id in seen:
                return
            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                return
            seen.add(task_id)
            remaining.append(task_id)

        for task_id in list(vehicle.loaded_task_ids):
            append_if_valid(task_id)
        for task_id in list(vehicle.planned_task_ids):
            append_if_valid(task_id)

        vehicle.planned_task_ids = deque(remaining)
        vehicle.loaded_task_ids = deque(
            task_id for task_id in remaining if task_id in loaded_task_ids
        )
        vehicle.assigned_task_id = (
            vehicle.loaded_task_ids[0]
            if vehicle.loaded_task_ids
            else (remaining[0] if remaining else None)
        )
        return remaining

    def _recover_vehicle_plan_if_needed(self, vehicle: Vehicle, tick: int) -> None:
        """
        世界侧恢复机制：
        当车辆空闲且已丢失可执行路线缓存时，自动重建当前任务的执行计划。
        """

        if vehicle.state != VehicleState.IDLE:
            return
        if vehicle.route or vehicle.planned_arrivals or vehicle.planned_actions:
            return
        if vehicle.next_node is not None or vehicle.edge_remaining > 1e-6:
            return

        remaining_task_ids = self._collect_remaining_task_chain(vehicle)
        if not remaining_task_ids:
            return

        loaded_task_ids = list(vehicle.loaded_task_ids)
        if loaded_task_ids:
            destination_chain: List[int] = []
            action_chain: List[str] = []
            for task_id in loaded_task_ids:
                task = self.tasks.get(task_id)
                if task is None:
                    continue
                destination_chain.append(task.destination_node)
                action_chain.append("unload")
            if not destination_chain:
                return
            vehicle.assigned_task_id = loaded_task_ids[0]
        else:
            current_task = self.tasks.get(remaining_task_ids[0])
            if current_task is None:
                return

            if current_task.weight > vehicle.load_capacity:
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=f"车辆#{vehicle.vehicle_id}恢复任务链时发现超载任务#{current_task.task_id}",
                )
                return

            if current_task.assigned_vehicle_id not in (None, vehicle.vehicle_id):
                self._trigger_simulation_failure(
                    tick=tick,
                    reason=(
                        f"车辆#{vehicle.vehicle_id}恢复任务链时冲突："
                        f"任务#{current_task.task_id}已被车辆#{current_task.assigned_vehicle_id}占用"
                    ),
                )
                return

            if current_task.status == TaskStatus.PENDING:
                current_task.status = TaskStatus.ASSIGNED
            current_task.assigned_vehicle_id = vehicle.vehicle_id
            vehicle.assigned_task_id = current_task.task_id
            destination_chain = [self.config.depot_node, current_task.destination_node]
            action_chain = ["load", "unload"]

        self.events.append(
            f"车辆#{vehicle.vehicle_id}由世界管理器恢复任务链，剩余任务{remaining_task_ids}"
        )
        self._install_vehicle_plan(
            vehicle=vehicle,
            task_chain=deque(remaining_task_ids),
            destination_chain=destination_chain,
            action_chain=action_chain,
            tick=tick,
        )

    def _start_loading(self, vehicle: Vehicle, tick: int) -> None:
        """开始装货：仅允许在仓库执行。"""
        if vehicle.current_node != self.config.depot_node:
            self._trigger_simulation_failure(
                tick=tick,
                reason=f"车辆#{vehicle.vehicle_id}在非仓库节点尝试装货",
            )
            return

        task = self._ensure_next_load_task_from_chain(vehicle, tick)
        if task is None:
            vehicle.state = VehicleState.IDLE
            return

        if vehicle.carried_weight + task.weight > vehicle.load_capacity + 1e-6:
            self._trigger_simulation_failure(
                tick=tick,
                reason=(
                    f"车辆#{vehicle.vehicle_id}装载任务#{task.task_id}后超载："
                    f"{round(vehicle.carried_weight + task.weight, 2)} > {vehicle.load_capacity}"
                ),
            )
            return

        vehicle.state = VehicleState.LOADING
        vehicle.operation_timer = self.config.loading_duration
        vehicle.carried_weight += task.weight
        self.events.append(f"车辆#{vehicle.vehicle_id}开始装货(任务#{task.task_id})")

    def _finish_loading(self, vehicle: Vehicle, tick: int) -> None:
        """装货计时结束后，任务状态切换到 IN_PROGRESS。"""
        if vehicle.assigned_task_id is None:
            vehicle.state = VehicleState.IDLE
            return

        task = self.tasks.get(vehicle.assigned_task_id)
        if task is None or task.status == TaskStatus.COMPLETED:
            vehicle.state = VehicleState.IDLE
            return

        task.status = TaskStatus.IN_PROGRESS
        if task.task_id not in vehicle.loaded_task_ids:
            vehicle.loaded_task_ids.append(task.task_id)
        self.events.append(f"车辆#{vehicle.vehicle_id}装货完成，前往任务#{task.task_id}目的地")

        vehicle.state = VehicleState.MOVING if vehicle.route else VehicleState.IDLE
        self._apply_due_arrival_actions(vehicle, tick)

    def _start_unloading(self, vehicle: Vehicle, tick: int) -> None:
        """开始卸货；若无任务则直接回空闲。"""
        task = self._ensure_next_unload_task(vehicle, tick)
        if task is None:
            vehicle.state = VehicleState.IDLE
            return

        if vehicle.current_node != task.destination_node:
            self._trigger_simulation_failure(
                tick=tick,
                reason=(
                    f"车辆#{vehicle.vehicle_id}在节点{vehicle.current_node}尝试卸载任务#{task.task_id}，"
                    f"但目的地应为节点{task.destination_node}"
                ),
            )
            return

        vehicle.state = VehicleState.UNLOADING
        vehicle.operation_timer = self.config.unloading_duration
        self.events.append(f"车辆#{vehicle.vehicle_id}开始卸货(任务#{task.task_id})")

    def _finish_unloading(self, vehicle: Vehicle, tick: int) -> None:
        """卸货计时结束后结算得分并清理车辆任务状态。"""
        if vehicle.assigned_task_id is None:
            vehicle.state = VehicleState.IDLE
            vehicle.carried_weight = max(0.0, vehicle.carried_weight)
            self._apply_due_arrival_actions(vehicle, tick)
            return

        task = self.tasks.get(vehicle.assigned_task_id)
        if task is None:
            missing_task_id = vehicle.assigned_task_id
            vehicle.assigned_task_id = None
            vehicle.state = VehicleState.IDLE
            vehicle.loaded_task_ids = deque(
                task_id for task_id in vehicle.loaded_task_ids if task_id != missing_task_id
            )
            self._apply_due_arrival_actions(vehicle, tick)
            return

        task.status = TaskStatus.COMPLETED
        task.completion_time = tick

        reward = completion_score(
            task,
            completion_time=tick,
            base_reward=self.config.completion_base_reward,
            late_penalty_factor=self.config.late_completion_penalty_factor,
        )
        self.completion_score_total += reward
        self.score += reward
        self.events.append(f"任务#{task.task_id}完成，得分变化{round(reward, 2)}")

        if vehicle.loaded_task_ids and vehicle.loaded_task_ids[0] == task.task_id:
            vehicle.loaded_task_ids.popleft()
        elif task.task_id in vehicle.loaded_task_ids:
            vehicle.loaded_task_ids = deque(
                task_id for task_id in vehicle.loaded_task_ids if task_id != task.task_id
            )

        if vehicle.planned_task_ids and vehicle.planned_task_ids[0] == task.task_id:
            vehicle.planned_task_ids.popleft()
        else:
            vehicle.planned_task_ids = deque(
                task_id for task_id in vehicle.planned_task_ids if task_id != task.task_id
            )

        vehicle.assigned_task_id = None
        vehicle.carried_weight = max(0.0, vehicle.carried_weight - task.weight)

        vehicle.state = VehicleState.MOVING if vehicle.route else VehicleState.IDLE
        self._apply_due_arrival_actions(vehicle, tick)

    def _enqueue_or_charge(self, vehicle: Vehicle, station: ChargingStation) -> None:
        """到达充电站后执行入桩或排队，不清空未完成任务的路线与动作缓存。"""
        self._remove_vehicle_from_station_lists(vehicle.vehicle_id)

        # 充电是临时状态切换；保留 route/planned_arrivals/planned_actions，
        # 充电完成后可由 world_manager_step 自动续跑后续任务链。
        vehicle.next_node = None
        vehicle.edge_remaining = 0.0
        vehicle.route_final_action = None

        if len(station.charging_vehicle_ids) < station.piles:
            station.charging_vehicle_ids.append(vehicle.vehicle_id)
            vehicle.state = VehicleState.CHARGING
            vehicle.visiting_station_id = station.station_id
            self.events.append(f"车辆#{vehicle.vehicle_id}直接开始充电")
            return

        if vehicle.vehicle_id not in station.queue_vehicle_ids:
            station.queue_vehicle_ids.append(vehicle.vehicle_id)
        vehicle.state = VehicleState.WAITING_CHARGE
        vehicle.visiting_station_id = station.station_id
        self.events.append(f"车辆#{vehicle.vehicle_id}进入充电排队")

    def _remove_vehicle_from_station_lists(self, vehicle_id: int) -> None:
        """将车辆从所有充电站的充电列表与排队列表移除（幂等）。"""
        for station in self.stations.values():
            if vehicle_id in station.charging_vehicle_ids:
                station.charging_vehicle_ids.remove(vehicle_id)
            if vehicle_id in station.queue_vehicle_ids:
                station.queue_vehicle_ids.remove(vehicle_id)

    def _save_timestep(self, tick: int) -> None:
        """保存当前 tick 的完整快照，供回放 UI 与调试分析使用。"""
        pending = 0
        assigned = 0
        in_progress = 0
        completed = 0

        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                pending += 1
            elif task.status == TaskStatus.ASSIGNED:
                assigned += 1
            elif task.status == TaskStatus.IN_PROGRESS:
                in_progress += 1
            elif task.status == TaskStatus.COMPLETED:
                completed += 1

        snapshot = {
            "tick": tick,
            "score": round(self.score, 2),
            "task_stats": {
                "total": len(self.tasks),
                "pending": pending,
                "assigned": assigned,
                "in_progress": in_progress,
                "completed": completed,
                "overdue": sum(1 for t in self.tasks.values() if t.overdue_penalized),
                "timeout_rate": round(
                    sum(1 for t in self.tasks.values() if t.overdue_penalized) / max(1, len(self.tasks)),
                    4,
                ),
            },
            "vehicles": [
                {
                    "vehicle_id": vehicle.vehicle_id,
                    "state": vehicle.state.value,
                    "current_node": vehicle.current_node,
                    "battery": round(vehicle.battery, 2),
                    "battery_capacity": vehicle.battery_capacity,
                    "load_capacity": vehicle.load_capacity,
                    "assigned_task_id": vehicle.assigned_task_id,
                    "loaded_task_ids": list(vehicle.loaded_task_ids),
                    "carried_weight": vehicle.carried_weight,
                    "distance_travelled": round(vehicle.distance_travelled, 2),
                    "route": list(vehicle.route),
                    "planned_arrivals": list(vehicle.planned_arrivals),
                    "planned_actions": list(vehicle.planned_actions),
                    "planned_task_ids": list(vehicle.planned_task_ids),
                    "next_node": vehicle.next_node,
                    "edge_remaining": round(vehicle.edge_remaining, 2),
                }
                for vehicle in self.vehicles.values()
            ],
            "stations": [
                {
                    "station_id": station.station_id,
                    "node_id": station.node_id,
                    "piles": station.piles,
                    "charge_rate": station.charge_rate,
                    "charging_vehicle_ids": list(station.charging_vehicle_ids),
                    "queue_vehicle_ids": list(station.queue_vehicle_ids),
                    "pressure": round(station.pressure_index(), 2),
                }
                for station in self.stations.values()
            ],
            "tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "release_time": task.release_time,
                    "destination_node": task.destination_node,
                    "weight": task.weight,
                    "deadline": task.deadline,
                    "assigned_vehicle_id": task.assigned_vehicle_id,
                    "completion_time": task.completion_time,
                    "overdue_penalized": task.overdue_penalized,
                }
                for task in self.tasks.values()
            ],
            "strategy_plans": list(self.last_strategy_plans.values()),
            "simulation": {
                "failed": self.simulation_failed,
                "failure_reason": self.failure_reason,
                "failure_tick": self.failure_tick,
            },
            "events": list(self.events),
        }
        self.timeline.append(snapshot)

    def _build_replay_meta(self) -> dict:
        """构建回放 UI 使用的静态元信息。"""

        nodes = []
        for node_id, node in self.graph.nodes.items():
            node_type = "intersection"
            if node_id == self.config.depot_node:
                node_type = "warehouse"
            elif node_id in self.station_by_node:
                node_type = "station"

            nodes.append(
                {
                    "node_id": node_id,
                    "x": round(node.x, 3),
                    "y": round(node.y, 3),
                    "type": node_type,
                }
            )

        edges = []
        seen = set()
        for u, neighbors in self.graph.adjacency.items():
            for v, distance in neighbors.items():
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"u": u, "v": v, "distance": round(distance, 3)})

        return {
            "scale_name": self.scale.name,
            "strategy_name": self.strategy.name,
            "depot_node": self.config.depot_node,
            "nodes": nodes,
            "edges": edges,
            "score_breakdown": {
                "completion_score": round(self.completion_score_total, 2),
                "timeout_penalty": round(self.timeout_penalty_total, 2),
                "distance_penalty": round(self.distance_penalty_total, 2),
                "failure_penalty": round(self.failure_penalty_total, 2),
                "total_score": round(self.score, 2),
            },
        }

    def build_task_distribution(self) -> dict:
        """构建本次仿真预生成任务的分布摘要。"""

        tasks = sorted(
            (task for tick_tasks in self.precomputed_tasks_by_tick.values() for task in tick_tasks),
            key=lambda task: task.task_id,
        )
        spawn_cutoff_tick = int(self.scale.horizon * 0.9)

        release_counts = [0] * max(0, spawn_cutoff_tick)
        release_times: List[int] = []
        weights: List[float] = []
        deadline_offsets: List[int] = []

        for task in tasks:
            if 0 <= task.release_time < spawn_cutoff_tick:
                release_counts[task.release_time] += 1
            release_times.append(task.release_time)
            weights.append(task.weight)
            deadline_offsets.append(task.deadline - task.release_time)

        weight_min = min(weights) if weights else 0.0
        weight_max = max(weights) if weights else 0.0
        weight_bucket_count = min(20, max(1, int(math.sqrt(max(1, len(weights))))))
        weight_histogram = self._build_histogram(
            values=weights,
            bucket_count=weight_bucket_count,
            lower=weight_min,
            upper=weight_max,
        )

        return {
            "schema_version": 1,
            "scale_name": self.scale.name,
            "strategy_name": self.strategy.name,
            "seed": self.scale.seed,
            "horizon": self.scale.horizon,
            "spawn_cutoff_tick": spawn_cutoff_tick,
            "task_count": len(tasks),
            "time_distribution": {
                "type": self.scale.task_time_distribution,
                "mean_ratio": self.scale.task_time_mean_ratio,
                "std_ratio": self.scale.task_time_std_ratio,
            },
            "weight_distribution": {
                "target_mean": self.scale.task_weight_mean,
                "range": list(self.config.task_weight_range),
                "effective_range": [
                    round(min(weights), 4) if weights else 0.0,
                    round(max(weights), 4) if weights else 0.0,
                ],
            },
            "deadline_range": list(self.config.deadline_range),
            "stats": {
                "release_time_min": min(release_times) if release_times else None,
                "release_time_max": max(release_times) if release_times else None,
                "release_time_mean": round(sum(release_times) / len(release_times), 4) if release_times else None,
                "weight_min": round(weight_min, 4) if weights else None,
                "weight_max": round(weight_max, 4) if weights else None,
                "weight_mean": round(sum(weights) / len(weights), 4) if weights else None,
                "deadline_offset_min": min(deadline_offsets) if deadline_offsets else None,
                "deadline_offset_max": max(deadline_offsets) if deadline_offsets else None,
                "deadline_offset_mean": round(sum(deadline_offsets) / len(deadline_offsets), 4) if deadline_offsets else None,
            },
            "release_time_counts": release_counts,
            "release_times": release_times,
            "weights": weights,
            "deadline_offsets": deadline_offsets,
            "weight_histogram": weight_histogram,
        }

    def _build_histogram(
        self,
        values: List[float],
        bucket_count: int,
        lower: float,
        upper: float,
    ) -> List[dict]:
        """构建简单直方图桶。"""

        if not values or bucket_count <= 0:
            return []

        if upper <= lower:
            return [
                {
                    "start": round(lower, 4),
                    "end": round(upper, 4),
                    "count": len(values),
                }
            ]

        width = (upper - lower) / bucket_count
        counts = [0] * bucket_count
        for value in values:
            if value >= upper:
                index = bucket_count - 1
            else:
                index = min(bucket_count - 1, max(0, int((value - lower) / width)))
            counts[index] += 1

        histogram: List[dict] = []
        for index, count in enumerate(counts):
            start = lower + index * width
            end = upper if index == bucket_count - 1 else lower + (index + 1) * width
            histogram.append(
                {
                    "start": round(start, 4),
                    "end": round(end, 4),
                    "count": count,
                }
            )
        return histogram















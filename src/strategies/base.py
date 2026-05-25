from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import SimulationConfig
from ..graph_utils import RoadGraph, ShortestPathOracle
from ..models import ChargingStation, Task, Vehicle, VehicleState


@dataclass
class VehiclePlan:
    """单车计划输出（按约定仅保留四个字段）。"""

    vehicle_id: int
    task_id: List[int] = field(default_factory=list)
    action: List[str] = field(default_factory=list)
    planned_path: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "vehicle_id": self.vehicle_id,
            "task_id": list(self.task_id),
            "action": list(self.action),
            "planned_path": list(self.planned_path),
        }


@dataclass
class StrategyContext:
    """策略输入上下文：包含调度决策所需的完整世界信息。"""

    tick: int
    depot_node: int
    config: SimulationConfig
    graph: RoadGraph
    oracle: ShortestPathOracle
    vehicles: Dict[int, Vehicle]
    tasks: Dict[int, Task]
    stations: Dict[int, ChargingStation]


class SchedulingStrategy(ABC):
    """
    调度策略基类。

    设计边界：
    - 基类只负责“打包子类输出 + 合法性校验”。
    - 任务-车辆匹配、充电策略、任务链编排全部在子类实现。
    """

    name: str
    _allowed_actions = {"keep", "load", "unload", "charge"}

    def plan(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        """统一入口：调用子类计划器并补齐缺省车辆计划。"""

        raw_plans = self.build_plans(context)
        packed: Dict[int, VehiclePlan] = {}

        # 对每辆车都产出计划：子类未返回时补空计划。
        for vehicle_id in sorted(context.vehicles):
            plan = raw_plans.get(vehicle_id, VehiclePlan(vehicle_id=vehicle_id))

            # 防御式修正：若子类返回的 vehicle_id 不一致，统一改为当前键值。
            if plan.vehicle_id != vehicle_id:
                plan = VehiclePlan(
                    vehicle_id=vehicle_id,
                    task_id=list(plan.task_id),
                    action=list(plan.action),
                    planned_path=list(plan.planned_path),
                )

            self._validate_plan(plan)
            packed[vehicle_id] = plan

        return packed

    @abstractmethod
    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        """子类实现：输入全局状态，输出全车计划。"""
        raise NotImplementedError

    def _validate_plan(self, plan: VehiclePlan) -> None:
        """校验计划结构与动作合法性。"""

        if len(plan.action) != len(plan.planned_path):
            raise ValueError(
                f"VehiclePlan 不合法：vehicle_id={plan.vehicle_id} 的 action 与 planned_path 长度不一致"
            )

        for action in plan.action:
            if action not in self._allowed_actions:
                raise ValueError(
                    f"VehiclePlan 不合法：vehicle_id={plan.vehicle_id} 含未知动作 '{action}'"
                )

    def _can_interrupt_charge_for_plan(
        self,
        vehicle: Vehicle,
        planned_path: List[int],
        action: List[str],
        context: StrategyContext,
    ) -> bool:
        """仅当充电态车辆已满足首次发车安全条件时，才允许中断充电执行计划。"""

        if vehicle.state not in (VehicleState.CHARGING, VehicleState.WAITING_CHARGE):
            return True

        battery = vehicle.battery
        cursor = vehicle.current_node
        reserve_energy = float(getattr(self, "reserve_energy", 2.0))

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

            energy_to_depot = (
                context.oracle.shortest_distance(next_node, context.depot_node)
                * vehicle.energy_per_distance
            )
            energy_to_station = min(
                context.oracle.shortest_distance(next_node, station.node_id)
                * vehicle.energy_per_distance
                for station in context.stations.values()
            )
            required = energy_for_edge + min(energy_to_depot, energy_to_station) + reserve_energy
            return battery + 1e-6 >= required

        return True

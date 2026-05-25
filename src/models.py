from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


class VehicleState(str, Enum):
    """车辆状态枚举。"""

    IDLE = "idle"
    MOVING = "moving"
    LOADING = "loading"
    UNLOADING = "unloading"
    WAITING_CHARGE = "waiting_charge"
    CHARGING = "charging"


class TaskStatus(str, Enum):
    """任务状态枚举。"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Node:
    """路网节点。"""

    node_id: int
    x: float
    y: float


@dataclass
class Task:
    """配送任务：从中心仓库出发，送达目的节点。"""

    task_id: int
    release_time: int
    destination_node: int
    weight: float
    deadline: int
    status: TaskStatus = TaskStatus.PENDING
    assigned_vehicle_id: Optional[int] = None
    completion_time: Optional[int] = None
    overdue_penalized: bool = False


@dataclass
class Vehicle:
    """车辆实体，包含路径和动作状态。"""

    vehicle_id: int
    current_node: int
    battery_capacity: float
    load_capacity: float
    speed: float
    energy_per_distance: float
    battery: float
    state: VehicleState = VehicleState.IDLE
    carried_weight: float = 0.0
    assigned_task_id: Optional[int] = None
    loaded_task_ids: Deque[int] = field(default_factory=deque)
    route: Deque[int] = field(default_factory=deque)
    planned_arrivals: List[Tuple[int, int]] = field(default_factory=list)
    planned_actions: Deque[str] = field(default_factory=deque)
    planned_task_ids: Deque[int] = field(default_factory=deque)
    route_final_action: Optional[str] = None
    next_node: Optional[int] = None
    edge_remaining: float = 0.0
    operation_timer: int = 0
    distance_travelled: float = 0.0
    visiting_station_id: Optional[int] = None


@dataclass
class ChargingStation:
    """充电站实体，管理充电桩和排队车辆。"""

    station_id: int
    node_id: int
    piles: int
    charge_rate: float
    charging_vehicle_ids: List[int] = field(default_factory=list)
    queue_vehicle_ids: Deque[int] = field(default_factory=deque)

    def pressure_index(self) -> float:
        """返回站点负载压力，值越大代表越拥挤。"""

        occupied = len(self.charging_vehicle_ids) + len(self.queue_vehicle_ids)
        return occupied / max(1, self.piles)


@dataclass
class SimulationResult:
    """一次仿真运行的汇总结果。"""

    scale_name: str
    strategy_name: str
    total_score: float
    completed_tasks: int
    total_tasks: int
    overdue_tasks: int
    timeout_rate: float
    total_distance: float
    timeline: List[Dict]
    replay_meta: Dict
    simulation_failed: bool = False
    failure_reason: Optional[str] = None
    failure_tick: Optional[int] = None

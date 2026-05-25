from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models import TaskStatus
from .base import StrategyContext, VehiclePlan
from .time_first_bundle import TimeFirstBundleStrategy


@dataclass
class _WindowStats:
    completed: int
    overdue_penalized: int
    total_distance: float


class HyperSelectorStrategy(TimeFirstBundleStrategy):
    """
    基于 Q-learning 的内部超启发式（内嵌在 TimeFirstBundle 骨架内）。

    设计要点：
    - 不再在“策略级”切换 nearest/max_weight/tfb。
    - 改为在 TFB 内部切换 dispatch 模式与 charging 模式。
    - 每个窗口根据窗口奖励更新两张 Q 表。
    """

    name = "hyper_selector"

    def __init__(self) -> None:
        super().__init__()

        # Q-learning 超参数
        self.window_ticks = 20
        self.epsilon = 0.12
        self.alpha = 0.25
        self.gamma = 0.85
        self._rng = random.Random(20260504)

        # 窗口奖励权重
        self.reward_completed = 1.0
        self.reward_overdue = 3.0
        self.reward_distance = 0.02

        # 内部动作空间：任务派发子层 + 充电子层
        self._dispatch_actions = [
            "dispatch_balanced",
            "dispatch_urgent_first",
            "dispatch_bundle_first",
            "dispatch_energy_safe",
        ]
        self._charge_actions = [
            "charge_balanced",
            "charge_proactive",
            "charge_task_first",
        ]

        # 双 Q 表：state -> action_values
        self._q_dispatch: Dict[Tuple[int, ...], List[float]] = {}
        self._q_charge: Dict[Tuple[int, ...], List[float]] = {}

        # 当前窗口状态
        self._window_index = 0
        self._next_switch_tick: Optional[int] = None
        self._window_start_tick: Optional[int] = None
        self._window_start_stats: Optional[_WindowStats] = None
        self._active_state: Optional[Tuple[int, ...]] = None
        self._active_dispatch_idx: Optional[int] = None
        self._active_charge_idx: Optional[int] = None
        self._trace_path = Path("outputs") / "hyper_selector_trace.csv"
        self._trace_header_written = False
        self._run_tag = f"{self._rng.randrange(10**7, 10**8)}"

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        self._ensure_window_policy(context)
        self._apply_active_modes()
        return super().build_plans(context)

    def _ensure_window_policy(self, context: StrategyContext) -> None:
        if self._active_state is None:
            self._start_new_window(context)
            return

        assert self._next_switch_tick is not None
        if context.tick < self._next_switch_tick:
            return

        # 窗口结束：更新 Q 表
        end_stats = self._collect_stats(context)
        next_state = self._build_state(context)
        assert self._window_start_stats is not None
        assert self._active_state is not None
        assert self._active_dispatch_idx is not None
        assert self._active_charge_idx is not None
        reward = self._compute_window_reward(self._window_start_stats, end_stats)
        q_dispatch_before = self._q_dispatch.setdefault(
            self._active_state, [0.0] * len(self._dispatch_actions)
        )[self._active_dispatch_idx]
        q_charge_before = self._q_charge.setdefault(
            self._active_state, [0.0] * len(self._charge_actions)
        )[self._active_charge_idx]
        self._q_update(
            q_table=self._q_dispatch,
            state=self._active_state,
            action_idx=self._active_dispatch_idx,
            reward=reward,
            next_state=next_state,
            action_size=len(self._dispatch_actions),
        )
        self._q_update(
            q_table=self._q_charge,
            state=self._active_state,
            action_idx=self._active_charge_idx,
            reward=reward,
            next_state=next_state,
            action_size=len(self._charge_actions),
        )
        q_dispatch_after = self._q_dispatch[self._active_state][self._active_dispatch_idx]
        q_charge_after = self._q_charge[self._active_state][self._active_charge_idx]
        self._append_trace_row(
            tick=context.tick,
            state=self._active_state,
            reward=reward,
            dispatch_idx=self._active_dispatch_idx,
            charge_idx=self._active_charge_idx,
            q_dispatch_before=q_dispatch_before,
            q_dispatch_after=q_dispatch_after,
            q_charge_before=q_charge_before,
            q_charge_after=q_charge_after,
            completed_delta=end_stats.completed - self._window_start_stats.completed,
            overdue_delta=end_stats.overdue_penalized - self._window_start_stats.overdue_penalized,
            distance_delta=end_stats.total_distance - self._window_start_stats.total_distance,
            vehicle_count=len(context.vehicles),
        )

        print(
            "[hyper_selector] "
            f"window={self._window_index} "
            f"tick={context.tick} "
            f"dispatch={self._dispatch_actions[self._active_dispatch_idx]} "
            f"charge={self._charge_actions[self._active_charge_idx]} "
            f"reward={reward:.3f} "
            f"state={self._active_state} "
            f"q_dispatch={self._q_dispatch[self._active_state][self._active_dispatch_idx]:.3f} "
            f"q_charge={self._q_charge[self._active_state][self._active_charge_idx]:.3f}"
        )
        self._start_new_window(context)

    def _start_new_window(self, context: StrategyContext) -> None:
        state = self._build_state(context)
        dispatch_idx = self._select_action(
            q_table=self._q_dispatch,
            state=state,
            action_size=len(self._dispatch_actions),
        )
        charge_idx = self._select_action(
            q_table=self._q_charge,
            state=state,
            action_size=len(self._charge_actions),
        )

        self._active_state = state
        self._active_dispatch_idx = dispatch_idx
        self._active_charge_idx = charge_idx
        self._window_start_tick = context.tick
        self._next_switch_tick = context.tick + self.window_ticks
        self._window_start_stats = self._collect_stats(context)
        self._window_index += 1

        print(
            "[hyper_selector] "
            f"window={self._window_index} "
            f"start_tick={context.tick} "
            f"end_tick={self._next_switch_tick} "
            f"state={state} "
            f"dispatch={self._dispatch_actions[dispatch_idx]} "
            f"charge={self._charge_actions[charge_idx]}"
        )

    def _select_action(
        self,
        q_table: Dict[Tuple[int, ...], List[float]],
        state: Tuple[int, ...],
        action_size: int,
    ) -> int:
        values = q_table.setdefault(state, [0.0] * action_size)
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(action_size)
        best = max(values)
        best_indices = [idx for idx, value in enumerate(values) if abs(value - best) <= 1e-9]
        return self._rng.choice(best_indices)

    def _q_update(
        self,
        q_table: Dict[Tuple[int, ...], List[float]],
        state: Tuple[int, ...],
        action_idx: int,
        reward: float,
        next_state: Tuple[int, ...],
        action_size: int,
    ) -> None:
        values = q_table.setdefault(state, [0.0] * action_size)
        next_values = q_table.setdefault(next_state, [0.0] * action_size)
        target = reward + self.gamma * max(next_values)
        values[action_idx] += self.alpha * (target - values[action_idx])

    def _append_trace_row(
        self,
        *,
        tick: int,
        state: Tuple[int, ...],
        reward: float,
        dispatch_idx: int,
        charge_idx: int,
        q_dispatch_before: float,
        q_dispatch_after: float,
        q_charge_before: float,
        q_charge_after: float,
        completed_delta: int,
        overdue_delta: int,
        distance_delta: float,
        vehicle_count: int,
    ) -> None:
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._trace_path.exists()
        write_header = (not file_exists) or (not self._trace_header_written)
        with self._trace_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(
                    [
                        "run_tag",
                        "window_index",
                        "tick_end",
                        "state",
                        "dispatch_action",
                        "charge_action",
                        "reward",
                        "completed_delta",
                        "overdue_delta",
                        "distance_delta",
                        "q_dispatch_before",
                        "q_dispatch_after",
                        "q_charge_before",
                        "q_charge_after",
                        "vehicle_count",
                    ]
                )
                self._trace_header_written = True
            writer.writerow(
                [
                    self._run_tag,
                    self._window_index,
                    tick,
                    "|".join(str(v) for v in state),
                    self._dispatch_actions[dispatch_idx],
                    self._charge_actions[charge_idx],
                    round(reward, 6),
                    completed_delta,
                    overdue_delta,
                    round(distance_delta, 6),
                    round(q_dispatch_before, 6),
                    round(q_dispatch_after, 6),
                    round(q_charge_before, 6),
                    round(q_charge_after, 6),
                    vehicle_count,
                ]
            )

    def _build_state(self, context: StrategyContext) -> Tuple[int, ...]:
        pending_tasks = [
            task
            for task in context.tasks.values()
            if task.status == TaskStatus.PENDING and task.release_time <= context.tick
        ]
        pending_count = len(pending_tasks)
        vehicle_count = max(1, len(context.vehicles))
        backlog_ratio = pending_count / vehicle_count

        urgent_count = sum(1 for task in pending_tasks if self._is_emergency_task(task, context))
        urgent_ratio = urgent_count / max(1, pending_count)
        avg_battery = sum(self._battery_ratio(vehicle) for vehicle in context.vehicles.values()) / vehicle_count
        avg_station_pressure = (
            sum(station.pressure_index() for station in context.stations.values()) / max(1, len(context.stations))
        )
        idle_ratio = (
            sum(1 for vehicle in context.vehicles.values() if vehicle.state.value == "idle")
            / vehicle_count
        )
        phase = context.tick / max(1, context.config.__dict__.get("horizon", 1300))

        backlog_bin = min(4, int(backlog_ratio))
        urgent_bin = min(4, int(urgent_ratio * 5.0))
        battery_bin = min(4, int(avg_battery * 5.0))
        pressure_bin = min(3, int(min(3.99, avg_station_pressure)))
        idle_bin = min(4, int(idle_ratio * 5.0))
        phase_bin = min(2, int(phase * 3.0))
        return backlog_bin, urgent_bin, battery_bin, pressure_bin, idle_bin, phase_bin

    def _apply_active_modes(self) -> None:
        assert self._active_dispatch_idx is not None
        assert self._active_charge_idx is not None

        # 先恢复默认，再按动作覆写
        self._set_balanced_defaults()
        self._apply_dispatch_mode(self._active_dispatch_idx)
        self._apply_charge_mode(self._active_charge_idx)

    def _set_balanced_defaults(self) -> None:
        self.emergency_margin_ticks = 12
        self.max_detour_ratio = 0.25
        self.w_travel = 1.0
        self.w_queue = 1.2
        self.w_energy = 0.8
        self.cost_epsilon = 0.35
        self.keep_idle_ratio = 0.30
        self.all_low_battery_ratio = 0.30
        self.depot_charge_ratio_no_pressure = 0.90
        self.bundle_wait_escape_ticks = 2
        self.backlog_margin_gain = 3.0
        self.backlog_margin_cap = 16

        # 模式偏置：用于代价接近时倾向某个拼单模式
        self._mode_bias = {"mode1": 0.0, "mode2": 0.0, "mode3": 0.0}
        self._station_score_queue_scale = 1.0

    def _apply_dispatch_mode(self, action_idx: int) -> None:
        action = self._dispatch_actions[action_idx]
        if action == "dispatch_urgent_first":
            self.emergency_margin_ticks = 16
            self.bundle_wait_escape_ticks = 1
            self._mode_bias["mode1"] = -0.25
        elif action == "dispatch_bundle_first":
            self.emergency_margin_ticks = 9
            self.bundle_wait_escape_ticks = 4
            self.max_detour_ratio = 0.30
            self._mode_bias["mode2"] = -0.15
            self._mode_bias["mode3"] = -0.10
        elif action == "dispatch_energy_safe":
            self.emergency_margin_ticks = 11
            self.max_detour_ratio = 0.20
            self.w_energy = 1.1
            self._mode_bias["mode1"] = -0.20

    def _apply_charge_mode(self, action_idx: int) -> None:
        action = self._charge_actions[action_idx]
        if action == "charge_proactive":
            self.keep_idle_ratio = 0.22
            self.depot_charge_ratio_no_pressure = 0.95
            self.all_low_battery_ratio = 0.38
            self._station_score_queue_scale = 1.2
        elif action == "charge_task_first":
            self.keep_idle_ratio = 0.45
            self.depot_charge_ratio_no_pressure = 0.75
            self.all_low_battery_ratio = 0.24
            self._station_score_queue_scale = 0.8

    def _compute_window_reward(self, start: _WindowStats, end: _WindowStats) -> float:
        completed_delta = end.completed - start.completed
        overdue_delta = end.overdue_penalized - start.overdue_penalized
        distance_delta = end.total_distance - start.total_distance
        return (
            self.reward_completed * completed_delta
            - self.reward_overdue * overdue_delta
            - self.reward_distance * distance_delta
        )

    def _feasible_modes_for_pair(
        self,
        vehicle,
        first_task,
        second_task,
        context: StrategyContext,
    ):
        infos = super()._feasible_modes_for_pair(vehicle, first_task, second_task, context)
        for info in infos:
            info["cost"] += self._mode_bias.get(info["mode"], 0.0)
        return infos

    def _select_best_station(
        self,
        from_node: int,
        available_energy: float,
        energy_per_distance: float,
        context: StrategyContext,
    ):
        best_station = None
        best_score = float("inf")
        for station in context.stations.values():
            distance = context.oracle.shortest_distance(from_node, station.node_id)
            required = distance * energy_per_distance + self.reserve_energy
            if required > available_energy + 1e-6:
                continue

            queue_cost = (
                station.pressure_index()
                * context.config.queue_penalty_factor
                * self._station_score_queue_scale
            )
            score = distance + queue_cost
            if score < best_score:
                best_score = score
                best_station = station
        return best_station

    @staticmethod
    def _collect_stats(context: StrategyContext) -> _WindowStats:
        completed = sum(1 for task in context.tasks.values() if task.status == TaskStatus.COMPLETED)
        overdue_penalized = sum(1 for task in context.tasks.values() if task.overdue_penalized)
        total_distance = sum(vehicle.distance_travelled for vehicle in context.vehicles.values())
        return _WindowStats(
            completed=completed,
            overdue_penalized=overdue_penalized,
            total_distance=total_distance,
        )

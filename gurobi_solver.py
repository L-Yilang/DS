from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import ScaleConfig, SimulationConfig
from .models import Task
from .scoring import completion_score, distance_penalty
from .strategies.base import SchedulingStrategy, StrategyContext, VehiclePlan
from .world import WorldManager


class _NoOpStrategy(SchedulingStrategy):
    """Strategy placeholder used only to reuse WorldManager data generation."""

    name = "gurobi_data_generation"

    def build_plans(self, context: StrategyContext) -> Dict[int, VehiclePlan]:
        return {}


@dataclass(frozen=True)
class TaskData:
    task_id: int
    release_time: int
    destination_node: int
    weight: float
    deadline: int
    outbound_ticks: int
    return_ticks: int
    service_ticks: int
    roundtrip_distance: float
    roundtrip_energy: float


@dataclass(frozen=True)
class GurobiSolveOptions:
    objective_mode: str = "score"
    max_slots: Optional[int] = None
    slot_mode: str = "full"
    slot_buffer: int = 8
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    log_to_console: bool = True
    log_file: Optional[Path] = None
    distribution_path: Optional[Path] = None
    threads: Optional[int] = None


@dataclass
class GurobiRunResult:
    scale: str
    objective_mode: str
    round_index: int
    seed: int
    status: str
    status_code: int
    runtime: float
    objective_value: Optional[float]
    objective_bound: Optional[float]
    mip_gap: Optional[float]
    total_score: float
    completed_tasks: int
    total_tasks: int
    overdue_tasks: int
    completion_rate: float
    timeout_rate: float
    total_distance: float
    completed_weight: float
    generated_weight: float
    max_slots: int
    vehicles: List[dict]
    tasks: List[dict]
    model_notes: List[str]
    replay_meta: Optional[dict] = None
    timeline: Optional[List[dict]] = None

    def summary_row(self) -> dict:
        return {
            "scale": self.scale,
            "objective_mode": self.objective_mode,
            "round": self.round_index,
            "seed": self.seed,
            "status": self.status,
            "runtime": round(self.runtime, 4),
            "objective_value": self.objective_value,
            "objective_bound": self.objective_bound,
            "mip_gap": self.mip_gap,
            "total_score": self.total_score,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "overdue_tasks": self.overdue_tasks,
            "completion_rate": self.completion_rate,
            "timeout_rate": self.timeout_rate,
            "total_distance": self.total_distance,
            "completed_weight": self.completed_weight,
            "generated_weight": self.generated_weight,
            "max_slots": self.max_slots,
        }


def build_round_seeds(scale: ScaleConfig, rounds: int) -> List[int]:
    """Deterministic multi-run seeds, keeping the first run equal to scale.seed."""

    if rounds <= 1:
        return [scale.seed]

    rng = random.Random(scale.seed)
    seeds = [scale.seed]
    seen = {scale.seed}
    while len(seeds) < rounds:
        candidate = rng.randrange(1, 2**31 - 1)
        if candidate in seen:
            continue
        seen.add(candidate)
        seeds.append(candidate)
    return seeds


def build_world_data(
    scale: ScaleConfig,
    config: SimulationConfig,
) -> tuple[WorldManager, List[Task]]:
    """Reuse the existing generator to get graph, vehicles, stations and all tasks."""

    world = WorldManager(scale=scale, config=config, strategy=_NoOpStrategy())
    tasks = sorted(
        (task for tick_tasks in world.precomputed_tasks_by_tick.values() for task in tick_tasks),
        key=lambda task: task.task_id,
    )
    return world, tasks


def load_distribution_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scale_from_distribution(scale: ScaleConfig, distribution_path: Path) -> ScaleConfig:
    payload = load_distribution_payload(distribution_path)
    return ScaleConfig(
        name=str(payload.get("scale_name", scale.name)),
        node_count=scale.node_count,
        vehicle_count=scale.vehicle_count,
        station_count=scale.station_count,
        horizon=int(payload.get("horizon", scale.horizon)),
        task_count=int(payload.get("task_count", scale.task_count)),
        seed=int(payload.get("seed", scale.seed)),
        task_time_distribution=scale.task_time_distribution,
        task_time_mean_ratio=scale.task_time_mean_ratio,
        task_time_std_ratio=scale.task_time_std_ratio,
        task_weight_mean=scale.task_weight_mean,
        depot_station_piles=scale.depot_station_piles,
        non_depot_station_piles_range=scale.non_depot_station_piles_range,
    )


def apply_distribution_to_tasks(raw_tasks: Sequence[Task], distribution_path: Path) -> List[Task]:
    """Use stored release/weight/deadline arrays while keeping regenerated destinations."""

    payload = load_distribution_payload(distribution_path)
    release_times = list(payload.get("release_times", []))
    weights = list(payload.get("weights", []))
    deadline_offsets = list(payload.get("deadline_offsets", []))
    if not (len(release_times) == len(weights) == len(deadline_offsets) == len(raw_tasks)):
        raise ValueError(
            "distribution 文件字段长度与任务数量不一致："
            f"release={len(release_times)}, weight={len(weights)}, "
            f"deadline_offset={len(deadline_offsets)}, generated={len(raw_tasks)}"
        )

    patched: List[Task] = []
    for task, release_time, weight, deadline_offset in zip(
        sorted(raw_tasks, key=lambda item: item.task_id),
        release_times,
        weights,
        deadline_offsets,
    ):
        patched.append(
            Task(
                task_id=task.task_id,
                release_time=int(release_time),
                destination_node=task.destination_node,
                weight=float(weight),
                deadline=int(release_time) + int(deadline_offset),
            )
        )
    return patched


def solve_static_gurobi(
    *,
    scale: ScaleConfig,
    config: SimulationConfig,
    objective_mode: str,
    round_index: int,
    options: GurobiSolveOptions,
) -> GurobiRunResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "gurobipy is not installed in the active Python environment. "
            "Run with: conda run -n DS python run_gurobi.py ..."
        ) from exc

    world, raw_tasks = build_world_data(scale, config)
    if options.distribution_path is not None:
        raw_tasks = apply_distribution_to_tasks(raw_tasks, options.distribution_path)
    vehicles = list(sorted(world.vehicles.values(), key=lambda vehicle: vehicle.vehicle_id))
    depot = config.depot_node
    depot_station = world.station_by_node[depot]

    tasks = _build_task_data(raw_tasks, world, config)
    task_ids = [task.task_id for task in tasks]
    task_by_id = {task.task_id: task for task in tasks}
    vehicle_ids = [vehicle.vehicle_id for vehicle in vehicles]
    vehicle_by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    slots = list(range(_resolve_max_slots(tasks, len(vehicles), scale.horizon, options)))

    max_task_cycle = max(
        (
            task.service_ticks
            + task.outbound_ticks
            + task.return_ticks
            + math.ceil(config.vehicle_battery_capacity / max(0.1, depot_station.charge_rate))
            for task in tasks
        ),
        default=1,
    )
    max_time = scale.horizon + max(1, len(slots)) * max_task_cycle + 10
    big_m_time = float(max_time)
    big_m_battery = float(config.vehicle_battery_capacity * 2.0 + 100.0)

    model = gp.Model(f"static_ev_dispatch_{scale.name}_{objective_mode}_r{round_index}")
    model.Params.LogToConsole = 1 if options.log_to_console else 0
    if options.time_limit is not None:
        model.Params.TimeLimit = float(options.time_limit)
    if options.mip_gap is not None:
        model.Params.MIPGap = float(options.mip_gap)
    if options.threads is not None:
        model.Params.Threads = int(options.threads)
    if options.log_file is not None:
        options.log_file.parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(options.log_file)

    x = model.addVars(vehicle_ids, slots, task_ids, vtype=GRB.BINARY, name="assign")
    used = model.addVars(vehicle_ids, slots, vtype=GRB.BINARY, name="slot_used")
    start = model.addVars(vehicle_ids, slots, lb=0, ub=max_time, vtype=GRB.INTEGER, name="load_start")
    complete = model.addVars(vehicle_ids, slots, lb=0, ub=max_time, vtype=GRB.INTEGER, name="complete")
    back = model.addVars(vehicle_ids, slots, lb=0, ub=max_time, vtype=GRB.INTEGER, name="back_depot")
    charge = model.addVars(vehicle_ids, slots, lb=0, ub=max_time, vtype=GRB.INTEGER, name="depot_charge_ticks")
    battery_depart = model.addVars(
        vehicle_ids,
        slots,
        lb=0.0,
        ub=config.vehicle_battery_capacity,
        vtype=GRB.CONTINUOUS,
        name="battery_depart",
    )
    battery_back = model.addVars(
        vehicle_ids,
        slots,
        lb=0.0,
        ub=config.vehicle_battery_capacity,
        vtype=GRB.CONTINUOUS,
        name="battery_back",
    )

    y = model.addVars(task_ids, vtype=GRB.BINARY, name="task_completed")
    on_time = model.addVars(task_ids, vtype=GRB.BINARY, name="task_on_time")
    late = model.addVars(task_ids, vtype=GRB.BINARY, name="task_late")
    overdue = model.addVars(task_ids, vtype=GRB.BINARY, name="task_overdue")
    task_completion = model.addVars(task_ids, lb=0, ub=scale.horizon, vtype=GRB.INTEGER, name="task_completion")
    reward = model.addVars(task_ids, lb=-1e5, ub=1e5, vtype=GRB.CONTINUOUS, name="task_reward")

    for vehicle_id in vehicle_ids:
        for slot in slots:
            model.addConstr(
                gp.quicksum(x[vehicle_id, slot, task_id] for task_id in task_ids)
                == used[vehicle_id, slot],
                name=f"one_task_v{vehicle_id}_p{slot}",
            )
            if slot + 1 in slots:
                model.addConstr(
                    used[vehicle_id, slot + 1] <= used[vehicle_id, slot],
                    name=f"left_justified_v{vehicle_id}_p{slot}",
                )

            duration_expr = gp.quicksum(
                (task_by_id[task_id].service_ticks + task_by_id[task_id].outbound_ticks)
                * x[vehicle_id, slot, task_id]
                for task_id in task_ids
            )
            return_expr = gp.quicksum(
                task_by_id[task_id].return_ticks * x[vehicle_id, slot, task_id]
                for task_id in task_ids
            )
            energy_expr = gp.quicksum(
                task_by_id[task_id].roundtrip_energy * x[vehicle_id, slot, task_id]
                for task_id in task_ids
            )

            model.addConstr(
                complete[vehicle_id, slot]
                >= start[vehicle_id, slot] + duration_expr - big_m_time * (1 - used[vehicle_id, slot]),
                name=f"complete_lb_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                complete[vehicle_id, slot]
                <= start[vehicle_id, slot] + duration_expr + big_m_time * (1 - used[vehicle_id, slot]),
                name=f"complete_ub_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                complete[vehicle_id, slot] <= scale.horizon + big_m_time * (1 - used[vehicle_id, slot]),
                name=f"complete_inside_horizon_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                back[vehicle_id, slot]
                >= complete[vehicle_id, slot] + return_expr - big_m_time * (1 - used[vehicle_id, slot]),
                name=f"back_lb_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                back[vehicle_id, slot]
                <= complete[vehicle_id, slot] + return_expr + big_m_time * (1 - used[vehicle_id, slot]),
                name=f"back_ub_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                battery_back[vehicle_id, slot]
                >= battery_depart[vehicle_id, slot] - energy_expr - big_m_battery * (1 - used[vehicle_id, slot]),
                name=f"battery_back_lb_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                battery_back[vehicle_id, slot]
                <= battery_depart[vehicle_id, slot] - energy_expr + big_m_battery * (1 - used[vehicle_id, slot]),
                name=f"battery_back_ub_v{vehicle_id}_p{slot}",
            )
            model.addConstr(
                charge[vehicle_id, slot] <= max_time * used[vehicle_id, slot],
                name=f"charge_only_when_used_v{vehicle_id}_p{slot}",
            )

            if slot == 0:
                model.addConstr(start[vehicle_id, slot] >= 0, name=f"first_start_v{vehicle_id}")
                model.addConstr(
                    battery_depart[vehicle_id, slot]
                    >= config.vehicle_battery_capacity - big_m_battery * (1 - used[vehicle_id, slot]),
                    name=f"first_battery_full_v{vehicle_id}",
                )
                model.addConstr(charge[vehicle_id, slot] == 0, name=f"no_initial_charge_v{vehicle_id}")
            else:
                model.addConstr(
                    start[vehicle_id, slot]
                    >= back[vehicle_id, slot - 1] + charge[vehicle_id, slot]
                    - big_m_time * (1 - used[vehicle_id, slot]),
                    name=f"start_after_back_charge_v{vehicle_id}_p{slot}",
                )
                model.addConstr(
                    battery_depart[vehicle_id, slot]
                    <= battery_back[vehicle_id, slot - 1]
                    + depot_station.charge_rate * charge[vehicle_id, slot]
                    + big_m_battery * (1 - used[vehicle_id, slot]),
                    name=f"battery_after_depot_charge_v{vehicle_id}_p{slot}",
                )
                model.addConstr(
                    battery_depart[vehicle_id, slot]
                    >= battery_back[vehicle_id, slot - 1] - big_m_battery * (1 - used[vehicle_id, slot]),
                    name=f"battery_no_loss_at_depot_v{vehicle_id}_p{slot}",
                )

            for task_id in task_ids:
                task = task_by_id[task_id]
                vehicle = vehicle_by_id[vehicle_id]
                if task.weight > vehicle.load_capacity + 1e-6:
                    x[vehicle_id, slot, task_id].UB = 0.0
                if task.roundtrip_energy + 2.0 > vehicle.battery_capacity + 1e-6:
                    x[vehicle_id, slot, task_id].UB = 0.0

                model.addConstr(
                    start[vehicle_id, slot]
                    >= task.release_time - big_m_time * (1 - x[vehicle_id, slot, task_id]),
                    name=f"release_v{vehicle_id}_p{slot}_t{task_id}",
                )
                model.addConstr(
                    battery_depart[vehicle_id, slot]
                    >= task.roundtrip_energy + 2.0 - big_m_battery * (1 - x[vehicle_id, slot, task_id]),
                    name=f"safe_energy_v{vehicle_id}_p{slot}_t{task_id}",
                )
                model.addConstr(
                    task_completion[task_id]
                    >= complete[vehicle_id, slot] - big_m_time * (1 - x[vehicle_id, slot, task_id]),
                    name=f"task_completion_lb_v{vehicle_id}_p{slot}_t{task_id}",
                )
                model.addConstr(
                    task_completion[task_id]
                    <= complete[vehicle_id, slot] + big_m_time * (1 - x[vehicle_id, slot, task_id]),
                    name=f"task_completion_ub_v{vehicle_id}_p{slot}_t{task_id}",
                )

    for previous_vehicle, next_vehicle in zip(vehicle_ids, vehicle_ids[1:]):
        model.addConstr(
            gp.quicksum(used[previous_vehicle, slot] for slot in slots)
            >= gp.quicksum(used[next_vehicle, slot] for slot in slots),
            name=f"vehicle_count_symmetry_v{previous_vehicle}_v{next_vehicle}",
        )

    for task_id in task_ids:
        task = task_by_id[task_id]
        assigned_expr = gp.quicksum(
            x[vehicle_id, slot, task_id]
            for vehicle_id in vehicle_ids
            for slot in slots
        )
        model.addConstr(y[task_id] == assigned_expr, name=f"task_once_t{task_id}")
        model.addConstr(on_time[task_id] + late[task_id] == y[task_id], name=f"task_time_class_t{task_id}")
        model.addConstr(task_completion[task_id] <= scale.horizon * y[task_id], name=f"completion_zero_t{task_id}")
        model.addConstr(overdue[task_id] >= late[task_id], name=f"late_is_overdue_t{task_id}")
        if task.deadline < scale.horizon:
            model.addConstr(overdue[task_id] >= 1 - y[task_id], name=f"incomplete_overdue_t{task_id}")

        model.addGenConstrIndicator(
            on_time[task_id],
            True,
            task_completion[task_id] <= task.deadline,
            name=f"on_time_deadline_t{task_id}",
        )
        model.addGenConstrIndicator(
            late[task_id],
            True,
            task_completion[task_id] >= task.deadline + 1,
            name=f"late_deadline_t{task_id}",
        )

        allowed = max(1, task.deadline - task.release_time)
        on_expr = config.completion_base_reward * (
            1.0 - 0.6 * (task_completion[task_id] - task.release_time) / allowed
        )
        late_expr = -config.late_completion_penalty_factor * (
            0.5 + (task_completion[task_id] - task.deadline) / allowed
        )
        model.addGenConstrIndicator(y[task_id], False, reward[task_id] == 0.0, name=f"reward_zero_t{task_id}")
        model.addGenConstrIndicator(on_time[task_id], True, reward[task_id] == on_expr, name=f"reward_on_t{task_id}")
        model.addGenConstrIndicator(late[task_id], True, reward[task_id] == late_expr, name=f"reward_late_t{task_id}")

    total_distance_expr = gp.quicksum(
        task_by_id[task_id].roundtrip_distance * x[vehicle_id, slot, task_id]
        for vehicle_id in vehicle_ids
        for slot in slots
        for task_id in task_ids
    )
    total_score_expr = (
        gp.quicksum(reward[task_id] for task_id in task_ids)
        - config.timeout_penalty * gp.quicksum(overdue[task_id] for task_id in task_ids)
        + distance_penalty(total_distance_expr, config.distance_penalty_factor)
    )

    model.ModelSense = GRB.MAXIMIZE
    if objective_mode == "score":
        model.setObjective(total_score_expr)
    elif objective_mode == "multi":
        model.setObjectiveN(gp.quicksum(y[task_id] for task_id in task_ids), index=0, priority=3, name="max_completed")
        model.setObjectiveN(-gp.quicksum(overdue[task_id] for task_id in task_ids), index=1, priority=2, name="min_overdue")
        model.setObjectiveN(total_score_expr, index=2, priority=1, name="max_score")
    else:
        raise ValueError(f"Unknown objective mode: {objective_mode}")

    started_at = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - started_at

    status = _status_name(model.Status, GRB)
    has_solution = model.SolCount > 0
    if not has_solution:
        return GurobiRunResult(
            scale=scale.name,
            objective_mode=objective_mode,
            round_index=round_index,
            seed=scale.seed,
            status=status,
            status_code=int(model.Status),
            runtime=runtime,
            objective_value=None,
            objective_bound=_safe_attr(model, "ObjBound"),
            mip_gap=_safe_attr(model, "MIPGap"),
            total_score=0.0,
            completed_tasks=0,
            total_tasks=len(tasks),
            overdue_tasks=0,
            completion_rate=0.0,
            timeout_rate=0.0,
            total_distance=0.0,
            completed_weight=0.0,
            generated_weight=round(sum(task.weight for task in tasks), 4),
            max_slots=len(slots),
            vehicles=[],
            tasks=[],
            model_notes=_model_notes(depot_station.piles, len(vehicles), options.distribution_path),
            replay_meta=world._build_replay_meta(),
            timeline=[],
        )

    solution_vehicles, solution_tasks = _extract_solution(
        vehicles=vehicles,
        tasks=tasks,
        slots=slots,
        x=x,
        y=y,
        overdue=overdue,
        task_completion=task_completion,
        start=start,
        complete=complete,
        back=back,
        charge=charge,
        battery_depart=battery_depart,
        battery_back=battery_back,
    )
    metrics = _compute_metrics(solution_tasks, tasks, config, scale.horizon)
    replay_meta = _build_gurobi_replay_meta(world, metrics)
    timeline = _build_gurobi_timeline(
        horizon=scale.horizon,
        vehicles=vehicles,
        stations=world.stations.values(),
        tasks=solution_tasks,
        solution_vehicles=solution_vehicles,
        score=metrics["total_score"],
    )

    return GurobiRunResult(
        scale=scale.name,
        objective_mode=objective_mode,
        round_index=round_index,
        seed=scale.seed,
        status=status,
        status_code=int(model.Status),
        runtime=runtime,
        objective_value=_safe_attr(model, "ObjVal"),
        objective_bound=_safe_attr(model, "ObjBound"),
        mip_gap=_safe_attr(model, "MIPGap"),
        total_score=metrics["total_score"],
        completed_tasks=metrics["completed_tasks"],
        total_tasks=len(tasks),
        overdue_tasks=metrics["overdue_tasks"],
        completion_rate=metrics["completion_rate"],
        timeout_rate=metrics["timeout_rate"],
        total_distance=metrics["total_distance"],
        completed_weight=metrics["completed_weight"],
        generated_weight=metrics["generated_weight"],
        max_slots=len(slots),
        vehicles=solution_vehicles,
        tasks=solution_tasks,
        model_notes=_model_notes(depot_station.piles, len(vehicles), options.distribution_path),
        replay_meta=replay_meta,
        timeline=timeline,
    )


def write_solution_json(path: Path, result: GurobiRunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": result.summary_row(),
        "vehicles": result.vehicles,
        "tasks": result.tasks,
        "model_notes": result.model_notes,
        "replay_meta": result.replay_meta,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_gurobi_replay_json(path: Path, result: GurobiRunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "replay_meta": result.replay_meta or {},
        "timeline": result.timeline or [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_result_to_terminal(result: GurobiRunResult) -> None:
    print(
        "[gurobi-result] "
        f"scale={result.scale} objective={result.objective_mode} "
        f"round={result.round_index} seed={result.seed}",
        flush=True,
    )
    print(
        "[gurobi-result] "
        f"status={result.status} runtime={result.runtime:.2f}s "
        f"obj={result.objective_value} bound={result.objective_bound} gap={result.mip_gap}",
        flush=True,
    )
    print(
        "[gurobi-result] "
        f"score={result.total_score} completed={result.completed_tasks}/{result.total_tasks} "
        f"completion_rate={result.completion_rate} overdue={result.overdue_tasks} "
        f"timeout_rate={result.timeout_rate} distance={result.total_distance} "
        f"completed_weight={result.completed_weight}",
        flush=True,
    )


def write_rows_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, results_by_mode: Dict[str, Sequence[GurobiRunResult]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["# Gurobi 全局求解结果", ""]
    for mode, results in results_by_mode.items():
        if not results:
            continue
        lines.append(f"## {mode}")
        lines.append("")
        avg = average_summary(results)
        lines.append("平均结果：")
        lines.append("")
        lines.append(
            "- "
            + ", ".join(
                [
                    f"总分={avg['total_score']}",
                    f"超时率={avg['timeout_rate']}",
                    f"完成率={avg['completion_rate']}",
                    f"总载量={avg['completed_weight']}",
                    f"总里程={avg['total_distance']}",
                ]
            )
        )
        lines.append("")
        lines.append("详细结果：")
        lines.append("")
        lines.append(
            "| round | seed | status | score | completed | overdue | completion_rate | timeout_rate | completed_weight | distance | runtime |"
        )
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for result in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(result.round_index),
                        str(result.seed),
                        result.status,
                        str(result.total_score),
                        f"{result.completed_tasks}/{result.total_tasks}",
                        str(result.overdue_tasks),
                        str(result.completion_rate),
                        str(result.timeout_rate),
                        str(result.completed_weight),
                        str(result.total_distance),
                        str(round(result.runtime, 4)),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def average_summary(results: Sequence[GurobiRunResult]) -> dict:
    if not results:
        return {}

    def avg(field: str) -> float:
        return round(sum(float(getattr(result, field)) for result in results) / len(results), 4)

    return {
        "rounds": len(results),
        "total_score": avg("total_score"),
        "completed_tasks": avg("completed_tasks"),
        "total_tasks": avg("total_tasks"),
        "overdue_tasks": avg("overdue_tasks"),
        "completion_rate": avg("completion_rate"),
        "timeout_rate": avg("timeout_rate"),
        "total_distance": avg("total_distance"),
        "completed_weight": avg("completed_weight"),
        "generated_weight": avg("generated_weight"),
        "runtime": avg("runtime"),
    }


def _build_task_data(
    raw_tasks: Sequence[Task],
    world: WorldManager,
    config: SimulationConfig,
) -> List[TaskData]:
    tasks: List[TaskData] = []
    depot = config.depot_node
    for task in raw_tasks:
        outbound_distance = world.oracle.shortest_distance(depot, task.destination_node)
        return_distance = world.oracle.shortest_distance(task.destination_node, depot)
        outbound_ticks = math.ceil(outbound_distance / max(0.1, config.vehicle_speed))
        return_ticks = math.ceil(return_distance / max(0.1, config.vehicle_speed))
        roundtrip_distance = outbound_distance + return_distance
        tasks.append(
            TaskData(
                task_id=task.task_id,
                release_time=task.release_time,
                destination_node=task.destination_node,
                weight=task.weight,
                deadline=task.deadline,
                outbound_ticks=outbound_ticks,
                return_ticks=return_ticks,
                service_ticks=config.loading_duration + config.unloading_duration,
                roundtrip_distance=roundtrip_distance,
                roundtrip_energy=roundtrip_distance * config.energy_per_distance,
            )
        )
    return tasks


def _resolve_max_slots(
    tasks: Sequence[TaskData],
    vehicle_count: int,
    horizon: int,
    options: GurobiSolveOptions,
) -> int:
    task_count = len(tasks)
    if options.max_slots is not None:
        return max(1, min(task_count, int(options.max_slots)))
    if options.slot_mode == "balanced":
        return max(1, min(task_count, math.ceil(task_count / max(1, vehicle_count)) + options.slot_buffer))
    if options.slot_mode != "full":
        raise ValueError(f"Unknown slot mode: {options.slot_mode}")
    if not tasks:
        return 1

    fastest_first = min(task.service_ticks + task.outbound_ticks for task in tasks)
    fastest_between = min(task.return_ticks + task.service_ticks + task.outbound_ticks for task in tasks)
    if horizon < fastest_first:
        return 1
    physical_cap = 1 + math.floor((horizon - fastest_first) / max(1, fastest_between))
    return max(1, min(task_count, physical_cap))


def _extract_solution(
    *,
    vehicles,
    tasks: Sequence[TaskData],
    slots: Sequence[int],
    x,
    y,
    overdue,
    task_completion,
    start,
    complete,
    back,
    charge,
    battery_depart,
    battery_back,
) -> tuple[List[dict], List[dict]]:
    task_by_id = {task.task_id: task for task in tasks}
    solution_vehicles: List[dict] = []
    completed_task_ids: set[int] = set()

    for vehicle in vehicles:
        vehicle_events: List[dict] = []
        for slot in slots:
            chosen_task_id = None
            for task in tasks:
                if x[vehicle.vehicle_id, slot, task.task_id].X > 0.5:
                    chosen_task_id = task.task_id
                    break
            if chosen_task_id is None:
                continue
            task = task_by_id[chosen_task_id]
            completed_task_ids.add(chosen_task_id)
            vehicle_events.append(
                {
                    "slot": slot,
                    "task_id": chosen_task_id,
                    "destination_node": task.destination_node,
                    "weight": task.weight,
                    "release_time": task.release_time,
                    "deadline": task.deadline,
                    "load_start": int(round(start[vehicle.vehicle_id, slot].X)),
                    "completion_time": int(round(complete[vehicle.vehicle_id, slot].X)),
                    "back_depot_time": int(round(back[vehicle.vehicle_id, slot].X)),
                    "charge_before_ticks": int(round(charge[vehicle.vehicle_id, slot].X)),
                    "battery_depart": round(battery_depart[vehicle.vehicle_id, slot].X, 4),
                    "battery_back": round(battery_back[vehicle.vehicle_id, slot].X, 4),
                    "roundtrip_distance": round(task.roundtrip_distance, 4),
                }
            )
        solution_vehicles.append(
            {
                "vehicle_id": vehicle.vehicle_id,
                "events": vehicle_events,
                "completed_tasks": len(vehicle_events),
                "distance": round(sum(event["roundtrip_distance"] for event in vehicle_events), 4),
                "completed_weight": round(sum(event["weight"] for event in vehicle_events), 4),
            }
        )

    solution_tasks: List[dict] = []
    for task in tasks:
        is_completed = y[task.task_id].X > 0.5
        completion_time = int(round(task_completion[task.task_id].X)) if is_completed else None
        solution_tasks.append(
            {
                "task_id": task.task_id,
                "completed": is_completed,
                "completion_time": completion_time,
                "overdue": overdue[task.task_id].X > 0.5,
                "release_time": task.release_time,
                "deadline": task.deadline,
                "destination_node": task.destination_node,
                "weight": task.weight,
                "roundtrip_distance": round(task.roundtrip_distance, 4),
            }
        )
    return solution_vehicles, solution_tasks


def _compute_metrics(
    solution_tasks: Sequence[dict],
    tasks: Sequence[TaskData],
    config: SimulationConfig,
    horizon: int,
) -> dict:
    task_by_id = {task.task_id: task for task in tasks}
    completed_tasks = 0
    overdue_tasks = 0
    total_distance = 0.0
    completed_weight = 0.0
    generated_weight = sum(task.weight for task in tasks)
    reward_total = 0.0

    for item in solution_tasks:
        task = task_by_id[item["task_id"]]
        if item["completed"]:
            completed_tasks += 1
            completed_weight += task.weight
            total_distance += task.roundtrip_distance
            task_obj = Task(
                task_id=task.task_id,
                release_time=task.release_time,
                destination_node=task.destination_node,
                weight=task.weight,
                deadline=task.deadline,
            )
            reward_total += completion_score(
                task_obj,
                completion_time=int(item["completion_time"]),
                base_reward=config.completion_base_reward,
                late_penalty_factor=config.late_completion_penalty_factor,
            )
        if item["overdue"]:
            overdue_tasks += 1

    total_score = (
        reward_total
        - config.timeout_penalty * overdue_tasks
        + distance_penalty(total_distance, config.distance_penalty_factor)
    )
    total_tasks = len(tasks)
    return {
        "total_score": round(total_score, 2),
        "completed_tasks": completed_tasks,
        "overdue_tasks": overdue_tasks,
        "completion_rate": round(completed_tasks / max(1, total_tasks), 4),
        "timeout_rate": round(overdue_tasks / max(1, total_tasks), 4),
        "total_distance": round(total_distance, 2),
        "completed_weight": round(completed_weight, 2),
        "generated_weight": round(generated_weight, 2),
    }


def _build_gurobi_replay_meta(world: WorldManager, metrics: dict) -> dict:
    meta = world._build_replay_meta()
    meta["strategy_name"] = "gurobi_static_optimal"
    meta["score_breakdown"] = {
        "completion_score": None,
        "timeout_penalty": None,
        "distance_penalty": None,
        "failure_penalty": 0.0,
        "total_score": metrics["total_score"],
    }
    return meta


def _build_gurobi_timeline(
    *,
    horizon: int,
    vehicles,
    stations,
    tasks: Sequence[dict],
    solution_vehicles: Sequence[dict],
    score: float,
) -> List[dict]:
    event_by_vehicle: Dict[int, List[dict]] = {
        vehicle_info["vehicle_id"]: sorted(vehicle_info["events"], key=lambda event: event["load_start"])
        for vehicle_info in solution_vehicles
    }
    task_by_id = {task["task_id"]: task for task in tasks}
    interesting_ticks = {0, max(0, horizon - 1)}
    for vehicle_info in solution_vehicles:
        for event in vehicle_info["events"]:
            interesting_ticks.update(
                [
                    max(0, event["load_start"]),
                    max(0, event["completion_time"]),
                    max(0, min(horizon - 1, event["back_depot_time"])),
                ]
            )
    ticks = sorted(tick for tick in interesting_ticks if 0 <= tick < horizon)

    timeline: List[dict] = []
    for tick in ticks:
        vehicle_snapshots = []
        completed_ids = set()
        assigned_ids = set()
        in_progress_ids = set()

        for vehicle in vehicles:
            current_event = None
            for event in event_by_vehicle.get(vehicle.vehicle_id, []):
                if event["load_start"] <= tick <= event["back_depot_time"]:
                    current_event = event
                    break
                if event["completion_time"] <= tick:
                    completed_ids.add(event["task_id"])

            state = "idle"
            current_node = vehicle.current_node
            assigned_task_id = None
            carried_weight = 0.0
            battery = vehicle.battery_capacity
            route = []

            if current_event is not None:
                assigned_task_id = current_event["task_id"]
                task = task_by_id[assigned_task_id]
                if tick < current_event["load_start"]:
                    state = "idle"
                elif tick < current_event["completion_time"]:
                    state = "moving"
                    current_node = 0
                    route = [current_event["destination_node"]]
                    carried_weight = current_event["weight"]
                    in_progress_ids.add(assigned_task_id)
                elif tick < current_event["back_depot_time"]:
                    state = "moving"
                    current_node = current_event["destination_node"]
                    route = [0]
                    completed_ids.add(assigned_task_id)
                else:
                    completed_ids.add(assigned_task_id)
                battery = max(0.0, current_event["battery_back"])
                if task["completed"] and task["completion_time"] is not None and task["completion_time"] > tick:
                    assigned_ids.add(assigned_task_id)

            vehicle_snapshots.append(
                {
                    "vehicle_id": vehicle.vehicle_id,
                    "state": state,
                    "current_node": current_node,
                    "battery": round(battery, 2),
                    "battery_capacity": vehicle.battery_capacity,
                    "load_capacity": vehicle.load_capacity,
                    "assigned_task_id": assigned_task_id,
                    "carried_weight": carried_weight,
                    "distance_travelled": 0.0,
                    "route": route,
                    "planned_arrivals": [],
                    "planned_actions": [],
                    "planned_task_ids": [assigned_task_id] if assigned_task_id is not None else [],
                    "next_node": route[0] if route else None,
                    "edge_remaining": 0.0,
                }
            )

        task_snapshots = []
        for task in tasks:
            status = "pending"
            if tick < task["release_time"]:
                status = "pending"
            elif task["task_id"] in completed_ids or (
                task["completed"] and task["completion_time"] is not None and tick >= task["completion_time"]
            ):
                status = "completed"
            elif task["task_id"] in in_progress_ids:
                status = "in_progress"
            elif task["task_id"] in assigned_ids:
                status = "assigned"
            task_snapshots.append(
                {
                    "task_id": task["task_id"],
                    "status": status,
                    "release_time": task["release_time"],
                    "destination_node": task["destination_node"],
                    "weight": task["weight"],
                    "deadline": task["deadline"],
                    "assigned_vehicle_id": None,
                    "completion_time": task["completion_time"] if status == "completed" else None,
                    "overdue_penalized": bool(task["overdue"] and tick > task["deadline"]),
                }
            )

        pending = sum(1 for task in task_snapshots if task["status"] == "pending" and tick >= task["release_time"])
        assigned = sum(1 for task in task_snapshots if task["status"] == "assigned")
        in_progress = sum(1 for task in task_snapshots if task["status"] == "in_progress")
        completed = sum(1 for task in task_snapshots if task["status"] == "completed")
        overdue = sum(1 for task in task_snapshots if task["overdue_penalized"])

        timeline.append(
            {
                "tick": tick,
                "score": score,
                "task_stats": {
                    "total": len(tasks),
                    "pending": pending,
                    "assigned": assigned,
                    "in_progress": in_progress,
                    "completed": completed,
                    "overdue": overdue,
                    "timeout_rate": round(overdue / max(1, len(tasks)), 4),
                },
                "vehicles": vehicle_snapshots,
                "stations": [
                    {
                        "station_id": station.station_id,
                        "node_id": station.node_id,
                        "piles": station.piles,
                        "charge_rate": station.charge_rate,
                        "charging_vehicle_ids": [],
                        "queue_vehicle_ids": [],
                        "pressure": 0.0,
                    }
                    for station in stations
                ],
                "tasks": task_snapshots,
                "strategy_plans": [],
                "simulation": {
                    "failed": False,
                    "failure_reason": None,
                    "failure_tick": None,
                },
                "events": ["Gurobi static full-information solution snapshot"],
            }
        )
    return timeline


def _model_notes(depot_piles: int, vehicle_count: int, distribution_path: Optional[Path]) -> List[str]:
    notes = [
        "The model uses the original WorldManager generator and solves with full task knowledge.",
        "Each completed task is represented as depot loading, delivery, unloading, and return-to-depot availability.",
        "Partial depot charging is modeled through integer charging ticks between task slots.",
        "Because every task must be picked up at the depot and the depot station has piles equal to vehicle count by default, depot FIFO/capacity is non-binding in current scale configs.",
        "Non-depot charging-station FIFO is preserved in the modeling document as the required full extension; this compact solver focuses on the globally optimized depot-recharge schedule.",
        f"Depot piles={depot_piles}, vehicle_count={vehicle_count}.",
    ]
    if distribution_path is not None:
        notes.append(f"Task release/weight/deadline arrays were loaded from {distribution_path}.")
    return notes


def _status_name(status_code: int, grb) -> str:
    names = {
        grb.OPTIMAL: "OPTIMAL",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.CUTOFF: "CUTOFF",
        grb.ITERATION_LIMIT: "ITERATION_LIMIT",
        grb.NODE_LIMIT: "NODE_LIMIT",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.NUMERIC: "NUMERIC",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status_code, f"STATUS_{status_code}")


def _safe_attr(model, attr: str) -> Optional[float]:
    try:
        value = getattr(model, attr)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 6)

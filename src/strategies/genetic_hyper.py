from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Optional

from ..models import ChargingStation, Vehicle
from .base import StrategyContext
from .time_first_bundle import TimeFirstBundleStrategy


Gene = dict[str, float]


GENE_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "emergency_margin_ticks": (8.0, 24.0, True),
    "backlog_margin_gain": (1.0, 8.0, False),
    "bundle_wait_escape_ticks": (1.0, 5.0, True),
    "max_detour_ratio": (0.15, 0.42, False),
    "w_travel": (0.45, 1.9, False),
    "w_queue": (0.25, 2.4, False),
    "w_energy": (0.25, 2.2, False),
    "cost_epsilon": (0.05, 1.1, False),
    "mode1_bias": (-0.15, 0.55, False),
    "mode2_bias": (-0.55, 0.35, False),
    "mode3_bias": (-0.55, 0.35, False),
    "keep_idle_ratio": (0.12, 0.65, False),
    "all_low_battery_ratio": (0.18, 0.55, False),
    "depot_charge_ratio_no_pressure": (0.58, 0.98, False),
    "station_queue_scale": (0.45, 1.8, False),
}


DEFAULT_GENE: Gene = {
    "emergency_margin_ticks": 12.0,
    "backlog_margin_gain": 3.0,
    "bundle_wait_escape_ticks": 2.0,
    "max_detour_ratio": 0.25,
    "w_travel": 1.0,
    "w_queue": 1.2,
    "w_energy": 0.8,
    "cost_epsilon": 0.35,
    "mode1_bias": 0.0,
    "mode2_bias": 0.0,
    "mode3_bias": 0.0,
    "keep_idle_ratio": 0.30,
    "all_low_battery_ratio": 0.30,
    "depot_charge_ratio_no_pressure": 0.90,
    "station_queue_scale": 1.0,
}


def clamp_gene(gene: Mapping[str, float] | None = None) -> Gene:
    """Return a complete gene clipped to the configured search bounds."""

    merged: Gene = dict(DEFAULT_GENE)
    if gene:
        for key, value in gene.items():
            if key in GENE_BOUNDS:
                merged[key] = float(value)

    clipped: Gene = {}
    for key, (low, high, is_int) in GENE_BOUNDS.items():
        value = min(high, max(low, float(merged[key])))
        clipped[key] = float(int(round(value))) if is_int else value
    return clipped


def load_gene_model(path: Path) -> Gene:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "gene" in payload:
        return clamp_gene(payload["gene"])
    return clamp_gene(payload)


def save_gene_model(
    path: Path,
    *,
    scale_name: str,
    gene: Mapping[str, float],
    metadata: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "strategy": "genetic_hyper",
        "scale": scale_name,
        "gene": clamp_gene(gene),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def random_gene(rng: random.Random) -> Gene:
    gene: Gene = {}
    for key, (low, high, is_int) in GENE_BOUNDS.items():
        value = rng.uniform(low, high)
        gene[key] = float(int(round(value))) if is_int else value
    return clamp_gene(gene)


def perturb_gene(
    base_gene: Mapping[str, float],
    rng: random.Random,
    *,
    strength: float = 0.18,
) -> Gene:
    gene = clamp_gene(base_gene)
    for key, (low, high, is_int) in GENE_BOUNDS.items():
        width = high - low
        gene[key] = float(gene[key]) + rng.gauss(0.0, width * strength)
        if is_int:
            gene[key] = float(int(round(gene[key])))
    return clamp_gene(gene)


def crossover_gene(parent_a: Mapping[str, float], parent_b: Mapping[str, float], rng: random.Random) -> Gene:
    a = clamp_gene(parent_a)
    b = clamp_gene(parent_b)
    return clamp_gene({key: a[key] if rng.random() < 0.5 else b[key] for key in GENE_BOUNDS})


def mutate_gene(
    gene: Mapping[str, float],
    rng: random.Random,
    *,
    mutation_rate: float = 0.18,
    strength: float = 0.14,
) -> Gene:
    mutated = clamp_gene(gene)
    for key, (low, high, is_int) in GENE_BOUNDS.items():
        if rng.random() >= mutation_rate:
            continue
        width = high - low
        if rng.random() < 0.18:
            value = rng.uniform(low, high)
        else:
            value = float(mutated[key]) + rng.gauss(0.0, width * strength)
        mutated[key] = float(int(round(value))) if is_int else value
    return clamp_gene(mutated)


def initial_population(rng: random.Random, population_size: int) -> list[Gene]:
    seeds = [
        DEFAULT_GENE,
        {
            **DEFAULT_GENE,
            "bundle_wait_escape_ticks": 3.0,
            "max_detour_ratio": 0.34,
            "w_travel": 0.78,
            "w_queue": 1.55,
            "w_energy": 1.15,
            "mode1_bias": 0.20,
            "mode2_bias": -0.25,
            "mode3_bias": -0.18,
            "keep_idle_ratio": 0.36,
            "all_low_battery_ratio": 0.34,
            "depot_charge_ratio_no_pressure": 0.86,
            "station_queue_scale": 1.25,
        },
        {
            **DEFAULT_GENE,
            "emergency_margin_ticks": 16.0,
            "backlog_margin_gain": 5.2,
            "bundle_wait_escape_ticks": 2.0,
            "max_detour_ratio": 0.22,
            "w_travel": 1.22,
            "w_queue": 1.8,
            "w_energy": 1.45,
            "cost_epsilon": 0.18,
            "mode1_bias": -0.05,
            "mode2_bias": 0.08,
            "mode3_bias": -0.08,
            "keep_idle_ratio": 0.42,
            "all_low_battery_ratio": 0.38,
            "depot_charge_ratio_no_pressure": 0.82,
            "station_queue_scale": 1.45,
        },
        {
            **DEFAULT_GENE,
            "emergency_margin_ticks": 10.0,
            "backlog_margin_gain": 2.2,
            "bundle_wait_escape_ticks": 4.0,
            "max_detour_ratio": 0.38,
            "w_travel": 0.62,
            "w_queue": 0.92,
            "w_energy": 0.72,
            "cost_epsilon": 0.72,
            "mode1_bias": 0.28,
            "mode2_bias": -0.38,
            "mode3_bias": -0.34,
            "keep_idle_ratio": 0.22,
            "all_low_battery_ratio": 0.28,
            "depot_charge_ratio_no_pressure": 0.92,
            "station_queue_scale": 0.8,
        },
    ]

    population = [clamp_gene(seed) for seed in seeds[: max(0, population_size)]]
    while len(population) < population_size:
        base = rng.choice(population) if population and rng.random() < 0.55 else DEFAULT_GENE
        population.append(perturb_gene(base, rng, strength=0.22))
    return population[:population_size]


class GeneticHyperHeuristicStrategy(TimeFirstBundleStrategy):
    """GA-trained parameterized hyper-heuristic over TimeFirstBundle rules."""

    name = "genetic_hyper"

    def __init__(
        self,
        gene: Mapping[str, float] | None = None,
        gene_path: Path | None = None,
    ) -> None:
        super().__init__()
        loaded_gene = load_gene_model(gene_path) if gene_path is not None else gene
        self.gene = clamp_gene(loaded_gene)
        self._apply_gene()

    def _apply_gene(self) -> None:
        self.emergency_margin_ticks = int(self.gene["emergency_margin_ticks"])
        self.backlog_margin_gain = float(self.gene["backlog_margin_gain"])
        self.bundle_wait_escape_ticks = int(self.gene["bundle_wait_escape_ticks"])
        self.max_detour_ratio = float(self.gene["max_detour_ratio"])
        self.w_travel = float(self.gene["w_travel"])
        self.w_queue = float(self.gene["w_queue"])
        self.w_energy = float(self.gene["w_energy"])
        self.cost_epsilon = float(self.gene["cost_epsilon"])
        self.keep_idle_ratio = float(self.gene["keep_idle_ratio"])
        self.all_low_battery_ratio = float(self.gene["all_low_battery_ratio"])
        self.depot_charge_ratio_no_pressure = float(self.gene["depot_charge_ratio_no_pressure"])
        self.station_queue_scale = float(self.gene["station_queue_scale"])
        self.mode_biases = {
            "mode1": float(self.gene["mode1_bias"]),
            "mode2": float(self.gene["mode2_bias"]),
            "mode3": float(self.gene["mode3_bias"]),
        }

    def _feasible_modes_for_pair(
        self,
        vehicle: Vehicle,
        first_task,
        second_task,
        context: StrategyContext,
    ) -> list[dict]:
        modes = super()._feasible_modes_for_pair(vehicle, first_task, second_task, context)
        for mode_info in modes:
            bias = self.mode_biases.get(str(mode_info["mode"]), 0.0)
            mode_info["cost"] = max(0.0, float(mode_info["cost"]) * (1.0 + bias) + bias)
        return modes

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

            queue_cost = (
                station.pressure_index()
                * context.config.queue_penalty_factor
                * self.station_queue_scale
            )
            score = distance + queue_cost
            if score < best_score:
                best_score = score
                best_station = station

        return best_station

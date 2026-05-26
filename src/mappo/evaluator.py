from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

from ..config import ScaleConfig, SimulationConfig
from ..strategies import (
    EnergyAwareALNSStrategy,
    GeneticHyperHeuristicStrategy,
    MaxWeightStrategy,
    NearestTaskStrategy,
    TimeFirstBundleStrategy,
)
from ..strategies.mappo_strategy import MAPPOSTrategy
from ..world import WorldManager


def evaluate_strategies(
    *,
    scales: Iterable[ScaleConfig],
    sim_config: SimulationConfig,
    mappo_checkpoint: str,
    output_dir: str = "outputs/mappo",
    seeds_per_scale: int = 2,
) -> List[dict]:
    """固定种子评估基线策略与 MAPPO。"""

    strategy_factories = {
        "nearest_task": NearestTaskStrategy,
        "max_weight": MaxWeightStrategy,
        "time_first_bundle": TimeFirstBundleStrategy,
        "energy_aware_alns": EnergyAwareALNSStrategy,
        "genetic_hyper": GeneticHyperHeuristicStrategy,
        "mappo": lambda: MAPPOSTrategy(
            checkpoint_path=mappo_checkpoint,
            deterministic=True,
        ),
    }

    rows: List[dict] = []
    for scale in scales:
        for offset in range(seeds_per_scale):
            seeded_scale = replace(scale, seed=scale.seed + offset)
            for strategy_name, factory in strategy_factories.items():
                world = WorldManager(
                    scale=seeded_scale,
                    config=sim_config,
                    strategy=factory(),
                )
                result = world.run()
                rows.append(
                    {
                        "scale": scale.name,
                        "seed": seeded_scale.seed,
                        "strategy": strategy_name,
                        "total_score": result.total_score,
                        "completed_tasks": result.completed_tasks,
                        "total_tasks": result.total_tasks,
                        "overdue_tasks": result.overdue_tasks,
                        "timeout_rate": result.timeout_rate,
                        "total_distance": result.total_distance,
                        "simulation_failed": result.simulation_failed,
                    }
                )

    _write_eval_outputs(rows, output_dir)
    return rows


def _write_eval_outputs(rows: List[dict], output_dir: str) -> None:
    if not rows:
        return

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        "# MAPPO 评估报告",
        "",
        f"- 评估轮数：{len(rows)}",
        f"- 汇总文件：`{csv_path}`",
        "",
    ]
    by_strategy: Dict[str, List[dict]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)

    for strategy_name, strategy_rows in sorted(by_strategy.items()):
        avg_score = sum(float(item["total_score"]) for item in strategy_rows) / len(strategy_rows)
        avg_completion = sum(float(item["completed_tasks"]) / max(1, float(item["total_tasks"])) for item in strategy_rows) / len(strategy_rows)
        report_lines.append(
            f"- `{strategy_name}`: 平均得分 {avg_score:.2f}，平均完成率 {avg_completion:.4f}"
        )

    (root / "eval_report.md").write_text("\n".join(report_lines), encoding="utf-8")

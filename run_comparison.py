"""对比实验：TFB / ALNS / genetic_hyper / nearest_task / max_weight 在固定种子下的表现"""
from __future__ import annotations

from dataclasses import replace
from statistics import mean
from pathlib import Path

from src.config import SimulationConfig, default_scales
from src.strategies import (
    EnergyAwareALNSStrategy,
    GeneticHyperHeuristicStrategy,
    MaxWeightStrategy,
    NearestTaskStrategy,
    RLChargingStrategy,
    TimeFirstBundleStrategy,
)
from src.world import WorldManager

FIXED_SEEDS = {
    "small": [1393869867, 542157503, 1044925344, 92340001, 177964726],
    "medium": [354787187, 1866801784, 736577488, 108402446, 550665828],
    "large": [2118333588, 362326816, 1167052183, 2012485100, 1287757272],
}

STRATEGIES = {
    "time_first_bundle": TimeFirstBundleStrategy,
    "energy_aware_alns": EnergyAwareALNSStrategy,
    "genetic_hyper": GeneticHyperHeuristicStrategy,
    "nearest_task": NearestTaskStrategy,
    "max_weight": MaxWeightStrategy,
    "rl_charging": RLChargingStrategy,
}

def main():
    sim_config = SimulationConfig()
    scales = default_scales()

    for scale in scales:
        scale_name = scale.name
        seeds = FIXED_SEEDS[scale_name]
        print(f"\n{'='*60}")
        print(f"规模: {scale_name} | 种子: {seeds}")

        strategy_results = {}

        for strategy_name, strategy_factory in STRATEGIES.items():
            rows = []
            for seed in seeds:
                round_scale = replace(scale, seed=seed)
                if strategy_name == "genetic_hyper":
                    model_path = Path("outputs/genetic_hyper_eval/models") / f"{scale_name}_genetic_hyper_best.json"
                    strategy = (
                        GeneticHyperHeuristicStrategy(gene_path=model_path)
                        if model_path.exists()
                        else GeneticHyperHeuristicStrategy()
                    )
                else:
                    strategy = strategy_factory()
                world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
                result = world.run()
                rows.append({
                    "score": result.total_score,
                    "completion_rate": result.completed_tasks / max(1, result.total_tasks),
                    "timeout_rate": result.timeout_rate,
                    "overdue": result.overdue_tasks,
                    "distance": result.total_distance,
                    "failed": result.simulation_failed,
                })

            avg_score = mean(r["score"] for r in rows)
            avg_comp = mean(r["completion_rate"] for r in rows)
            avg_timeout = mean(r["timeout_rate"] for r in rows)
            avg_overdue = mean(r["overdue"] for r in rows)
            avg_distance = mean(r["distance"] for r in rows)
            fail_count = sum(1 for r in rows if r["failed"])

            strategy_results[strategy_name] = {
                "avg_score": avg_score,
                "avg_comp": avg_comp,
                "avg_timeout": avg_timeout,
                "avg_overdue": avg_overdue,
                "avg_distance": avg_distance,
                "fail_count": fail_count,
                "scores": [r["score"] for r in rows],
            }

        # Print table header
        print(f"{'策略':<22} {'平均分':>10} {'完成率':>8} {'超时率':>8} {'超时数':>7} {'总里程':>10} {'失败':>4}")
        print("-" * 75)

        # Sort by avg_score descending
        for name, stats in sorted(strategy_results.items(), key=lambda x: -x[1]["avg_score"]):
            print(
                f"{name:<22} {stats['avg_score']:>10.2f} {stats['avg_comp']:>8.4f} "
                f"{stats['avg_timeout']:>8.4f} {stats['avg_overdue']:>7.1f} "
                f"{stats['avg_distance']:>10.1f} {stats['fail_count']:>4}"
            )
            print(f"  各轮得分: {[round(s, 2) for s in stats['scores']]}")


if __name__ == "__main__":
    main()


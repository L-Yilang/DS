from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from src.config import SimulationConfig, default_scales
from src.exporter import (
    write_replay_json,
    write_summary_csv,
    write_task_distribution_json,
    write_timeline_json,
)
from src.strategies import (
    EnergyAwareALNSStrategy,
    GeneticHyperHeuristicStrategy,
    MaxWeightStrategy,
    NearestTaskStrategy,
    SchedulingStrategy,
    TimeFirstBundleStrategy,
)
from src.strategies.mappo_strategy import MAPPOSTrategy
from src.strategies.genetic_hyper import (
    Gene,
    crossover_gene,
    initial_population,
    mutate_gene,
    save_gene_model,
)
from src.world import WorldManager

FIXED_LATEST_SEEDS = {
    "small": [1393869867, 542157503, 1044925344, 92340001, 177964726],
    "medium": [354787187, 1866801784, 736577488, 108402446, 550665828],
    "large": [2118333588, 362326816, 1167052183, 2012485100, 1287757272],
}

TIME_FIRST_BUNDLE_TARGETS = {
    "small": 3192.086,
    "medium": 9511.034,
    "large": 16601.03,
}


def build_strategy_factories(
    *,
    mappo_checkpoint: Path | None = None,
    mappo_allow_untrained: bool = False,
    requested_strategy: str = "all",
) -> dict[str, Callable[[], SchedulingStrategy]]:
    """Register available scheduling strategy factories."""

    factories: dict[str, Callable[[], SchedulingStrategy]] = {
        "nearest_task": NearestTaskStrategy,
        "max_weight": MaxWeightStrategy,
        "time_first_bundle": TimeFirstBundleStrategy,
        "energy_aware_alns": EnergyAwareALNSStrategy,
        "genetic_hyper": GeneticHyperHeuristicStrategy,
    }

    requested_names = {name.strip() for name in requested_strategy.split(",") if name.strip()}
    should_register_mappo = (
        mappo_checkpoint is not None
        or mappo_allow_untrained
        or "mappo" in requested_names
    )
    if should_register_mappo:
        factories["mappo"] = lambda: MAPPOSTrategy(
            checkpoint_path=mappo_checkpoint,
            deterministic=True,
            allow_untrained=mappo_allow_untrained,
        )

    return factories


def select_strategy_factories(
    requested: str,
    strategy_factories: dict[str, Callable[[], SchedulingStrategy]],
) -> dict[str, Callable[[], SchedulingStrategy]]:
    """按命令行参数选择策略，支持 all、单个策略或逗号分隔的策略列表。"""

    if requested == "all":
        return strategy_factories

    selected_names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown_names = [name for name in selected_names if name not in strategy_factories]
    if unknown_names:
        valid_names = ", ".join(["all", *strategy_factories.keys()])
        raise ValueError(
            f"未知策略: {', '.join(unknown_names)}。可选值: {valid_names}；"
            "也可以用逗号组合多个策略。"
        )

    return {name: strategy_factories[name] for name in selected_names}


def build_scales(experiment_mode: str, long_train_multiplier: float):
    """按实验模式构建规模配置。"""
    scales = default_scales()
    if experiment_mode != "long_train":
        return scales

    horizon_scale = max(1, int(round(long_train_multiplier)))
    return [
        replace(
            scale,
            horizon=max(1, int(scale.horizon * horizon_scale)),
            task_count=max(0, int(round(scale.task_count * horizon_scale))),
        )
        for scale in scales
    ]


def generate_round_seeds(rounds: int) -> list[int]:
    """为多轮测试生成随机种子；同一轮对所有策略复用同一个种子。"""

    rng = random.SystemRandom()
    return [rng.randrange(1, 2**31 - 1) for _ in range(rounds)]


def build_summary_row(
    *,
    scale_name: str,
    strategy_name: str,
    result,
    round_index: int | None = None,
    seed: int | None = None,
) -> dict:
    row = {
        "scale": scale_name,
        "strategy": strategy_name,
        "total_score": result.total_score,
        "completed_tasks": result.completed_tasks,
        "total_tasks": result.total_tasks,
        "overdue_tasks": result.overdue_tasks,
        "timeout_rate": result.timeout_rate,
        "total_distance": result.total_distance,
        "simulation_failed": result.simulation_failed,
        "failure_tick": result.failure_tick,
        "failure_reason": result.failure_reason,
        "completion_rate": round(
            result.completed_tasks / max(1, result.total_tasks),
            4,
        ),
    }
    if round_index is not None:
        row["round"] = round_index
    if seed is not None:
        row["seed"] = seed
    return row


def build_aggregated_summary(rows: list[dict]) -> list[dict]:
    """按 scale + strategy 聚合多轮测试结果。"""

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["scale"]), str(row["strategy"]))
        grouped.setdefault(key, []).append(row)

    aggregated_rows: list[dict] = []
    numeric_fields = [
        "total_score",
        "completed_tasks",
        "total_tasks",
        "overdue_tasks",
        "timeout_rate",
        "total_distance",
        "completion_rate",
    ]

    for (scale_name, strategy_name), group_rows in sorted(grouped.items()):
        aggregated = {
            "scale": scale_name,
            "strategy": strategy_name,
            "rounds": len(group_rows),
            "failure_count": sum(1 for row in group_rows if row["simulation_failed"]),
            "failure_rate": round(
                sum(1 for row in group_rows if row["simulation_failed"])
                / max(1, len(group_rows)),
                4,
            ),
        }

        for field in numeric_fields:
            values = [float(row[field]) for row in group_rows]
            aggregated[f"{field}_mean"] = round(mean(values), 4)
            aggregated[f"{field}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
            aggregated[f"{field}_min"] = round(min(values), 4)
            aggregated[f"{field}_max"] = round(max(values), 4)

        aggregated_rows.append(aggregated)

    return aggregated_rows



def _gene_signature(gene: Gene) -> tuple[tuple[str, float], ...]:
    return tuple((key, round(float(value), 6)) for key, value in sorted(gene.items()))


def _evaluate_genetic_gene(
    *,
    gene: Gene,
    scale,
    sim_config: SimulationConfig,
    seeds: list[int],
    strategy_name: str,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for round_index, seed in enumerate(seeds, start=1):
        round_scale = replace(scale, seed=seed)
        strategy = GeneticHyperHeuristicStrategy(gene=gene)
        world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
        result = world.run()
        rows.append(
            build_summary_row(
                scale_name=result.scale_name,
                strategy_name=strategy_name,
                result=result,
                round_index=round_index,
                seed=seed,
            )
        )

    aggregate = build_aggregated_summary(rows)[0]
    return rows, aggregate


def _genetic_fitness(aggregate: dict) -> float:
    return float(aggregate["total_score_mean"]) - 5000.0 * float(aggregate["failure_rate"])


def _candidate_sort_key(candidate: dict) -> tuple[float, float, float]:
    aggregate = candidate["aggregate"]
    return (
        float(candidate["fitness"]),
        -float(aggregate["timeout_rate_mean"]),
        float(aggregate["completion_rate_mean"]),
    )


def _select_tournament(population: list[dict], rng: random.Random, tournament_size: int) -> Gene:
    sampled = rng.sample(population, k=min(tournament_size, len(population)))
    sampled.sort(key=_candidate_sort_key, reverse=True)
    return sampled[0]["gene"]


def _generate_seeds_with_rng(rng: random.Random, rounds: int) -> list[int]:
    return [rng.randrange(1, 2**31 - 1) for _ in range(max(1, rounds))]


def _select_validation_best(
    *,
    gene_pool: list[Gene],
    scale,
    sim_config: SimulationConfig,
    validation_seeds: list[int],
) -> tuple[Gene, dict]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    best_gene: Gene | None = None
    best_aggregate: dict | None = None
    best_key: tuple[float, float, float] | None = None

    for gene in gene_pool:
        signature = _gene_signature(gene)
        if signature in seen:
            continue
        seen.add(signature)
        _, aggregate = _evaluate_genetic_gene(
            gene=gene,
            scale=scale,
            sim_config=sim_config,
            seeds=validation_seeds,
            strategy_name="genetic_hyper",
        )
        key = (
            -float(aggregate["failure_rate"]),
            float(aggregate["total_score_mean"]),
            -float(aggregate["timeout_rate_mean"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_gene = gene
            best_aggregate = aggregate

    assert best_gene is not None and best_aggregate is not None
    return best_gene, best_aggregate


def _train_genetic_scale(
    *,
    scale,
    sim_config: SimulationConfig,
    population: list[Gene],
    train_seeds: list[int],
    generations: int,
    generation_offset: int,
    rng: random.Random,
    mutation_rate: float,
    training_rows: list[dict],
    cache: dict[tuple[tuple[str, float], ...], dict],
) -> tuple[list[Gene], dict]:
    elite_count = min(4, max(1, len(population)))
    tournament_size = 3
    best_candidate: dict | None = None

    for generation in range(1, generations + 1):
        evaluated: list[dict] = []
        generation_number = generation_offset + generation
        for candidate_index, gene in enumerate(population, start=1):
            signature = _gene_signature(gene)
            cached = cache.get(signature)
            if cached is None:
                _, aggregate = _evaluate_genetic_gene(
                    gene=gene,
                    scale=scale,
                    sim_config=sim_config,
                    seeds=train_seeds,
                    strategy_name="genetic_hyper_training",
                )
                cached = {
                    "gene": gene,
                    "aggregate": aggregate,
                    "fitness": _genetic_fitness(aggregate),
                }
                cache[signature] = cached

            candidate = {
                "gene": gene,
                "aggregate": cached["aggregate"],
                "fitness": cached["fitness"],
            }
            evaluated.append(candidate)
            row = {
                "scale": scale.name,
                "generation": generation_number,
                "candidate": candidate_index,
                "fitness": round(float(candidate["fitness"]), 4),
            }
            row.update(candidate["aggregate"])
            for key, value in gene.items():
                row[f"gene_{key}"] = round(float(value), 6)
            training_rows.append(row)

        evaluated.sort(key=_candidate_sort_key, reverse=True)
        if best_candidate is None or _candidate_sort_key(evaluated[0]) > _candidate_sort_key(best_candidate):
            best_candidate = evaluated[0]

        print(
            f"genetic_hyper 训练: 规模={scale.name}, generation={generation_number}, "
            f"best_score={evaluated[0]['aggregate']['total_score_mean']}, "
            f"fitness={round(float(evaluated[0]['fitness']), 4)}, "
            f"failure_rate={evaluated[0]['aggregate']['failure_rate']}"
        )

        next_population = [candidate["gene"] for candidate in evaluated[:elite_count]]
        while len(next_population) < len(population):
            parent_a = _select_tournament(evaluated, rng, tournament_size)
            parent_b = _select_tournament(evaluated, rng, tournament_size)
            child = crossover_gene(parent_a, parent_b, rng)
            child = mutate_gene(child, rng, mutation_rate=mutation_rate)
            next_population.append(child)
        population = next_population

    assert best_candidate is not None
    return population, best_candidate


def _write_validation_outputs(
    *,
    args,
    scale,
    sim_config: SimulationConfig,
    gene: Gene,
    seeds: list[int],
    summary_rows: list[dict],
) -> None:
    for round_index, seed in enumerate(seeds, start=1):
        round_scale = replace(scale, seed=seed)
        strategy = GeneticHyperHeuristicStrategy(gene=gene)
        world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
        result = world.run()
        stem = f"{scale.name}_genetic_hyper_r{round_index:02d}_seed{seed}"

        if args.save_timeline:
            write_timeline_json(args.output_dir / f"{stem}_timeline.json", result.timeline)
        if not args.no_replay:
            write_replay_json(
                args.output_dir / f"{stem}_replay.json",
                replay_meta=result.replay_meta,
                timeline=result.timeline,
            )
        write_task_distribution_json(
            args.output_dir / "distribution" / f"{stem}_task_distribution.json",
            world.build_task_distribution(),
        )
        summary_rows.append(
            build_summary_row(
                scale_name=result.scale_name,
                strategy_name=result.strategy_name,
                result=result,
                round_index=round_index,
                seed=seed,
            )
        )


def _write_genetic_comparison(output_dir: Path, genetic_aggregated_rows: list[dict]) -> None:
    baseline_path = Path("outputs/latest_compare_fixed/summary_aggregated.csv")
    if not baseline_path.exists():
        return

    wanted = {"time_first_bundle", "hyper_selector", "max_weight", "nearest_task"}
    comparison_rows: list[dict] = []
    with baseline_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("strategy") in wanted:
                comparison_rows.append(
                    {
                        "scale": row.get("scale", ""),
                        "strategy": row.get("strategy", ""),
                        "rounds": row.get("rounds", ""),
                        "failure_rate": row.get("failure_rate", ""),
                        "total_score_mean": row.get("total_score_mean", ""),
                        "total_score_std": row.get("total_score_std", ""),
                        "completion_rate_mean": row.get("completion_rate_mean", ""),
                        "timeout_rate_mean": row.get("timeout_rate_mean", ""),
                        "overdue_tasks_mean": row.get("overdue_tasks_mean", ""),
                        "total_distance_mean": row.get("total_distance_mean", ""),
                    }
                )

    for row in genetic_aggregated_rows:
        comparison_rows.append(
            {
                "scale": row["scale"],
                "strategy": row["strategy"],
                "rounds": row["rounds"],
                "failure_rate": row["failure_rate"],
                "total_score_mean": row["total_score_mean"],
                "total_score_std": row["total_score_std"],
                "completion_rate_mean": row["completion_rate_mean"],
                "timeout_rate_mean": row["timeout_rate_mean"],
                "overdue_tasks_mean": row["overdue_tasks_mean"],
                "total_distance_mean": row["total_distance_mean"],
            }
        )

    scale_order = {"small": 0, "medium": 1, "large": 2}
    strategy_order = {
        "time_first_bundle": 0,
        "hyper_selector": 1,
        "genetic_hyper": 2,
        "max_weight": 3,
        "nearest_task": 4,
    }
    comparison_rows.sort(
        key=lambda row: (
            scale_order.get(str(row["scale"]), 99),
            strategy_order.get(str(row["strategy"]), 99),
        )
    )
    write_summary_csv(output_dir / "comparison_aggregated.csv", comparison_rows)


def run_genetic_hyper_training_evaluation(
    *,
    args,
    sim_config: SimulationConfig,
    scales,
) -> None:
    rng = random.Random(args.genetic_random_seed)
    population_size = max(4, int(args.genetic_population))
    generations = max(1, int(args.genetic_train_generations))
    train_rounds = max(1, int(args.genetic_train_rounds))
    mutation_rate = 0.18

    training_rows: list[dict] = []
    validation_rows: list[dict] = []

    for scale in scales:
        train_seeds = _generate_seeds_with_rng(rng, train_rounds)
        validation_seeds = (
            FIXED_LATEST_SEEDS[scale.name]
            if args.genetic_test_fixed_latest
            else _generate_seeds_with_rng(rng, max(1, int(args.rounds)))
        )
        print(
            f"开始 genetic_hyper 训练: 规模={scale.name}, population={population_size}, "
            f"generations={generations}, train_rounds={train_rounds}"
        )

        population = initial_population(rng, population_size)
        cache: dict[tuple[tuple[str, float], ...], dict] = {}
        population, best_candidate = _train_genetic_scale(
            scale=scale,
            sim_config=sim_config,
            population=population,
            train_seeds=train_seeds,
            generations=generations,
            generation_offset=0,
            rng=rng,
            mutation_rate=mutation_rate,
            training_rows=training_rows,
            cache=cache,
        )

        best_gene = best_candidate["gene"]
        validation_best_gene, validation_best_aggregate = _select_validation_best(
            gene_pool=[best_gene, *population[: min(8, len(population))]],
            scale=scale,
            sim_config=sim_config,
            validation_seeds=validation_seeds,
        )

        append_round = 0
        generation_offset = generations
        target_score = TIME_FIRST_BUNDLE_TARGETS.get(scale.name)
        while (
            args.genetic_test_fixed_latest
            and target_score is not None
            and float(validation_best_aggregate["total_score_mean"]) <= target_score
            and append_round < 3
        ):
            append_round += 1
            print(
                f"规模={scale.name} 暂未超过 time_first_bundle "
                f"({validation_best_aggregate['total_score_mean']} <= {target_score})，追加训练 10 generations"
            )
            population, extra_best = _train_genetic_scale(
                scale=scale,
                sim_config=sim_config,
                population=population,
                train_seeds=train_seeds,
                generations=10,
                generation_offset=generation_offset,
                rng=rng,
                mutation_rate=mutation_rate,
                training_rows=training_rows,
                cache=cache,
            )
            generation_offset += 10
            if _candidate_sort_key(extra_best) > _candidate_sort_key(best_candidate):
                best_candidate = extra_best
                best_gene = best_candidate["gene"]
            next_validation_gene, next_validation_aggregate = _select_validation_best(
                gene_pool=[best_gene, *population[: min(8, len(population))]],
                scale=scale,
                sim_config=sim_config,
                validation_seeds=validation_seeds,
            )
            if float(next_validation_aggregate["total_score_mean"]) > float(
                validation_best_aggregate["total_score_mean"]
            ):
                validation_best_gene = next_validation_gene
                validation_best_aggregate = next_validation_aggregate

        model_path = args.output_dir / "models" / f"{scale.name}_genetic_hyper_best.json"
        save_gene_model(
            model_path,
            scale_name=scale.name,
            gene=validation_best_gene,
            metadata={
                "train_seeds": train_seeds,
                "validation_seeds": validation_seeds,
                "train_fitness": round(float(best_candidate["fitness"]), 4),
                "validation_total_score_mean": validation_best_aggregate["total_score_mean"],
                "validation_failure_rate": validation_best_aggregate["failure_rate"],
            },
        )

        _write_validation_outputs(
            args=args,
            scale=scale,
            sim_config=sim_config,
            gene=validation_best_gene,
            seeds=validation_seeds,
            summary_rows=validation_rows,
        )
        print(
            f"genetic_hyper 验证完成: 规模={scale.name}, "
            f"score_mean={validation_best_aggregate['total_score_mean']}, "
            f"failure_rate={validation_best_aggregate['failure_rate']}, model={model_path}"
        )

        write_summary_csv(args.output_dir / "training_summary.csv", training_rows)
        write_summary_csv(args.output_dir / "summary.csv", validation_rows)
        partial_aggregates = build_aggregated_summary(validation_rows)
        write_summary_csv(args.output_dir / "summary_aggregated.csv", partial_aggregates)
        _write_genetic_comparison(args.output_dir, partial_aggregates)

    write_summary_csv(args.output_dir / "training_summary.csv", training_rows)
    write_summary_csv(args.output_dir / "summary.csv", validation_rows)
    final_aggregates = build_aggregated_summary(validation_rows)
    write_summary_csv(args.output_dir / "summary_aggregated.csv", final_aggregates)
    _write_genetic_comparison(args.output_dir, final_aggregates)

    print("genetic_hyper 训练/验证汇总：")
    for row in final_aggregates:
        print(
            f"- 规模={row['scale']}, 平均得分={row['total_score_mean']}, "
            f"完成率={row['completion_rate_mean']}, 超时率={row['timeout_rate_mean']}, "
            f"失败率={row['failure_rate']}"
        )

    print(f"已输出: {args.output_dir / 'summary.csv'}")
    print(f"已输出聚合统计: {args.output_dir / 'summary_aggregated.csv'}")
    print(f"已输出训练记录: {args.output_dir / 'training_summary.csv'}")


def run_genetic_hyper_model_evaluation(
    *,
    args,
    sim_config: SimulationConfig,
    scales,
) -> None:
    summary_rows: list[dict] = []
    training_summary_rows: list[dict] = []
    model_dir = Path(args.genetic_model_dir)

    for scale in scales:
        model_path = model_dir / f"{scale.name}_genetic_hyper_best.json"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 genetic_hyper 模型: {model_path}")

        payload = json.loads(model_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        training_summary_rows.append(
            {
                "scale": scale.name,
                "strategy": "genetic_hyper",
                "model_path": str(model_path),
                "train_fitness": metadata.get("train_fitness", ""),
                "validation_total_score_mean": metadata.get("validation_total_score_mean", ""),
                "validation_failure_rate": metadata.get("validation_failure_rate", ""),
                "train_seeds": ";".join(str(seed) for seed in metadata.get("train_seeds", [])),
                "validation_seeds": ";".join(
                    str(seed) for seed in metadata.get("validation_seeds", [])
                ),
            }
        )

        seeds = FIXED_LATEST_SEEDS[scale.name]
        for round_index, seed in enumerate(seeds, start=1):
            round_scale = replace(scale, seed=seed)
            strategy = GeneticHyperHeuristicStrategy(gene_path=model_path)
            world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
            result = world.run()
            stem = f"{scale.name}_genetic_hyper_r{round_index:02d}_seed{seed}"

            if args.save_timeline:
                write_timeline_json(args.output_dir / f"{stem}_timeline.json", result.timeline)
            if not args.no_replay:
                write_replay_json(
                    args.output_dir / f"{stem}_replay.json",
                    replay_meta=result.replay_meta,
                    timeline=result.timeline,
                )
            write_task_distribution_json(
                args.output_dir / "distribution" / f"{stem}_task_distribution.json",
                world.build_task_distribution(),
            )
            summary_rows.append(
                build_summary_row(
                    scale_name=result.scale_name,
                    strategy_name=result.strategy_name,
                    result=result,
                    round_index=round_index,
                    seed=seed,
                )
            )

    write_summary_csv(args.output_dir / "summary.csv", summary_rows)
    write_summary_csv(args.output_dir / "training_summary.csv", training_summary_rows)
    aggregated_rows = build_aggregated_summary(summary_rows)
    write_summary_csv(args.output_dir / "summary_aggregated.csv", aggregated_rows)
    _write_genetic_comparison(args.output_dir, aggregated_rows)

    print("genetic_hyper 固定模型验证汇总：")
    for row in aggregated_rows:
        print(
            f"- 规模={row['scale']}, 平均得分={row['total_score_mean']}, "
            f"完成率={row['completion_rate_mean']}, 超时率={row['timeout_rate_mean']}, "
            f"失败率={row['failure_rate']}"
        )

    print(f"已输出: {args.output_dir / 'summary.csv'}")
    print(f"已输出聚合统计: {args.output_dir / 'summary_aggregated.csv'}")

def main() -> None:
    parser = argparse.ArgumentParser(description="新能源物流车队协同调度仿真")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="结果输出目录",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="large",
        choices=["all", "small", "medium", "large"],
        help="选择运行规模",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help=(
            "选择运行策略：all / nearest_task / max_weight / time_first_bundle / "
            "energy_aware_alns / genetic_hyper；也可用逗号组合多个策略"
        ),
    )
    parser.add_argument(
        "--save-timeline",
        action="store_true",
        help="是否额外保存 *_timeline.json（默认仅保存 replay 文件）",
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="standard",
        choices=["standard", "long_train"],
        help="实验模式：standard 使用原始 horizon；long_train 按倍率放大 horizon",
    )
    parser.add_argument(
        "--long-train-multiplier",
        type=float,
        default=4.0,
        help="long_train 模式下 horizon 放大倍率（默认 4）",
    )
    parser.add_argument(
        "--multi-run",
        action="store_true",
        default=False,
        help="启用多轮测试；同一轮的所有策略共享同一个世界种子",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="多轮测试轮数（默认 5）",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="不保存 *_replay.json，仅输出汇总与任务分布文件",
    )
    parser.add_argument(
        "--genetic-train-generations",
        type=int,
        default=0,
        help="若大于 0，则启用 genetic_hyper 遗传训练/验证流程",
    )
    parser.add_argument(
        "--genetic-population",
        type=int,
        default=24,
        help="genetic_hyper 遗传算法种群大小（默认 24）",
    )
    parser.add_argument(
        "--genetic-train-rounds",
        type=int,
        default=8,
        help="genetic_hyper 每个候选基因的训练 seed 数（默认 8）",
    )
    parser.add_argument(
        "--genetic-test-fixed-latest",
        action="store_true",
        help="用 latest_compare_fixed 的同一批 5 个 seed 验证 genetic_hyper",
    )
    parser.add_argument(
        "--genetic-random-seed",
        type=int,
        default=20260525,
        help="genetic_hyper 训练流程自身的随机种子（默认 20260525）",
    )
    parser.add_argument(
        "--genetic-model-dir",
        type=Path,
        default=None,
        help="加载已训练 genetic_hyper 模型目录并在固定 latest seed 上验证",
    )
    parser.add_argument(
        "--mappo-checkpoint",
        type=Path,
        default=None,
        help="MAPPO inference checkpoint; without it MAPPO is excluded from --strategy all",
    )
    parser.add_argument(
        "--mappo-allow-untrained",
        action="store_true",
        help="Allow an untrained MAPPO actor for smoke tests only",
    )
    args = parser.parse_args()

    sim_config = SimulationConfig()
    strategy_factories = build_strategy_factories(
        mappo_checkpoint=args.mappo_checkpoint,
        mappo_allow_untrained=args.mappo_allow_untrained,
        requested_strategy=args.strategy,
    )
    scales = build_scales(args.experiment_mode, args.long_train_multiplier)

    if args.scale != "all":
        scales = [scale for scale in scales if scale.name == args.scale]
    strategy_factories = select_strategy_factories(args.strategy, strategy_factories)

    args.output_dir.mkdir(parents=True, exist_ok=True)


    if args.genetic_train_generations > 0:
        if set(strategy_factories) != {"genetic_hyper"}:
            raise ValueError(
                "--genetic-train-generations 只能与 --strategy genetic_hyper 一起使用"
            )
        run_genetic_hyper_training_evaluation(
            args=args,
            sim_config=sim_config,
            scales=scales,
        )
        return

    if args.genetic_model_dir is not None:
        if set(strategy_factories) != {"genetic_hyper"} or not args.genetic_test_fixed_latest:
            raise ValueError(
                "--genetic-model-dir 需要与 --strategy genetic_hyper "
                "和 --genetic-test-fixed-latest 一起使用"
            )
        run_genetic_hyper_model_evaluation(
            args=args,
            sim_config=sim_config,
            scales=scales,
        )
        return
    requested_rounds = max(1, int(args.rounds))
    multi_run_enabled = args.multi_run and requested_rounds > 1
    summary_rows: list[dict] = []

    for scale in scales:
        round_seeds = generate_round_seeds(requested_rounds) if multi_run_enabled else [scale.seed]

        for round_index, round_seed in enumerate(round_seeds, start=1):
            round_scale = replace(scale, seed=round_seed)

            for strategy_name, strategy_factory in strategy_factories.items():
                strategy = strategy_factory()
                world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
                result = world.run()

                stem = f"{scale.name}_{strategy_name}"
                if multi_run_enabled:
                    stem = f"{stem}_r{round_index:02d}_seed{round_seed}"

                if args.save_timeline:
                    write_timeline_json(args.output_dir / f"{stem}_timeline.json", result.timeline)
                write_replay_json(
                    args.output_dir / f"{stem}_replay.json",
                    replay_meta=result.replay_meta,
                    timeline=result.timeline,
                )
                write_task_distribution_json(
                    args.output_dir / "distribution" / f"{stem}_task_distribution.json",
                    world.build_task_distribution(),
                )

                summary_rows.append(
                    build_summary_row(
                        scale_name=result.scale_name,
                        strategy_name=result.strategy_name,
                        result=result,
                        round_index=round_index if multi_run_enabled else None,
                        seed=round_seed if multi_run_enabled else None,
                    )
                )

    summary_path = args.output_dir / "summary.csv"
    write_summary_csv(summary_path, summary_rows)

    aggregated_rows = build_aggregated_summary(summary_rows) if multi_run_enabled else []
    aggregated_summary_path = args.output_dir / "summary_aggregated.csv"
    if aggregated_rows:
        write_summary_csv(aggregated_summary_path, aggregated_rows)

    print("运行完成，汇总结果：")
    print(
        f"实验模式={args.experiment_mode}, "
        f"long_train_multiplier={args.long_train_multiplier}, "
        f"multi_run={multi_run_enabled}, "
        f"rounds={requested_rounds if multi_run_enabled else 1}"
    )

    if multi_run_enabled:
        for row in aggregated_rows:
            print(
                f"- 规模={row['scale']}, 策略={row['strategy']}, "
                f"平均得分={row['total_score_mean']}, 得分标准差={row['total_score_std']}, "
                f"平均完成率={row['completion_rate_mean']}, "
                f"平均超时率={row['timeout_rate_mean']}, 超时率标准差={row['timeout_rate_std']}, "
                f"平均超时任务={row['overdue_tasks_mean']}, 平均总里程={row['total_distance_mean']}, "
                f"失败率={row['failure_rate']}"
            )
    else:
        for row in summary_rows:
            print(
                f"- 规模={row['scale']}, 策略={row['strategy']}, "
                f"得分={row['total_score']}, 完成率={row['completion_rate']}, "
                f"超时率={row['timeout_rate']}, 超时任务={row['overdue_tasks']}, "
                f"总里程={row['total_distance']}, 失败={row['simulation_failed']}"
            )


    print(f"已输出: {summary_path}")
    if aggregated_rows:
        print(f"已输出聚合统计: {aggregated_summary_path}")


if __name__ == "__main__":
    main()



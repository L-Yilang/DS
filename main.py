from __future__ import annotations

import argparse
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
    RLChargingStrategy,
    SchedulingStrategy,
    TimeFirstBundleStrategy,
)
from src.world import WorldManager


def build_strategy_factories() -> dict[str, Callable[[], SchedulingStrategy]]:
    """注册可用调度策略构造器。"""

    return {
        "nearest_task": NearestTaskStrategy,
        "max_weight": MaxWeightStrategy,
        "time_first_bundle": TimeFirstBundleStrategy,
        "rl_charging": RLChargingStrategy,
        "energy_aware_alns": EnergyAwareALNSStrategy,
        "genetic_hyper": GeneticHyperHeuristicStrategy,
    }


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
            "rl_charging / energy_aware_alns / genetic_hyper；也可用逗号组合多个策略"
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
    args = parser.parse_args()

    sim_config = SimulationConfig()
    strategy_factories = build_strategy_factories()
    scales = build_scales(args.experiment_mode, args.long_train_multiplier)

    if args.scale != "all":
        scales = [scale for scale in scales if scale.name == args.scale]
    strategy_factories = select_strategy_factories(args.strategy, strategy_factories)

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

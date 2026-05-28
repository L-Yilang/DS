from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from src.config import SimulationConfig, default_scales
from src.exporter import write_replay_json, write_summary_csv, write_task_distribution_json
from src.strategies import (
    EnergyAwareALNSStrategy,
    GeneticHyperHeuristicStrategy,
    MaxWeightStrategy,
    NearestTaskStrategy,
    SchedulingStrategy,
    TimeFirstBundleStrategy,
)
from src.strategies.mappo_strategy import MAPPOSTrategy
from src.world import WorldManager


FIXED_SEEDS = {
    "small": [1393869867, 542157503, 1044925344, 92340001, 177964726],
    "medium": [354787187, 1866801784, 736577488, 108402446, 550665828],
    "large": [2118333588, 362326816, 1167052183, 2012485100, 1287757272],
}

STRATEGY_ORDER = [
    "genetic_hyper",
    "mappo",
    "energy_aware_alns",
    "time_first_bundle",
    "max_weight",
    "nearest_task",
]

DISPLAY_NAMES = {
    "genetic_hyper": "genetic_hyper",
    "mappo": "MAPPO",
    "energy_aware_alns": "ALNS",
    "time_first_bundle": "time_first_bundle",
    "max_weight": "max_weight",
    "nearest_task": "nearest_task",
}

MODEL_DIR = Path("outputs/genetic_hyper_eval/models")
MAPPO_CHECKPOINT_DIR = Path("outputs/mappo/checkpoints")


def resolve_mappo_checkpoint(checkpoint_dir: Path, scale_name: str) -> Path | None:
    """优先选择该规模专属的 best 权重，回退到全局 best 与 latest。"""

    candidates = [
        checkpoint_dir / f"best_{scale_name}.pt",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "latest.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_strategy_factory(
    scale_name: str,
    strategy_name: str,
    *,
    mappo_checkpoint_dir: Path = MAPPO_CHECKPOINT_DIR,
) -> Callable[[], SchedulingStrategy]:
    if strategy_name == "genetic_hyper":
        model_path = MODEL_DIR / f"{scale_name}_genetic_hyper_best.json"
        return lambda: GeneticHyperHeuristicStrategy(gene_path=model_path)
    if strategy_name == "mappo":
        checkpoint_path = resolve_mappo_checkpoint(mappo_checkpoint_dir, scale_name)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"未找到 MAPPO 权重，期望路径之一存在: "
                f"{mappo_checkpoint_dir / f'best_{scale_name}.pt'}、"
                f"{mappo_checkpoint_dir / 'best.pt'} 或 "
                f"{mappo_checkpoint_dir / 'latest.pt'}。"
                "请先运行 `python train_mappo.py` 训练得到权重，或使用 "
                "`--skip-mappo` 跳过 MAPPO 验证。"
            )
        return lambda: MAPPOSTrategy(
            checkpoint_path=checkpoint_path,
            deterministic=True,
        )
    if strategy_name == "energy_aware_alns":
        return EnergyAwareALNSStrategy
    if strategy_name == "time_first_bundle":
        return TimeFirstBundleStrategy
    if strategy_name == "max_weight":
        return MaxWeightStrategy
    if strategy_name == "nearest_task":
        return NearestTaskStrategy
    raise ValueError(f"Unknown strategy: {strategy_name}")


def clean_outputs(output_dir: Path, *, mappo_checkpoint_dir: Path = MAPPO_CHECKPOINT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_models: dict[str, str] = {}
    if MODEL_DIR.exists():
        for model_path in MODEL_DIR.glob("*_genetic_hyper_best.json"):
            keep_models[model_path.name] = model_path.read_text(encoding="utf-8")

    keep_mappo: dict[str, bytes] = {}
    if mappo_checkpoint_dir.exists():
        for ckpt_path in mappo_checkpoint_dir.glob("*.pt"):
            keep_mappo[ckpt_path.name] = ckpt_path.read_bytes()

    resolved_output = output_dir.resolve()
    resolved_cwd = Path.cwd().resolve()
    if resolved_cwd not in resolved_output.parents and resolved_output != resolved_cwd:
        raise RuntimeError(f"Refuse to clean outside workspace: {resolved_output}")

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in keep_models.items():
        (MODEL_DIR / name).write_text(content, encoding="utf-8")

    if keep_mappo:
        mappo_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in keep_mappo.items():
            (mappo_checkpoint_dir / name).write_bytes(payload)


def build_summary_row(*, scale_name: str, strategy_name: str, result, round_index: int, seed: int) -> dict:
    return {
        "scale": scale_name,
        "strategy": strategy_name,
        "round": round_index,
        "seed": seed,
        "total_score": result.total_score,
        "completed_tasks": result.completed_tasks,
        "total_tasks": result.total_tasks,
        "overdue_tasks": result.overdue_tasks,
        "timeout_rate": result.timeout_rate,
        "total_distance": result.total_distance,
        "simulation_failed": result.simulation_failed,
        "failure_tick": result.failure_tick,
        "failure_reason": result.failure_reason,
        "completion_rate": round(result.completed_tasks / max(1, result.total_tasks), 4),
    }


def build_aggregated_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["scale"]), str(row["strategy"])), []).append(row)

    numeric_fields = [
        "total_score",
        "completed_tasks",
        "total_tasks",
        "overdue_tasks",
        "timeout_rate",
        "total_distance",
        "completion_rate",
    ]
    aggregated_rows: list[dict] = []
    scale_order = {"small": 0, "medium": 1, "large": 2}
    strategy_order = {name: index for index, name in enumerate(STRATEGY_ORDER)}

    for (scale_name, strategy_name), group_rows in grouped.items():
        row = {
            "scale": scale_name,
            "strategy": strategy_name,
            "rounds": len(group_rows),
            "failure_count": sum(1 for item in group_rows if item["simulation_failed"]),
            "failure_rate": round(
                sum(1 for item in group_rows if item["simulation_failed"]) / max(1, len(group_rows)),
                4,
            ),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in group_rows]
            row[f"{field}_mean"] = round(mean(values), 4)
            row[f"{field}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
            row[f"{field}_min"] = round(min(values), 4)
            row[f"{field}_max"] = round(max(values), 4)
        aggregated_rows.append(row)

    aggregated_rows.sort(
        key=lambda row: (
            scale_order.get(str(row["scale"]), 99),
            strategy_order.get(str(row["strategy"]), 99),
        )
    )
    return aggregated_rows


def write_comparison_outputs(output_dir: Path, aggregated_rows: list[dict]) -> None:
    comparison_rows = [
        {
            "scale": row["scale"],
            "strategy": DISPLAY_NAMES.get(str(row["strategy"]), str(row["strategy"])),
            "total_score_mean": row["total_score_mean"],
            "completion_rate_mean": row["completion_rate_mean"],
            "timeout_rate_mean": row["timeout_rate_mean"],
            "overdue_tasks_mean": row["overdue_tasks_mean"],
            "total_distance_mean": row["total_distance_mean"],
            "failure_rate": row["failure_rate"],
        }
        for row in aggregated_rows
    ]
    write_summary_csv(output_dir / "comparison_aggregated.csv", comparison_rows)

    markdown_lines = []
    headers = ["策略", "平均得分", "平均完成率", "平均超时率", "平均超时任务", "平均总里程", "失败率"]
    for scale_name in ["small", "medium", "large"]:
        markdown_lines.append(f"## {scale_name} 规模")
        markdown_lines.append("")
        markdown_lines.append("| " + " | ".join(headers) + " |")
        markdown_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in comparison_rows:
            if row["scale"] != scale_name:
                continue
            markdown_lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["strategy"]),
                        f"{float(row['total_score_mean']):.3f}",
                        f"{float(row['completion_rate_mean']):.4f}",
                        f"{float(row['timeout_rate_mean']):.4f}",
                        f"{float(row['overdue_tasks_mean']):.1f}",
                        f"{float(row['total_distance_mean']):.2f}",
                        f"{float(row['failure_rate']):.1f}",
                    ]
                )
                + " |"
            )
        markdown_lines.append("")
    (output_dir / "comparison_tables.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    with (output_dir / "comparison_tables.json").open("w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final fixed-seed output experiments")
    parser.add_argument("--clean", action="store_true", help="清理 outputs 后重新生成最终结果")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="结果输出目录")
    parser.add_argument(
        "--mappo-checkpoint-dir",
        type=Path,
        default=MAPPO_CHECKPOINT_DIR,
        help="MAPPO 权重目录（按规模优先选用 best_{scale}.pt，回退到 best.pt 或 latest.pt）",
    )
    parser.add_argument(
        "--skip-mappo",
        action="store_true",
        help="跳过 MAPPO 验证（在无可用权重或只想跑基线策略时使用）",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    mappo_checkpoint_dir = args.mappo_checkpoint_dir
    if args.clean:
        clean_outputs(output_dir, mappo_checkpoint_dir=mappo_checkpoint_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = SimulationConfig()
    summary_rows: list[dict] = []

    for scale in default_scales():
        for strategy_name in STRATEGY_ORDER:
            if strategy_name == "mappo":
                if args.skip_mappo:
                    print(f"{scale.name} mappo: 已通过 --skip-mappo 跳过")
                    continue
                if resolve_mappo_checkpoint(mappo_checkpoint_dir, scale.name) is None:
                    print(
                        f"{scale.name} mappo: 未在 {mappo_checkpoint_dir} 找到可用权重，"
                        "跳过 MAPPO 验证（可先运行 `python train_mappo.py` 训练得到权重）"
                    )
                    continue

            strategy_factory = build_strategy_factory(
                scale.name,
                strategy_name,
                mappo_checkpoint_dir=mappo_checkpoint_dir,
            )
            for round_index, seed in enumerate(FIXED_SEEDS[scale.name], start=1):
                round_scale = replace(scale, seed=seed)
                strategy = strategy_factory()
                world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
                result = world.run()
                stem = f"{scale.name}_{strategy_name}_r{round_index:02d}_seed{seed}"

                write_replay_json(
                    output_dir / f"{stem}_replay.json",
                    replay_meta=result.replay_meta,
                    timeline=result.timeline,
                )
                write_task_distribution_json(
                    output_dir / "distribution" / f"{stem}_task_distribution.json",
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
                print(
                    f"{scale.name} {strategy_name} r{round_index:02d}: "
                    f"score={result.total_score}, failed={result.simulation_failed}"
                )

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    aggregated_rows = build_aggregated_summary(summary_rows)
    write_summary_csv(output_dir / "summary_aggregated.csv", aggregated_rows)
    write_comparison_outputs(output_dir, aggregated_rows)

    print("Final comparison complete:")
    for row in aggregated_rows:
        print(
            f"- {row['scale']} {row['strategy']}: "
            f"score={row['total_score_mean']}, completion={row['completion_rate_mean']}, "
            f"timeout={row['timeout_rate_mean']}, failed={row['failure_rate']}"
        )


if __name__ == "__main__":
    main()



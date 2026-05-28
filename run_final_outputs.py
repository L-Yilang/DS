from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from src.config import SimulationConfig, default_scales
from src.exporter import write_replay_json, write_summary_csv, write_task_distribution_json
from src.mappo.config import build_preset_config
from src.mappo.trainer import MAPPOTrainer
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

SUPPLEMENTARY_STRATEGY_ORDER = [*STRATEGY_ORDER, "mappo"]

DISPLAY_NAMES = {
    "genetic_hyper": "genetic_hyper",
    "mappo": "MAPPO",
    "energy_aware_alns": "ALNS",
    "time_first_bundle": "time_first_bundle",
    "max_weight": "max_weight",
    "nearest_task": "nearest_task",
    "mappo": "MAPPO",
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
    if strategy_name == "mappo":
        if mappo_checkpoint is None:
            raise ValueError("MAPPO strategy requires a checkpoint")
        return lambda: MAPPOSTrategy(checkpoint_path=mappo_checkpoint, deterministic=True)
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
    strategy_order = {name: index for index, name in enumerate(SUPPLEMENTARY_STRATEGY_ORDER)}

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


def apply_overrides(instance, overrides: dict):
    if not overrides:
        return instance
    return replace(instance, **overrides)


def medium_scale_for_experiment(experiment: SupplementaryExperiment):
    medium = next(scale for scale in default_scales() if scale.name == "medium")
    return apply_overrides(medium, experiment.scale_overrides)


def sim_config_for_experiment(experiment: SupplementaryExperiment) -> SimulationConfig:
    return apply_overrides(SimulationConfig(), experiment.sim_overrides)


def train_mappo_for_experiment(
    *,
    experiment: SupplementaryExperiment,
    scale,
    sim_config: SimulationConfig,
    output_dir: Path,
    episodes: int,
    preset: str,
    device: str,
    retrain: bool,
) -> Path:
    checkpoint_dir = output_dir / "mappo" / experiment.name / "checkpoints"
    eval_dir = output_dir / "mappo" / experiment.name
    checkpoint_path = checkpoint_dir / "best_medium.pt"
    if checkpoint_path.exists() and not retrain:
        return checkpoint_path

    mappo_config = build_preset_config(preset, device=device)
    if preset == "smoke":
        mappo_config = replace(mappo_config, smoke_episodes_per_stage=max(1, episodes))
    else:
        mappo_config = replace(mappo_config, episodes_per_stage=max(1, episodes))
    mappo_config = replace(
        mappo_config,
        checkpoint_dir=str(checkpoint_dir),
        eval_dir=str(eval_dir),
        seed=20260525,
    )
    trainer = MAPPOTrainer(
        scales={"medium": scale},
        sim_config=sim_config,
        mappo_config=mappo_config,
    )
    print(
        f"MAPPO training: experiment={experiment.name}, scale=medium, "
        f"episodes={episodes}, preset={preset}, device={trainer.device}"
    )
    trainer.train(preset=preset, stage_names=["medium"])
    return checkpoint_path


def markdown_table(rows: list[dict]) -> list[str]:
    headers = ["策略", "平均得分", "完成率", "超时率", "超时任务", "总里程", "失败率"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES.get(str(row["strategy"]), str(row["strategy"])),
                    f"{float(row['total_score_mean']):.3f}",
                    f"{float(row['completion_rate_mean']):.4f}",
                    f"{float(row['timeout_rate_mean']):.4f}",
                    f"{float(row['overdue_tasks_mean']):.1f}",
                    f"{float(row['total_distance_mean']):.2f}",
                    f"{float(row['failure_rate']):.4f}",
                ]
            )
            + " |"
        )
    return lines


def describe_best_and_risk(aggregated_rows: list[dict]) -> tuple[dict, dict]:
    best = max(
        aggregated_rows,
        key=lambda row: (
            -float(row["failure_rate"]),
            float(row["total_score_mean"]),
            float(row["completion_rate_mean"]),
            -float(row["timeout_rate_mean"]),
        ),
    )
    riskiest = max(
        aggregated_rows,
        key=lambda row: (
            float(row["failure_rate"]),
            float(row["timeout_rate_mean"]),
            -float(row["completion_rate_mean"]),
        ),
    )
    return best, riskiest


def baseline_map(all_rows: list[dict]) -> dict[str, dict]:
    return {
        str(row["strategy"]): row
        for row in all_rows
        if row.get("experiment") == "standard_baseline"
    }


def write_experiment_markdown(
    *,
    output_dir: Path,
    experiment: SupplementaryExperiment,
    scale,
    sim_config: SimulationConfig,
    aggregated_rows: list[dict],
    checkpoint_path: Path,
    mappo_episodes: int,
    baseline_rows: dict[str, dict] | None = None,
) -> None:
    best, riskiest = describe_best_and_risk(aggregated_rows)
    baseline_rows = baseline_rows or {}
    lines = [
        f"# {experiment.title}",
        "",
        "## 实验设置",
        "",
        f"- 规模：仅运行中尺度，固定测试种子为 `{', '.join(str(seed) for seed in FIXED_SEEDS['medium'])}`。",
        f"- 规模扰动：{experiment.scale_description}",
        f"- 仿真扰动：{experiment.sim_description}",
        f"- MAPPO：在该实验设置下单独训练 {mappo_episodes} 轮，仅训练中尺度；评估 checkpoint 为 `{checkpoint_path}`。",
        f"- 中尺度实际参数：task_count={scale.task_count}, "
        f"task_time_distribution={scale.task_time_distribution}, "
        f"task_time_mean_ratio={scale.task_time_mean_ratio}, "
        f"task_time_std_ratio={scale.task_time_std_ratio}, "
        f"energy_per_distance={sim_config.energy_per_distance}, "
        f"traffic=({sim_config.traffic_jam_probability}, {sim_config.traffic_jam_multiplier})。",
        "",
        "## 聚合结果",
        "",
        *markdown_table(aggregated_rows),
        "",
    ]
    if experiment.name != "standard_baseline" and baseline_rows:
        lines.extend(
            [
                "## 与普通情况对比",
                "",
                "| 策略 | 得分变化 | 完成率变化 | 超时率变化 | 总里程变化 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in aggregated_rows:
            baseline = baseline_rows.get(str(row["strategy"]))
            if baseline is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        DISPLAY_NAMES.get(str(row["strategy"]), str(row["strategy"])),
                        f"{float(row['total_score_mean']) - float(baseline['total_score_mean']):.3f}",
                        f"{float(row['completion_rate_mean']) - float(baseline['completion_rate_mean']):.4f}",
                        f"{float(row['timeout_rate_mean']) - float(baseline['timeout_rate_mean']):.4f}",
                        f"{float(row['total_distance_mean']) - float(baseline['total_distance_mean']):.2f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
        "## 简要分析",
        "",
        f"- {experiment.analysis_hint}",
        (
            f"- 综合失败率、平均得分和完成率，本实验中表现最好的策略为 "
            f"`{DISPLAY_NAMES.get(str(best['strategy']), str(best['strategy']))}`，"
            f"平均得分 {float(best['total_score_mean']):.3f}，"
            f"完成率 {float(best['completion_rate_mean']):.4f}。"
        ),
        (
            f"- 风险最高的策略为 `{DISPLAY_NAMES.get(str(riskiest['strategy']), str(riskiest['strategy']))}`，"
            f"失败率 {float(riskiest['failure_rate']):.4f}，"
            f"超时率 {float(riskiest['timeout_rate_mean']):.4f}。"
        ),
        "",
        "## 输出文件",
        "",
        f"- 明细：`{experiment.name}/summary.csv`",
        f"- 聚合：`{experiment.name}/summary_aggregated.csv`",
        f"- 回放：`{experiment.name}/*_replay.json`",
        f"- 任务分布：`{experiment.name}/distribution/*_task_distribution.json`",
        ]
    )
    (output_dir / f"{experiment.name}.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_supplementary_experiment(
    *,
    experiment: SupplementaryExperiment,
    output_dir: Path,
    args,
    baseline_rows: dict[str, dict] | None = None,
) -> list[dict]:
    experiment_dir = output_dir / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    scale = medium_scale_for_experiment(experiment)
    sim_config = sim_config_for_experiment(experiment)
    checkpoint_path = train_mappo_for_experiment(
        experiment=experiment,
        scale=scale,
        sim_config=sim_config,
        output_dir=output_dir,
        episodes=args.mappo_episodes,
        preset=args.mappo_preset,
        device=args.mappo_device,
        retrain=args.mappo_retrain,
    )

    summary_rows: list[dict] = []
    for strategy_name in SUPPLEMENTARY_STRATEGY_ORDER:
        strategy_factory = build_strategy_factory(
            scale.name,
            strategy_name,
            mappo_checkpoint=checkpoint_path if strategy_name == "mappo" else None,
        )
        for round_index, seed in enumerate(FIXED_SEEDS["medium"], start=1):
            round_scale = replace(scale, seed=seed)
            strategy = strategy_factory()
            world = WorldManager(scale=round_scale, config=sim_config, strategy=strategy)
            result = world.run()
            stem = f"{scale.name}_{strategy_name}_r{round_index:02d}_seed{seed}"

            write_replay_json(
                experiment_dir / f"{stem}_replay.json",
                replay_meta=result.replay_meta,
                timeline=result.timeline,
            )
            write_task_distribution_json(
                experiment_dir / "distribution" / f"{stem}_task_distribution.json",
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
                f"{experiment.name} {scale.name} {strategy_name} r{round_index:02d}: "
                f"score={result.total_score}, failed={result.simulation_failed}"
            )

    write_summary_csv(experiment_dir / "summary.csv", summary_rows)
    aggregated_rows = build_aggregated_summary(summary_rows)
    write_summary_csv(experiment_dir / "summary_aggregated.csv", aggregated_rows)
    write_comparison_outputs(experiment_dir, aggregated_rows)
    write_experiment_markdown(
        output_dir=output_dir,
        experiment=experiment,
        scale=scale,
        sim_config=sim_config,
        aggregated_rows=aggregated_rows,
        checkpoint_path=checkpoint_path,
        mappo_episodes=args.mappo_episodes,
        baseline_rows=baseline_rows,
    )

    return [
        {
            "experiment": experiment.name,
            "experiment_title": experiment.title,
            **row,
        }
        for row in aggregated_rows
    ]


def write_supplementary_texts(output_dir: Path, all_rows: list[dict]) -> None:
    by_experiment: dict[str, list[dict]] = {}
    titles: dict[str, str] = {}
    for row in all_rows:
        by_experiment.setdefault(str(row["experiment"]), []).append(row)
        titles[str(row["experiment"])] = str(row["experiment_title"])
    standard = baseline_map(all_rows)

    analysis_lines = [
        "补充实验设计、结果与分析",
        "",
        "统一设置：所有补充实验均只运行中尺度，使用 run_final_outputs.py 中尺度固定测试种子 "
        f"{', '.join(str(seed) for seed in FIXED_SEEDS['medium'])}。每个场景下 MAPPO 均单独训练 200 轮后参与同种子评估。",
        "",
    ]
    for experiment_name, rows in by_experiment.items():
        best, riskiest = describe_best_and_risk(rows)
        analysis_lines.extend(
            [
                f"{titles[experiment_name]}（{experiment_name}）",
                f"结果表：{experiment_name}.md",
                (
                    f"最优策略为 {DISPLAY_NAMES.get(str(best['strategy']), str(best['strategy']))}，"
                    f"平均得分 {float(best['total_score_mean']):.3f}，"
                    f"完成率 {float(best['completion_rate_mean']):.4f}，"
                    f"超时率 {float(best['timeout_rate_mean']):.4f}，"
                    f"失败率 {float(best['failure_rate']):.4f}。"
                ),
                (
                    f"风险最高策略为 {DISPLAY_NAMES.get(str(riskiest['strategy']), str(riskiest['strategy']))}，"
                    f"失败率 {float(riskiest['failure_rate']):.4f}，"
                    f"超时率 {float(riskiest['timeout_rate_mean']):.4f}。"
                ),
            ]
        )
        if experiment_name != "standard_baseline" and standard:
            comparable = [
                (row, standard.get(str(row["strategy"])))
                for row in rows
                if standard.get(str(row["strategy"])) is not None
            ]
            if comparable:
                avg_score_delta = mean(
                    float(row["total_score_mean"]) - float(base["total_score_mean"])
                    for row, base in comparable
                )
                avg_completion_delta = mean(
                    float(row["completion_rate_mean"]) - float(base["completion_rate_mean"])
                    for row, base in comparable
                )
                analysis_lines.append(
                    f"相对普通基准，所有策略平均得分变化 {avg_score_delta:.3f}，"
                    f"平均完成率变化 {avg_completion_delta:.4f}。"
                )
        analysis_lines.append("")

    analysis_lines.extend(
        [
            "MAPPO 训练检查",
            (
                "上一轮 100 轮实验中，MAPPO 的训练日志存在明显不稳定现象："
                "部分场景后段 episode 的完成任务数显著低于前段，评估结果也出现接单不足。"
                "这说明当前 MAPPO 训练过程对场景和随机种子较敏感，best_medium.pt 按单个训练 episode 得分选择，"
                "未必等价于固定五个测试种子的泛化最优。本轮改为 200 轮并保留每个场景 train_log.csv 与 best_medium.pt，"
                "用于在报告中说明学习型策略受训练稳定性影响较大。"
            ),
            "",
        ]
    )
    (output_dir / "supplementary_experiment_analysis.txt").write_text(
        "\n".join(analysis_lines),
        encoding="utf-8",
    )

    insertion_lines = [
        "DS 报告插入建议",
        "",
        "1. 实验设置/仿真参数部分",
        "插入一段说明：为检验调度策略在非标准运行环境下的鲁棒性，新增普通中尺度基准和六组中尺度补充实验，所有实验复用最终实验的中尺度固定随机种子，并保持路网、车辆数、充电站数等基础设置一致，仅改变任务负载、任务释放时间分布、目的地空间热点、行驶时间扰动或单位距离耗电速率。",
        "",
        "2. 算法对比实验部分",
        "在主对比表之后先插入 standard_baseline.md 作为普通中尺度基准，再插入 extreme_load.md、peak_gaussian.md、correlated_hotspot.md、random_traffic.md、energy_rate_0_5.md、energy_rate_2_0.md 中的聚合结果表和与普通情况对比表。",
        "",
        "3. MAPPO 方法或实验实现部分",
        "补充说明：MAPPO 在每个场景下均只针对中尺度重新训练 200 轮，然后在相同五个测试种子上与启发式/元启发式策略进行公平评估；同时说明上一轮 100 轮训练中观察到训练不稳定，因此本轮增加训练轮数并保留训练日志。",
        "",
        "4. 结果分析部分",
        "加入跨场景讨论：极大负载和峰值释放主要考察任务压力，相关性测试考察时间峰值与空间热点叠加，随机堵车考察旅行时间不确定性，耗电速率 0.5 与 2.0 分别反映电量约束放松和收紧。建议围绕得分、完成率、超时率、总里程和失败率解释策略差异。",
        "",
        "5. 结论部分",
        "增加一句总结：补充实验表明，策略性能不仅取决于平均任务规模，也会受到任务到达峰值、路况扰动和能耗约束的显著影响，因此鲁棒性评估是新能源物流车队调度方案选择的重要依据。",
    ]
    (output_dir / "ds_report_insertion_suggestions.txt").write_text(
        "\n".join(insertion_lines),
        encoding="utf-8",
    )


def run_supplementary_experiments(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for experiment in SUPPLEMENTARY_EXPERIMENTS:
        print(f"Supplementary experiment started: {experiment.name} - {experiment.title}")
        all_rows.extend(
            run_one_supplementary_experiment(
                experiment=experiment,
                output_dir=output_dir,
                args=args,
                baseline_rows=baseline_map(all_rows),
            )
        )
        write_summary_csv(output_dir / "supplementary_summary_aggregated.csv", all_rows)

    write_summary_csv(output_dir / "supplementary_summary_aggregated.csv", all_rows)
    write_supplementary_texts(output_dir, all_rows)

    print("Supplementary experiments complete:")
    for experiment_name in sorted({str(row["experiment"]) for row in all_rows}):
        rows = [row for row in all_rows if row["experiment"] == experiment_name]
        best, _ = describe_best_and_risk(rows)
        print(
            f"- {experiment_name}: best={best['strategy']}, "
            f"score={best['total_score_mean']}, completion={best['completion_rate_mean']}"
        )


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



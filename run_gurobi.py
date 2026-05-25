from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re
from typing import Optional

from src.config import SimulationConfig, default_scales
from src.gurobi_solver import (
    GurobiSolveOptions,
    average_summary,
    build_round_seeds,
    load_distribution_payload,
    print_result_to_terminal,
    scale_from_distribution,
    solve_static_gurobi,
    write_gurobi_replay_json,
    write_markdown_report,
    write_rows_csv,
    write_solution_json,
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gurobi static full-information solver")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large", "all"],
        default="small",
        help="Problem scale to solve.",
    )
    parser.add_argument(
        "--objective",
        choices=["score", "multi", "all"],
        default="all",
        help="score=max project score, multi=max completed then min overdue then max score.",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of generated task sequences.")
    parser.add_argument(
        "--task-source",
        choices=["distribution", "generator"],
        default="distribution",
        help="distribution reads existing outputs/distribution files; generator regenerates tasks by seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed generator seed. Passing this switches --task-source to generator unless a distribution file is used.",
    )
    parser.add_argument(
        "--seed-map",
        type=str,
        default=None,
        help="Scale-specific generator seeds, e.g. small=1,medium=2,large=3. Passing this switches to generator.",
    )
    parser.add_argument(
        "--distribution-dir",
        type=Path,
        default=Path("outputs") / "distribution",
        help="Directory containing *_task_distribution.json files.",
    )
    parser.add_argument(
        "--distribution-strategy",
        type=str,
        default="hyper_selector",
        help="Strategy name used in distribution filenames, e.g. hyper_selector.",
    )
    parser.add_argument(
        "--distribution-file",
        type=Path,
        default=None,
        help="Use one explicit distribution json file instead of auto-discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "gurobi",
        help="Directory for Gurobi outputs.",
    )
    parser.add_argument(
        "--slot-mode",
        choices=["full", "balanced"],
        default="balanced",
        help="balanced gives each vehicle ceil(tasks/vehicles)+buffer slots; full uses a physical upper bound.",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Override route slots per vehicle.",
    )
    parser.add_argument(
        "--slot-buffer",
        type=int,
        default=8,
        help="Extra slots per vehicle when --slot-mode balanced is used.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional debug time limit in seconds. Omit for strict solve.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional debug MIP gap. Omit for strict solve.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Optional Gurobi thread count.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable Gurobi solver log on console.",
    )
    return parser.parse_args()


def parse_seed_map(raw: Optional[str]) -> dict[str, int]:
    if not raw:
        return {}

    allowed_scales = {"small", "medium", "large"}
    parsed: dict[str, int] = {}
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" in entry:
            scale_name, seed_text = entry.split("=", 1)
        elif ":" in entry:
            scale_name, seed_text = entry.split(":", 1)
        else:
            raise ValueError(
                "--seed-map entries must look like small=123 or small:123; "
                f"got {entry!r}."
            )
        scale_name = scale_name.strip()
        seed_text = seed_text.strip()
        if scale_name not in allowed_scales:
            raise ValueError(f"--seed-map contains unknown scale {scale_name!r}.")
        if scale_name in parsed:
            raise ValueError(f"--seed-map contains duplicate scale {scale_name!r}.")
        try:
            parsed[scale_name] = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"--seed-map seed for {scale_name!r} is not an integer: {seed_text!r}") from exc
    return parsed


def main() -> None:
    args = parse_args()
    log("[gurobi] run_gurobi.py started")
    config = SimulationConfig()
    scales = default_scales()
    if args.scale != "all":
        scales = [scale for scale in scales if scale.name == args.scale]

    objective_modes = ["score", "multi"] if args.objective == "all" else [args.objective]
    seed_map = parse_seed_map(args.seed_map)
    if args.seed is not None and seed_map:
        raise ValueError("--seed and --seed-map cannot be used together.")
    if (args.seed is not None or seed_map) and args.distribution_file is not None:
        raise ValueError("--seed/--seed-map use the generator and cannot be combined with --distribution-file.")
    if (args.seed is not None or seed_map) and args.task_source == "distribution":
        args.task_source = "generator"
        log("[gurobi] --seed/--seed-map detected; switching task source to generator")
    if seed_map:
        selected_names = {scale.name for scale in scales}
        missing = sorted(selected_names - set(seed_map))
        if missing:
            raise ValueError(f"--seed-map is missing selected scale(s): {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    grouped_results = {}

    for scale in scales:
        round_count = max(1, args.rounds)
        distribution_files = []
        if args.task_source == "distribution":
            distribution_files = discover_distribution_files(
                scale_name=scale.name,
                strategy_name=args.distribution_strategy,
                distribution_dir=args.distribution_dir,
                explicit_file=args.distribution_file,
                rounds=round_count,
            )
            log("[gurobi] using distribution files:")
            for path in distribution_files:
                payload = load_distribution_payload(path)
                log(
                    f"  - {path} "
                    f"(seed={payload.get('seed')}, horizon={payload.get('horizon')}, "
                    f"tasks={payload.get('task_count')})"
                )
        base_seed = seed_map.get(scale.name, args.seed)
        seed_scale = replace(scale, seed=int(base_seed)) if base_seed is not None else scale
        seeds = build_round_seeds(seed_scale, round_count)
        for objective_mode in objective_modes:
            group_key = f"{scale.name}_{objective_mode}"
            grouped_results[group_key] = []

            round_inputs = distribution_files if args.task_source == "distribution" else seeds
            for round_index, round_input in enumerate(round_inputs, start=1):
                distribution_path = round_input if args.task_source == "distribution" else None
                if distribution_path is not None:
                    round_scale = scale_from_distribution(scale, distribution_path)
                    seed = round_scale.seed
                else:
                    seed = int(round_input)
                    round_scale = replace(scale, seed=seed)
                log(
                    f"[gurobi] scale={scale.name} objective={objective_mode} "
                    f"round={round_index}/{len(round_inputs)} seed={seed} "
                    f"slot_mode={args.slot_mode}"
                )
                options = GurobiSolveOptions(
                    objective_mode=objective_mode,
                    max_slots=args.max_slots,
                    slot_mode=args.slot_mode,
                    slot_buffer=args.slot_buffer,
                    time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                    log_to_console=not args.quiet,
                    log_file=(
                        args.output_dir
                        / "logs"
                        / f"{scale.name}_{objective_mode}_r{round_index:02d}_seed{seed}.log"
                    ),
                    distribution_path=distribution_path,
                    threads=args.threads,
                )
                log("[gurobi] building model and starting optimizer...")
                result = solve_static_gurobi(
                    scale=round_scale,
                    config=config,
                    objective_mode=objective_mode,
                    round_index=round_index,
                    options=options,
                )
                grouped_results[group_key].append(result)
                all_rows.append(result.summary_row())

                solution_path = (
                    args.output_dir
                    / f"{scale.name}_{objective_mode}_r{round_index:02d}_seed{seed}_solution.json"
                )
                write_solution_json(solution_path, result)
                replay_path = (
                    args.output_dir
                    / "replay"
                    / f"{scale.name}_{objective_mode}_r{round_index:02d}_seed{seed}_replay.json"
                )
                write_gurobi_replay_json(replay_path, result)
                print_result_to_terminal(result)
                log(f"[gurobi] solution_json={solution_path}")
                log(f"[gurobi] replay_json={replay_path}")

            mode_rows = [result.summary_row() for result in grouped_results[group_key]]
            write_rows_csv(args.output_dir / f"{scale.name}_{objective_mode}_rounds.csv", mode_rows)
            avg = average_summary(grouped_results[group_key])
            write_rows_csv(args.output_dir / f"{scale.name}_{objective_mode}_average.csv", [avg])

    write_rows_csv(args.output_dir / "summary.csv", all_rows)
    write_rows_csv(args.output_dir / "summary_original_compatible.csv", to_original_summary_rows(all_rows))
    write_markdown_report(args.output_dir / "report.md", grouped_results)
    log(f"[gurobi] wrote {args.output_dir / 'summary.csv'}")
    log(f"[gurobi] wrote {args.output_dir / 'summary_original_compatible.csv'}")
    log(f"[gurobi] wrote {args.output_dir / 'report.md'}")


def to_original_summary_rows(rows: list[dict]) -> list[dict]:
    compatible_rows: list[dict] = []
    for row in rows:
        compatible_rows.append(
            {
                "scale": row["scale"],
                "strategy": f"gurobi_{row['objective_mode']}",
                "round": row["round"],
                "seed": row["seed"],
                "total_score": row["total_score"],
                "completed_tasks": row["completed_tasks"],
                "total_tasks": row["total_tasks"],
                "overdue_tasks": row["overdue_tasks"],
                "timeout_rate": row["timeout_rate"],
                "total_distance": row["total_distance"],
                "simulation_failed": row["status"] in {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "NUMERIC"},
                "failure_tick": "",
                "failure_reason": "" if row["status"] in {"OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"} else row["status"],
                "completion_rate": row["completion_rate"],
            }
        )
    return compatible_rows


def discover_distribution_files(
    *,
    scale_name: str,
    strategy_name: str,
    distribution_dir: Path,
    explicit_file: Optional[Path],
    rounds: int,
) -> list[Path]:
    if explicit_file is not None:
        if not explicit_file.exists():
            raise FileNotFoundError(f"指定的 distribution 文件不存在: {explicit_file}")
        return [explicit_file]

    if not distribution_dir.exists():
        raise FileNotFoundError(f"distribution 目录不存在: {distribution_dir}")

    pattern = f"{scale_name}_{strategy_name}*_task_distribution.json"
    candidates = sorted(distribution_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"没有找到 distribution 文件: {distribution_dir / pattern}。"
            "如果要重新生成任务，请改用 --task-source generator。"
        )

    round_re = re.compile(r"_r(\d+)_seed")
    grouped: dict[int, list[Path]] = {}
    base_files: list[Path] = []
    for path in candidates:
        match = round_re.search(path.name)
        if match:
            grouped.setdefault(int(match.group(1)), []).append(path)
        else:
            base_files.append(path)

    chosen: list[Path] = []
    for round_no in sorted(grouped):
        newest = max(grouped[round_no], key=lambda item: item.stat().st_mtime)
        chosen.append(newest)
        if len(chosen) >= rounds:
            return chosen

    for path in sorted(base_files, key=lambda item: item.stat().st_mtime, reverse=True):
        chosen.append(path)
        if len(chosen) >= rounds:
            return chosen

    if len(chosen) < rounds:
        log(f"[gurobi] warning: 只找到 {len(chosen)} 个 distribution 文件，少于请求 rounds={rounds}")
    return chosen


if __name__ == "__main__":
    main()

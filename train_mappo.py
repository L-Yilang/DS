from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean

from src.config import SimulationConfig, default_scales
from src.mappo.config import build_preset_config
from src.mappo.trainer import MAPPOTrainer


def select_scales(scale_name: str):
    scales = {scale.name: scale for scale in default_scales()}
    if scale_name == "all":
        return scales
    if scale_name not in scales:
        valid = ", ".join(["all", *scales.keys()])
        raise ValueError(f"Unknown scale: {scale_name}. Valid values: {valid}")
    return {scale_name: scales[scale_name]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO macro-action policy")
    parser.add_argument(
        "--scale",
        choices=["all", "small", "medium", "large"],
        default="small",
        help="Scale/stage to train on. Use all for small->medium->large.",
    )
    parser.add_argument(
        "--preset",
        choices=["smoke", "default"],
        default="smoke",
        help="Training preset. smoke is a quick integration test; default is longer.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, e.g. cpu or cuda. Falls back to cpu if cuda is unavailable.",
    )
    parser.add_argument(
        "--episodes-per-stage",
        type=int,
        default=None,
        help="Override default preset episodes per stage.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/mappo/checkpoints"),
        help="Directory for best/latest MAPPO checkpoints.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("outputs/mappo"),
        help="Directory for MAPPO training logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260525,
        help="Training random seed.",
    )
    args = parser.parse_args()

    mappo_config = build_preset_config(args.preset, device=args.device)
    if args.episodes_per_stage is not None:
        if args.preset == "smoke":
            mappo_config = replace(
                mappo_config,
                smoke_episodes_per_stage=max(1, args.episodes_per_stage),
            )
        else:
            mappo_config = replace(
                mappo_config,
                episodes_per_stage=max(1, args.episodes_per_stage),
            )
    mappo_config = replace(
        mappo_config,
        checkpoint_dir=str(args.checkpoint_dir),
        eval_dir=str(args.eval_dir),
        seed=args.seed,
    )

    selected_scales = select_scales(args.scale)
    trainer = MAPPOTrainer(
        scales=selected_scales,
        sim_config=SimulationConfig(),
        mappo_config=mappo_config,
    )
    history = trainer.train(
        preset=args.preset,
        stage_names=list(selected_scales.keys()),
    )

    print("MAPPO training finished.")
    print(f"preset={args.preset}, scale={args.scale}, device={trainer.device}")
    print(f"episodes={len(history)}")
    if history:
        scores = [float(row["total_score"]) for row in history]
        completion = [float(row["completed_tasks"]) for row in history]
        print(f"score_mean={mean(scores):.3f}, score_best={max(scores):.3f}")
        print(f"completed_mean={mean(completion):.3f}")
        last = history[-1]
        print(
            "last_episode="
            f"stage={last['stage']}, episode={last['episode']}, "
            f"score={last['total_score']}, failed={last['simulation_failed']}"
        )
    print(f"checkpoints: {args.checkpoint_dir}")
    print(f"train_log: {args.eval_dir / 'train_log.csv'}")


if __name__ == "__main__":
    main()

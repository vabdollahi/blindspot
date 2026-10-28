"""Evaluate the frozen-backbone SmolVLM2 VLA baseline on T3-L instructions."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.runner import evaluate
from eval.vla_profile import evaluate_limited_vla
from policies.vla import load_vla_policy


def main() -> None:
    """Run seeded T3-L episodes and write an evaluation JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument(
        "--scope",
        choices=("full", "limited"),
        default="full",
        help="Evaluate the full T3-L distribution or the limited training slice.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help="Maximum evaluation seeds to scan for limited-scope episodes.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--replan-interval",
        type=int,
        help="Steps between VLM queries for chunked heads (default: 2).",
    )
    parser.add_argument(
        "--ensemble-decay",
        type=float,
        default=0.0,
        help="Weight exp(-decay * age) for overlapping plans; 0 averages them equally.",
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--trajectories-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.episodes < 1:
        parser.error("--episodes must be positive")
    if arguments.seed_start < 0:
        parser.error("--seed-start must be non-negative")
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.scope == "limited":
        if arguments.workers != 1:
            parser.error("Limited-scope VLA evaluation requires --workers 1")
        if arguments.max_candidates < arguments.episodes:
            parser.error("--max-candidates must be at least --episodes")
    elif arguments.trajectories_dir is not None:
        parser.error("--trajectories-dir is available only with --scope limited")

    checkpoint = arguments.checkpoint.expanduser().resolve()
    policy = load_vla_policy(
        checkpoint,
        device=arguments.device,
        replan_interval=arguments.replan_interval,
        ensemble_decay=arguments.ensemble_decay,
    )
    if arguments.scope == "limited":
        if policy.training_profile != "limited":
            parser.error("--scope limited requires a limited-profile checkpoint")
        result, result_path, trajectory_root, candidates_scanned = evaluate_limited_vla(
            policy,
            checkpoint,
            arguments.level,
            arguments.episodes,
            arguments.seed_start,
            arguments.max_candidates,
            results_dir=arguments.results_dir,
            trajectories_dir=arguments.trajectories_dir,
        )
        print(
            f"Matched {len(result.seeds)} episodes after scanning "
            f"{candidates_scanned} candidate seeds."
        )
        print(f"Trajectories: {trajectory_root}")
    else:
        seeds = range(
            arguments.seed_start,
            arguments.seed_start + arguments.episodes,
        )
        method = (
            "vla-prototype-limited"
            if policy.training_profile == "limited"
            else "vla-prototype"
        )
        if policy.chunk_size > 1:
            method += (
                f"-chunk{policy.chunk_size}-replan{policy.replan_interval}"
                f"-decay{policy.ensemble_decay:g}"
            )
        result = evaluate(
            policy,
            "T3-L",
            arguments.level,
            seeds,
            obs_mode="image",
            image_size=256,
            workers=arguments.workers,
            method=method,
            results_dir=arguments.results_dir,
        )
        results_root = (
            arguments.results_dir
            if arguments.results_dir is not None
            else Path(__file__).resolve().parents[1] / "eval" / "results"
        )
        result_path = results_root / result.method / f"T3-L-{result.level}.json"

    lower, upper = result.summary.wilson_95
    print(
        f"{result.method} T3-L-{result.level}: "
        f"{result.summary.successes}/{result.summary.episodes} "
        f"({result.summary.success_rate:.1%}; "
        f"95% CI {lower:.1%} to {upper:.1%})"
    )
    print(result_path)


if __name__ == "__main__":
    main()

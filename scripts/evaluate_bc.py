"""Evaluate a trained behavior-cloning checkpoint on fixed task seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.runner import evaluate
from policies.bc import ImageBCPolicy, load_bc_policy


def _parse_args() -> argparse.Namespace:
    """Parse checkpoint evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=("T2", "T3"), default="T2")
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device; CPU keeps policies portable to spawned workers.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the BC policy through the shared evaluation harness."""
    arguments = _parse_args()
    if arguments.episodes < 1:
        raise SystemExit("--episodes must be positive")
    if arguments.seed_start < 0:
        raise SystemExit("--seed-start must be non-negative")
    if not arguments.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {arguments.checkpoint}")

    policy = load_bc_policy(arguments.checkpoint, device=arguments.device)
    obs_mode = "image" if isinstance(policy, ImageBCPolicy) else "state"
    method = "image-bc" if obs_mode == "image" else "state-bc"
    result = evaluate(
        policy,
        arguments.task,
        arguments.level,
        list(range(arguments.seed_start, arguments.seed_start + arguments.episodes)),
        obs_mode=obs_mode,
        workers=arguments.workers,
        method=method,
        results_dir=arguments.results_dir,
    )
    lower, upper = result.summary.wilson_95
    print(
        f"{method} {result.task}-{result.level}: "
        f"{result.summary.successes}/{result.summary.episodes} "
        f"({result.summary.success_rate:.1%}; 95% CI "
        f"{lower:.1%} to {upper:.1%})"
    )
    results_root = (
        arguments.results_dir
        if arguments.results_dir is not None
        else Path(__file__).resolve().parents[1] / "eval" / "results"
    )
    print(results_root / method / f"{result.task}-{result.level}.json")


if __name__ == "__main__":
    main()

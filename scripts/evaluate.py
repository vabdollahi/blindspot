"""Evaluate the scripted expert on reproducible task and level seeds."""

import argparse
from pathlib import Path

from eval.runner import evaluate
from skills.expert import ScriptedExpert


def main() -> None:
    """Run the scripted expert and print its generated confidence interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("T1", "T2", "T3", "T3-L", "T4"), required=True
    )
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--obs-mode", choices=("state", "image"), default="state")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--results-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.episodes < 1:
        parser.error("--episodes must be positive")
    if arguments.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    policy = ScriptedExpert(arguments.task)
    seeds = range(arguments.seed_start, arguments.seed_start + arguments.episodes)
    result = evaluate(
        policy,
        arguments.task,
        arguments.level,
        seeds,
        obs_mode=arguments.obs_mode,
        workers=arguments.workers,
        results_dir=arguments.results_dir,
    )
    lower, upper = result.summary.wilson_95
    print(
        f"{result.method} {result.task}-{result.level}: "
        f"{result.summary.successes}/{result.summary.episodes} "
        f"({result.summary.success_rate:.1%}; 95% CI {lower:.1%} to {upper:.1%})"
    )
    results_root = (
        arguments.results_dir
        if arguments.results_dir is not None
        else Path(__file__).resolve().parents[1] / "eval" / "results"
    )
    print(results_root / result.method / f"{result.task}-{result.level}.json")


if __name__ == "__main__":
    main()

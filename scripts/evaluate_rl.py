"""Evaluate a trained Phase 1b RL checkpoint on fixed task seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from envs.rl_env import RL_TASKS
from eval.runner import evaluate
from policies.rl import load_rl_policy


def _parse_args() -> argparse.Namespace:
    """Parse checkpoint evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--task", choices=(*RL_TASKS, "T2-H"), default="T1")
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Run the frozen RL checkpoint through the shared evaluation harness."""
    arguments = _parse_args()
    if arguments.episodes < 1:
        raise SystemExit("--episodes must be positive")
    if arguments.seed_start < 0:
        raise SystemExit("--seed-start must be non-negative")
    if arguments.task == "T2-H" and arguments.level != "L1":
        raise SystemExit("T2-H evaluation is defined only at level L1")
    if not (arguments.checkpoint_dir / "model.zip").is_file():
        raise SystemExit(f"No model.zip found under {arguments.checkpoint_dir}")

    policy = load_rl_policy(arguments.checkpoint_dir, device=arguments.device)
    result = evaluate(
        policy,
        arguments.task,
        arguments.level,
        list(range(arguments.seed_start, arguments.seed_start + arguments.episodes)),
        obs_mode="state",
        workers=arguments.workers,
        method="pure-ppo",
        results_dir=arguments.results_dir,
    )
    lower, upper = result.summary.wilson_95
    print(
        f"pure-ppo {result.task}-{result.level}: "
        f"{result.summary.successes}/{result.summary.episodes} "
        f"({result.summary.success_rate:.1%}; 95% CI "
        f"{lower:.1%} to {upper:.1%})"
    )
    if result.summary.failure_breakdown:
        print("Failure breakdown:", result.summary.failure_breakdown)
    results_root = (
        arguments.results_dir
        if arguments.results_dir is not None
        else Path(__file__).resolve().parents[1] / "eval" / "results"
    )
    print(results_root / "pure-ppo" / f"{result.task}-{result.level}.json")


if __name__ == "__main__":
    main()

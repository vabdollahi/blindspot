"""Evaluate the scripted expert across the Phase 1.8 task/level matrix."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.runner import evaluate
from eval.schema import EvaluationResult
from skills.expert import ScriptedExpert

TASKS = ("T1", "T2", "T3", "T4")
LEVELS = ("L0", "L1", "L2")
EVALUATION_SEED_START = 1_000_000
SUCCESS_THRESHOLDS = {
    ("T1", "L0"): 0.99,
    ("T1", "L1"): 0.99,
    ("T1", "L2"): 0.99,
    ("T2", "L1"): 0.90,
    ("T3", "L1"): 0.90,
    ("T4", "L1"): 0.70,
}


def evaluate_baseline(
    tasks: Sequence[str],
    levels: Sequence[str],
    *,
    episodes: int = 100,
    seed_start: int = EVALUATION_SEED_START,
    workers: int | None = None,
    results_dir: Path | None = None,
) -> tuple[list[EvaluationResult], bool]:
    """Evaluate each requested cell on the same contiguous evaluation seeds."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a non-empty sequence without duplicates")
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("levels must be a non-empty sequence without duplicates")
    if any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be selected from {TASKS}")
    if any(level not in LEVELS for level in levels):
        raise ValueError(f"levels must be selected from {LEVELS}")

    seeds = list(range(seed_start, seed_start + episodes))
    results: list[EvaluationResult] = []
    for task in tasks:
        for level in levels:
            result = evaluate(
                ScriptedExpert(task),
                task,
                level,
                seeds,
                workers=workers,
                method="scripted-expert",
                results_dir=results_dir,
            )
            results.append(result)
            lower, upper = result.summary.wilson_95
            failures = (
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(
                        result.summary.failure_breakdown.items()
                    )
                )
                or "none"
            )
            print(
                f"{task}-{level}: {result.summary.successes}/{episodes} "
                f"({result.summary.success_rate:.1%}; 95% CI "
                f"{lower:.1%} to {upper:.1%}); failures: {failures}"
            )

    rates = {
        (result.task, result.level): result.summary.success_rate for result in results
    }
    passed = True
    for cell, threshold in SUCCESS_THRESHOLDS.items():
        if cell not in rates:
            continue
        rate = rates[cell]
        cell_passed = rate >= threshold
        passed = passed and cell_passed
        state = "PASS" if cell_passed else "FAIL"
        print(f"Gate {cell[0]}-{cell[1]} >= {threshold:.0%}: {state} ({rate:.1%})")
    return results, passed


def main() -> None:
    """Run the requested baseline cells and return nonzero when an evaluated gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=EVALUATION_SEED_START)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--results-dir", type=Path)
    arguments = parser.parse_args()

    try:
        _, passed = evaluate_baseline(
            arguments.tasks,
            arguments.levels,
            episodes=arguments.episodes,
            seed_start=arguments.seed_start,
            workers=arguments.workers,
            results_dir=arguments.results_dir,
        )
    except ValueError as error:
        parser.error(str(error))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

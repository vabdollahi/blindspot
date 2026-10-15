from pathlib import Path
from typing import Any

import pytest

import scripts.evaluate_baseline as baseline
from eval.schema import EpisodeResult, EvaluationResult, EvaluationSummary


def _evaluation_result(
    task: str,
    level: str,
    success_rate: float,
    seeds: list[int],
) -> EvaluationResult:
    """Build a small validated result for baseline matrix tests."""
    successes = round(success_rate * len(seeds))
    episodes = [
        EpisodeResult(
            seed=seed,
            success=index < successes,
            terminated=index < successes,
            truncated=False,
            failure_reason=None if index < successes else "timeout",
            episode_steps=10,
            episode_time=0.5,
            collision_count=0,
            total_reward=1.0 if index < successes else 0.0,
        )
        for index, seed in enumerate(seeds)
    ]
    return EvaluationResult(
        method="scripted-expert",
        task=task,
        level=level,
        obs_mode="state",
        seeds=seeds,
        git_commit="test",
        config_hash="0" * 64,
        summary=EvaluationSummary(
            episodes=len(seeds),
            successes=successes,
            success_rate=successes / len(seeds),
            wilson_95=(0.0, 1.0),
            failure_breakdown={},
        ),
        episodes=episodes,
    )


def test_baseline_runs_the_full_matrix_with_shared_evaluation_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run T1-T4 at all three levels on one common seed list."""
    calls: list[tuple[str, str, list[int]]] = []

    def fake_evaluate(
        policy: Any,
        task: str,
        level: str,
        seeds: list[int],
        **_: Any,
    ) -> EvaluationResult:
        calls.append((task, level, seeds.copy()))
        return _evaluation_result(task, level, 1.0, seeds)

    monkeypatch.setattr(baseline, "evaluate", fake_evaluate)

    results, passed = baseline.evaluate_baseline(
        baseline.TASKS,
        baseline.LEVELS,
        episodes=3,
        seed_start=1_000_100,
        results_dir=tmp_path,
    )

    assert len(results) == 12
    assert len(calls) == 12
    assert {(task, level) for task, level, _ in calls} == {
        (task, level) for task in baseline.TASKS for level in baseline.LEVELS
    }
    assert all(seeds == [1_000_100, 1_000_101, 1_000_102] for _, _, seeds in calls)
    assert passed


def test_baseline_gate_fails_when_an_l1_threshold_is_missed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return a failing gate status when T4 misses its documented L1 target."""

    def fake_evaluate(
        policy: Any,
        task: str,
        level: str,
        seeds: list[int],
        **_: Any,
    ) -> EvaluationResult:
        success_rate = 0.5 if (task, level) == ("T4", "L1") else 1.0
        return _evaluation_result(task, level, success_rate, seeds)

    monkeypatch.setattr(baseline, "evaluate", fake_evaluate)

    _, passed = baseline.evaluate_baseline(
        ("T4",),
        ("L1",),
        episodes=10,
        results_dir=tmp_path,
    )

    assert not passed

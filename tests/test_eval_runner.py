import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.report import build_report
from eval.runner import _config_hash, _seed_policy_rngs, evaluate
from skills.expert import ScriptedExpert


class OSErrorPolicy:
    """Policy fixture that simulates a per-episode model or I/O failure."""

    def reset(self, instruction: str | None = None) -> None:
        """Do nothing before the test episode."""

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Raise a common I/O exception while producing an action."""
        raise OSError("forced policy I/O failure")


class CustomPolicyError(Exception):
    """Unexpected policy exception used to verify episode isolation."""


class CustomExceptionPolicy:
    """Policy fixture that simulates an arbitrary user-defined failure."""

    def reset(self, instruction: str | None = None) -> None:
        """Do nothing before the test episode."""

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Raise a custom exception while producing an action."""
        raise CustomPolicyError("forced custom policy failure")


def test_policy_rngs_reseed_to_the_same_named_stream() -> None:
    """Recreate common policy RNG draws independently for the same seed."""
    _seed_policy_rngs(42)
    first = (random.random(), np.random.random())
    _seed_policy_rngs(42)
    second = (random.random(), np.random.random())

    assert first == second


def test_policy_oserror_is_saved_as_an_episode_failure(tmp_path: Path) -> None:
    """Keep one policy I/O error in the JSON results instead of aborting a worker."""
    result = evaluate(
        OSErrorPolicy(),
        "T1",
        "L0",
        [1_000_003],
        workers=1,
        results_dir=tmp_path,
    )

    assert not result.episodes[0].success
    assert result.episodes[0].failure_reason == "policy_error"
    assert result.episodes[0].error == "OSError: forced policy I/O failure"


def test_custom_policy_exception_is_saved_as_an_episode_failure(
    tmp_path: Path,
) -> None:
    """Isolate a user-defined exception to its episode in the results."""
    result = evaluate(
        CustomExceptionPolicy(),
        "T1",
        "L0",
        [1_000_004],
        workers=1,
        results_dir=tmp_path,
    )

    assert not result.episodes[0].success
    assert result.episodes[0].failure_reason == "policy_error"
    assert result.episodes[0].error == "CustomPolicyError: forced custom policy failure"


def test_t2h_is_available_through_the_shared_evaluation_harness(
    tmp_path: Path,
) -> None:
    """Create a named T2-H evaluation result using its dedicated L1 environment."""
    result = evaluate(
        OSErrorPolicy(),
        "T2-H",
        "L1",
        [1_000_005],
        workers=1,
        method="rl-t2-checkpoint",
        results_dir=tmp_path,
    )

    assert result.task == "T2-H"
    assert result.level == "L1"
    assert result.episodes[0].failure_reason == "policy_error"
    assert (tmp_path / "rl-t2-checkpoint" / "T2-H-L1.json").is_file()


def test_evaluation_image_size_is_validated_and_included_in_config_hash() -> None:
    """Distinguish VLA resolution from the default BC image resolution."""
    assert _config_hash("T3-L", "L1", "image", image_size=256) != _config_hash(
        "T3-L", "L1", "image", image_size=128
    )
    with pytest.raises(ValueError, match="only when obs_mode='image'"):
        evaluate(OSErrorPolicy(), "T3-L", "L1", [1_000_006], image_size=256)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate(
            OSErrorPolicy(),
            "T3-L",
            "L1",
            [1_000_006],
            obs_mode="image",
            image_size=0,
        )


def test_parallel_evaluation_writes_reproducible_episode_json(
    tmp_path: Path,
) -> None:
    """Reproduce identical T1 records across two spawned two-worker runs."""
    policy = ScriptedExpert("T1")
    seeds = [1_000_001, 1_000_002]

    first = evaluate(
        policy,
        "T1",
        "L0",
        seeds,
        workers=2,
        results_dir=tmp_path / "first",
    )
    second = evaluate(
        policy,
        "T1",
        "L0",
        seeds,
        workers=2,
        results_dir=tmp_path / "second",
    )

    assert first.json_dict() == second.json_dict()
    assert first.summary.episodes == 2
    assert first.summary.successes == 2
    assert len(first.config_hash) == 64
    for episode in first.episodes:
        assert episode.terminated
        assert not episode.truncated
        assert episode.episode_steps > 0
        assert episode.episode_time > 0.0
        assert episode.failure_reason is None

    first_file = tmp_path / "first" / "scripted-expert" / "T1-L0.json"
    second_file = tmp_path / "second" / "scripted-expert" / "T1-L0.json"
    assert json.loads(first_file.read_text(encoding="utf-8")) == first.json_dict()
    assert first_file.read_text(encoding="utf-8") == second_file.read_text(
        encoding="utf-8"
    )

    markdown_path, figure_path = build_report(
        tmp_path / "first",
        output_dir=tmp_path / "report",
    )
    report_text = markdown_path.read_text(encoding="utf-8")
    assert "T1" in report_text
    assert "Wilson 95%" in report_text
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0

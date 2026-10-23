from collections import Counter
from pathlib import Path

import numpy as np

from data.interaction import (
    SOURCE_IDS,
    InteractionConfig,
    action_coverage_summary,
    build_episode_requests,
    interaction_dataset_features,
    interaction_task_label,
    plot_action_coverage,
)


def test_interaction_requests_balance_tasks_and_sources() -> None:
    """Schedule every task against each of the four requested data sources."""
    config = InteractionConfig(episodes=60, seed_start=10)
    experts = {task: Path(f"/checkpoints/{task}/best") for task in ("T1", "T2", "T3")}
    intermediate = {
        task: [Path(f"/checkpoints/{task}/step_100")]
        for task in ("T1", "T2", "T3")
    }
    all_checkpoints = [*experts.values()]
    all_checkpoints.extend(
        checkpoint
        for task_checkpoints in intermediate.values()
        for checkpoint in task_checkpoints
    )
    checkpoint_digests = {
        path: f"sha-{index}" for index, path in enumerate(all_checkpoints)
    }

    requests = build_episode_requests(
        config,
        experts,
        intermediate,
        checkpoint_digests,
    )

    assert len(requests) == 60
    assert [request.seed for request in requests] == list(range(10, 70))
    assert Counter(request.task for request in requests) == {
        "T1": 20,
        "T2": 20,
        "T3": 20,
    }
    assert Counter(request.requested_source for request in requests) == {
        "rl-expert": 15,
        "rl-expert-noisy": 15,
        "intermediate-ppo": 15,
        "scripted-exploration": 15,
    }
    assert all(
        request.checkpoint_dir is not None
        for request in requests
        if request.requested_source != "scripted-exploration"
    )
    assert requests[0].checkpoint_sha256 is not None


def test_interaction_schema_records_sources_and_next_step_success() -> None:
    """Include corpus labels alongside the shared future-policy features."""
    features = interaction_dataset_features()

    assert features["observation.images.front"]["shape"] == (3, 256, 256)
    assert features["observation.environment_state"]["shape"] == (28,)
    assert features["action.executed"]["shape"] == (5,)
    assert features["source_id"]["dtype"] == "int64"
    assert features["source_id"]["shape"] == (1,)
    assert features["next.success"]["dtype"] == "int64"
    assert features["next.success"]["shape"] == (1,)
    assert set(SOURCE_IDS) == {
        "rl-expert",
        "rl-expert-failure",
        "rl-expert-noisy",
        "intermediate-ppo",
        "scripted-exploration",
    }


def test_task_labels_cover_t1_t2_and_t3() -> None:
    """Store task meaning as natural language in the mixed-task corpus."""
    assert interaction_task_label("T1", "") == "Reach the target point."
    assert (
        interaction_task_label("T2", "cube_red")
        == "Place the red cube in the target zone."
    )
    assert (
        interaction_task_label("T3", "cube_blue")
        == "Place the blue cube in the bowl."
    )


def test_action_coverage_summary_reports_workspace_and_gripper_histograms() -> None:
    """Summarize low/high target regions and both open/closed gripper commands."""
    actions = [
        np.array([0.25, -0.30, 0.02, -np.pi / 2, 0.0]),
        np.array([0.70, 0.30, 0.45, np.pi / 2, 1.0]),
    ]

    summary = action_coverage_summary(actions)

    assert summary["executed_action_count"] == 2
    assert summary["gripper_state_counts"] == {"closed": 1, "open": 1}
    np.testing.assert_allclose(summary["workspace_span_fraction"], [1.0, 1.0, 1.0])


def test_action_coverage_histogram_is_saved_as_a_png(tmp_path: Path) -> None:
    """Render the coverage summary to a standalone, shareable figure."""
    actions = [
        np.array([0.25, -0.30, 0.02, -np.pi / 2, 0.0]),
        np.array([0.70, 0.30, 0.45, np.pi / 2, 1.0]),
    ]
    summary = action_coverage_summary(actions)
    output = plot_action_coverage(summary, tmp_path / "action-coverage.png")

    assert output.is_file()
    assert output.stat().st_size > 0

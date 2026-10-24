from pathlib import Path

import numpy as np
import pytest

from common.seeding import named_generators
from data import generate as generate_module
from data.generate import (
    DATASET_IMAGE_SIZE,
    CollectionConfig,
    _default_dataset_name,
    _make_frame,
    _publish_staged_dataset,
    _select_dataset_name,
    dataset_features,
    demo_noise_generator,
    episode_task,
    generate_dataset,
    maybe_perturb_action,
    perturb_action,
    validation_episode_count,
)
from envs.pick_place_env import PickPlaceEnv


def test_validation_split_uses_reserved_five_percent() -> None:
    """Reserve exactly 20 validation episodes from the default 400 demos."""
    assert validation_episode_count(400, 0.05) == 20
    assert validation_episode_count(2, 0.99) == 1


def test_automatic_dataset_names_avoid_overwriting_existing_runs(
    tmp_path: Path,
) -> None:
    """Choose the next run suffix while keeping explicit collisions an error."""
    (tmp_path / "t2-l1").mkdir()
    (tmp_path / "t2-l1-run-02").mkdir()

    assert (
        _select_dataset_name(
            tmp_path,
            "t2-l1",
            automatic_versioning=True,
        )
        == "t2-l1-run-03"
    )
    with pytest.raises(FileExistsError, match="t2-l1"):
        _select_dataset_name(
            tmp_path,
            "t2-l1",
            automatic_versioning=False,
        )


def test_dataset_publish_refuses_to_replace_existing_data(tmp_path: Path) -> None:
    """Keep existing dataset contents intact if the final name is already taken."""
    staging_root = tmp_path / ".staging"
    final_root = tmp_path / "t2-l1"
    staging_root.mkdir()
    final_root.mkdir()
    (staging_root / "new.json").write_text("new", encoding="utf-8")
    (final_root / "existing.json").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="appeared during collection"):
        _publish_staged_dataset(staging_root, final_root)

    assert (final_root / "existing.json").read_text(encoding="utf-8") == "existing"
    assert (staging_root / "new.json").read_text(encoding="utf-8") == "new"


def test_default_dataset_names_are_compact_and_distinguish_t3l() -> None:
    """Keep the language task distinct without a duplicated L token."""
    assert _default_dataset_name(CollectionConfig(task="T2")) == "t2-l1"
    assert _default_dataset_name(CollectionConfig(task="T3")) == "t3-l1"
    assert _default_dataset_name(CollectionConfig(task="T3-L")) == "t3l-l1"
    assert (
        _default_dataset_name(
            CollectionConfig(task="T3-L", instruction_template_split="held-out")
        )
        == "t3l-l1-held-out"
    )


def test_clean_action_label_is_not_modified_by_dart_noise() -> None:
    """Apply reproducible xyz/yaw perturbations without changing the gripper."""
    action = np.array([0.5, 0.0, 0.3, 0.1, 0.0], dtype=np.float64)
    first_random = demo_noise_generator(42)
    second_random = demo_noise_generator(42)

    first = perturb_action(action, first_random, position_std=0.01, yaw_std=0.05)
    second = perturb_action(action, second_random, position_std=0.01, yaw_std=0.05)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(action, [0.5, 0.0, 0.3, 0.1, 0.0])
    assert first[4] == action[4]
    assert not np.array_equal(first[:4], action[:4])


def test_sparse_noise_events_are_seeded_and_preserve_clean_steps() -> None:
    """Reproduce selected perturbations while retaining unperturbed expert steps."""
    assert CollectionConfig(task="T2").noise_probability == 0.01
    action = np.array([0.5, 0.0, 0.3, 0.1, 0.0], dtype=np.float64)
    first = maybe_perturb_action(
        action,
        demo_noise_generator(42),
        probability=1.0,
        position_std=0.01,
        yaw_std=np.deg2rad(3),
    )
    repeated = maybe_perturb_action(
        action,
        demo_noise_generator(42),
        probability=1.0,
        position_std=0.01,
        yaw_std=np.deg2rad(3),
    )
    other_seed = maybe_perturb_action(
        action,
        demo_noise_generator(43),
        probability=1.0,
        position_std=0.01,
        yaw_std=np.deg2rad(3),
    )
    clean = maybe_perturb_action(
        action,
        demo_noise_generator(42),
        probability=0.0,
        position_std=0.01,
        yaw_std=np.deg2rad(3),
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other_seed)
    np.testing.assert_array_equal(clean, action)


def test_demo_noise_stream_is_independent_of_environment_scene_stream() -> None:
    """Keep noise on the third child stream after scene and visual randomness."""
    scene_random = named_generators(42, ("scene", "visual"))["scene"]
    noise_random = demo_noise_generator(42)

    scene_draws = scene_random.integers(0, 2**32, size=16, dtype=np.uint32)
    noise_draws = noise_random.integers(0, 2**32, size=16, dtype=np.uint32)

    assert not np.array_equal(scene_draws, noise_draws)


@pytest.mark.parametrize("task", ["T2", "T3"])
def test_training_seeds_produce_distinct_l1_demo_scenes(task: str) -> None:
    """Vary object/goal poses across training seeds and T3 target colors."""
    environment = PickPlaceEnv(task, "L1", "state")
    signatures: set[tuple[float, ...]] = set()
    target_objects: set[str] = set()

    for seed in range(12):
        observation, _ = environment.reset(seed=seed)
        target_object = str(observation["target_object"])
        target_objects.add(target_object)
        target_pose = observation["object_poses"][target_object]
        parts = [
            np.asarray(target_pose["position"], dtype=np.float64),
            np.asarray(target_pose["quaternion_wxyz"], dtype=np.float64),
            np.asarray(observation["target_pose"], dtype=np.float64),
        ]
        if task == "T3":
            bowl = observation["object_poses"]["bowl"]
            parts.extend(
                (
                    np.asarray(bowl["position"], dtype=np.float64),
                    np.asarray(bowl["quaternion_wxyz"], dtype=np.float64),
                )
            )
        signatures.add(tuple(np.round(np.concatenate(parts), 5)))

    environment.close()
    assert len(signatures) == 12
    if task == "T3":
        assert target_objects == {"cube_red", "cube_blue", "cube_green"}


def test_demo_feature_schema_contains_visual_and_oracle_inputs() -> None:
    """Keep future-model inputs and executed actions in the LeRobot schema."""
    features = dataset_features()

    assert features["observation.images.front"]["shape"] == (
        3,
        DATASET_IMAGE_SIZE,
        DATASET_IMAGE_SIZE,
    )
    assert features["observation.images.wrist"]["shape"] == (
        3,
        DATASET_IMAGE_SIZE,
        DATASET_IMAGE_SIZE,
    )
    assert features["observation.state"]["shape"] == (12,)
    assert features["observation.object_poses"]["shape"] == (28,)
    assert features["observation.environment_state"]["shape"] == (28,)
    assert features["observation.target_pose"]["shape"] == (4,)
    assert features["action"]["shape"] == (5,)
    assert features["action.executed"]["shape"] == (5,)


def test_frame_preserves_clean_and_executed_actions_and_environment_state() -> None:
    """Label expert targets separately from actions applied to MuJoCo."""
    observation = {
        "observation.state": np.zeros(12, dtype=np.float32),
        "target_pose": np.zeros(4, dtype=np.float32),
        "object_poses": {
            name: {
                "position": np.array([index, 0.0, 0.0], dtype=np.float32),
                "quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
            for index, name in enumerate(
                ("cube_red", "cube_blue", "cube_green", "bowl")
            )
        },
    }
    clean_action = np.array([0.4, 0.0, 0.3, 0.1, 0.0], dtype=np.float64)
    executed_action = np.array([0.41, 0.0, 0.3, 0.1, 0.0], dtype=np.float64)
    images = {
        "observation.images.front": np.zeros(
            (DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE, 3), dtype=np.uint8
        ),
        "observation.images.wrist": np.zeros(
            (DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE, 3), dtype=np.uint8
        ),
    }

    frame = _make_frame(
        observation, images, clean_action, executed_action, "Place the red cube."
    )

    np.testing.assert_allclose(frame["action"], clean_action, rtol=1e-7)
    np.testing.assert_allclose(frame["action.executed"], executed_action, rtol=1e-7)
    np.testing.assert_array_equal(
        frame["observation.environment_state"],
        frame["observation.object_poses"],
    )
    assert frame["task"] == "Place the red cube."


@pytest.mark.parametrize(
    ("task", "target", "expected"),
    [
        ("T2", "cube_red", "Place the red cube in the target zone."),
        ("T3", "cube_green", "Place the green cube in the bowl."),
    ],
)
def test_episode_task_label_describes_the_expert_goal(
    task: str, target: str, expected: str
) -> None:
    """Use descriptive task strings instead of bare task IDs in LeRobot data."""
    assert episode_task(task, target) == expected


def test_t3l_episode_task_uses_instruction_verbatim() -> None:
    """Retain the sampled language instruction as the episode task label."""
    instruction = "Move the blue cube into the bowl."
    assert episode_task("T3-L", "cube_blue", instruction) == instruction


def test_open_loop_verification_requires_noise_free_collection(
    tmp_path: Path,
) -> None:
    """Avoid claiming action replay is open loop when labels came from noisy steps."""
    config = CollectionConfig(task="T2", episodes=20, verify_open_loop=True)

    with pytest.raises(ValueError, match="requires both noise values to be zero"):
        generate_dataset(config, output_root=tmp_path / "datasets")


def test_low_open_loop_rate_is_recorded_without_rejecting_dataset(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Treat open-loop replay as a diagnostic rather than a collection gate."""

    class CompletedFuture:
        """Return one completed synthetic rollout to the split collector."""

        def __init__(self, rollout: object) -> None:
            self._rollout = rollout

        def result(self) -> object:
            return self._rollout

        def cancel(self) -> bool:
            return False

    class ImmediateExecutor:
        """Run the attempt worker synchronously for this small accounting test."""

        def submit(self, worker: object, seed: int) -> CompletedFuture:
            del worker
            return CompletedFuture(
                generate_module.EpisodeRollout(
                    seed=seed,
                    success=True,
                    target_object="cube_red",
                    task_label="Place the red cube in the target zone.",
                    frame_count=0,
                    open_loop_success=False,
                )
            )

    class EmptyDatasetWriter:
        """Accept successful rollout frames without writing dataset files."""

        def add_frame(self, frame: dict[str, object]) -> None:
            del frame

        def save_episode(self, *, parallel_encoding: bool) -> None:
            del parallel_encoding

    monkeypatch.setattr(
        generate_module,
        "wait",
        lambda futures, return_when: (set(futures), set()),
    )
    config = CollectionConfig(
        task="T2",
        episodes=2,
        workers=1,
        verify_open_loop=True,
        noise_position_std=0.0,
        noise_yaw_degrees=0.0,
        noise_probability=0.0,
    )

    summary = generate_module._collect_split(
        ImmediateExecutor(),
        EmptyDatasetWriter(),
        split="train",
        target_episodes=2,
        seed_start=0,
        seed_stop=100,
        config=config,
    )

    assert summary.episodes == 2
    assert summary.open_loop_successes == 0
    assert summary.open_loop_rate == 0.0
    assert "recording this diagnostic without rejecting the dataset" in caplog.text

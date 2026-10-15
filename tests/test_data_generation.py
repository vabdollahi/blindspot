from pathlib import Path

import numpy as np
import pytest

from common.seeding import named_generators
from data.generate import (
    CollectionConfig,
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
    """Keep both policy input modes and the five-dimensional action in the data."""
    features = dataset_features()

    assert features["observation.images.front"]["shape"] == (3, 128, 128)
    assert features["observation.images.wrist"]["shape"] == (3, 128, 128)
    assert features["observation.state"]["shape"] == (12,)
    assert features["observation.object_poses"]["shape"] == (28,)
    assert features["observation.target_pose"]["shape"] == (4,)
    assert features["action"]["shape"] == (5,)


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


def test_open_loop_verification_requires_noise_free_collection(
    tmp_path: Path,
) -> None:
    """Avoid claiming action replay is open loop when labels came from noisy steps."""
    config = CollectionConfig(task="T2", episodes=20, verify_open_loop=True)

    with pytest.raises(ValueError, match="requires both noise values to be zero"):
        generate_dataset(config, output_root=tmp_path / "datasets")

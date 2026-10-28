from types import SimpleNamespace

import numpy as np
import pytest

import data.vla_subset as sampling
from envs.pick_place_env import CUBE_COLORS, STAGING_POSITIONS


class FakeNumericRows:
    """Provide the environment-state column needed for subset selection."""

    def __init__(self, states: list[np.ndarray]) -> None:
        self.states = states

    def with_format(self, format_name: str):
        """Return the numpy-formatted fake table."""
        if format_name != "numpy":
            raise ValueError("expected numpy format")
        return self

    def __getitem__(self, index):
        """Return row or column-oriented slice."""
        if isinstance(index, slice):
            return {"observation.environment_state": np.stack(self.states[index])}
        return {"observation.environment_state": self.states[index]}


class FakeSource:
    """Mimic LeRobot episode metadata and flattened object poses."""

    def __init__(
        self,
        episodes: list[tuple[int, str, tuple[str, ...]]],
        *,
        frames_per_episode: int = 24,
    ) -> None:
        self.meta = SimpleNamespace(episodes=[])
        self.states: list[np.ndarray] = []
        self.frames_per_episode = frames_per_episode
        for episode_id, (target_color, active_colors) in enumerate(episodes):
            start = len(self.states)
            state = np.zeros(28, dtype=np.float32)
            for cube_index, color in enumerate(CUBE_COLORS):
                position = (
                    np.array([0.4, -0.1, 0.02], dtype=np.float32)
                    if color in active_colors
                    else STAGING_POSITIONS[cube_index].astype(np.float32)
                )
                state[cube_index * 7 : cube_index * 7 + 3] = position
            self.states.extend(state.copy() for _ in range(frames_per_episode))
            stop = len(self.states)
            self.meta.episodes.append(
                {
                    "episode_index": episode_id,
                    "dataset_from_index": start,
                    "dataset_to_index": stop,
                    "tasks": [f"Put the {target_color} cube in the bowl."],
                }
            )
        self.numeric = FakeNumericRows(self.states)

    def select_columns(self, names):
        """Select the environment-state field."""
        if names != ["observation.environment_state"]:
            raise ValueError("unexpected columns")
        return self.numeric


def test_active_cube_colors_excludes_staged_cubes() -> None:
    """Identify the two visible cubes used by the limited rollout filter."""
    positions = {
        f"cube_{color}": (
            np.array([0.4, -0.1, 0.02], dtype=np.float32)
            if color in {"red", "blue"}
            else STAGING_POSITIONS[index].astype(np.float32)
        )
        for index, color in enumerate(CUBE_COLORS)
    }

    assert sampling.active_cube_colors_from_positions(positions) == ("red", "blue")


def test_limited_training_selection_balances_colors_and_spreads_episodes() -> None:
    """Select two-cube red/blue episodes and spaced samples deterministically."""
    episodes = [
        ("red", ("red", "blue")),
        ("blue", ("blue", "green")),
        ("red", ("red", "green")),
        ("blue", ("red", "blue")),
        ("red", ("red", "blue")),
        ("blue", ("blue", "green")),
        ("red", ("red", "green")),
        ("blue", ("red", "blue")),
        ("green", ("red", "blue")),
        ("red", ("red", "blue", "green")),
    ]
    source = FakeSource(episodes)

    selection = sampling.select_limited_t3l_frames(
        source,
        "train",
        episodes_per_color=2,
        frames_per_episode=4,
    )

    assert selection.episode_ids_by_color == {"red": [0, 6], "blue": [1, 7]}
    assert selection.frame_indices == [
        0,
        7,
        15,
        23,
        24,
        31,
        39,
        47,
        144,
        151,
        159,
        167,
        168,
        175,
        183,
        191,
    ]


def test_limited_validation_keeps_all_eligible_episodes() -> None:
    """Use every matching validation episode rather than truncating it."""
    source = FakeSource(
        [
            ("red", ("red", "blue")),
            ("blue", ("blue", "green")),
            ("green", ("green", "red")),
        ],
        frames_per_episode=8,
    )

    selection = sampling.select_limited_t3l_frames(
        source,
        "validation",
        episodes_per_color=40,
        frames_per_episode=4,
    )

    assert selection.episode_ids_by_color == {"red": [0], "blue": [1]}
    assert len(selection.frame_indices) == 8


def test_limited_profile_defaults_to_48_spaced_frames_per_episode() -> None:
    """Use denser coverage while retaining the episode-balanced profile."""
    source = FakeSource(
        [
            *(("red", ("red", "blue")) for _ in range(40)),
            *(("blue", ("blue", "green")) for _ in range(40)),
        ],
        frames_per_episode=64,
    )

    selection = sampling.select_limited_t3l_frames(source, "train")

    assert selection.episode_ids_by_color == {
        "red": list(range(40)),
        "blue": list(range(40, 80)),
    }
    assert selection.frames_per_episode == 48
    assert len(selection.frame_indices) == 40 * 2 * 48


def test_limited_training_defaults_to_the_balanced_maximum() -> None:
    """Use as many episodes per color as the smaller color class provides."""
    source = FakeSource(
        [
            *(("red", ("red", "blue")) for _ in range(3)),
            *(("blue", ("blue", "green")) for _ in range(5)),
        ],
        frames_per_episode=8,
    )

    selection = sampling.select_limited_t3l_frames(
        source, "train", frames_per_episode=2
    )

    assert selection.episode_ids_by_color == {"red": [0, 1, 2], "blue": [3, 5, 7]}
    assert len(selection.frame_indices) == 6 * 2


def test_limited_training_fails_if_a_target_color_lacks_enough_episodes() -> None:
    """Reject an undersized color class instead of silently shrinking the plan."""
    source = FakeSource(
        [
            ("red", ("red", "blue")),
            ("blue", ("blue", "green")),
        ]
    )

    with pytest.raises(ValueError, match="needs 2 training episodes for red"):
        sampling.select_limited_t3l_frames(
            source,
            "train",
            episodes_per_color=2,
            frames_per_episode=4,
        )


def test_limited_selection_rejects_unknown_split() -> None:
    """Reject invalid split values instead of treating them as validation."""
    source = FakeSource([("red", ("red", "blue")), ("blue", ("blue", "green"))])

    with pytest.raises(ValueError, match="split must be"):
        sampling.select_limited_t3l_frames(
            source,
            "test",
            episodes_per_color=1,
            frames_per_episode=4,
        )

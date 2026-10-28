from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from data.vla_subset import active_cube_colors_from_positions
from envs.pick_place_env import CUBE_COLORS, STAGING_POSITIONS
from eval.vla_profile import _run_episode, matches_limited_vla_profile
from eval.vla_trajectory import VLAEpisodeTrajectory, save_vla_trajectory


def _object_poses(active_colors: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """Build deterministic cube poses with the requested active colors."""
    poses: dict[str, dict[str, np.ndarray]] = {}
    for color, staging in zip(CUBE_COLORS, STAGING_POSITIONS, strict=True):
        position = np.asarray([staging[0], staging[1], 0.04], dtype=np.float32)
        if color in active_colors:
            position[0] += 0.1
        poses[f"cube_{color}"] = {"position": position}
    return poses


def test_limited_vla_profile_matches_only_red_blue_two_cube_scenes() -> None:
    """Filter by instruction target color and the count of active cubes."""
    poses = _object_poses(("blue", "green"))

    matched, target_color, active_colors = matches_limited_vla_profile(
        "Place the blue cube in the bowl.",
        poses,
    )
    green_matched, green_target, _ = matches_limited_vla_profile(
        "Place the green cube in the bowl.",
        poses,
    )
    three_cube_matched, _, three_active_colors = matches_limited_vla_profile(
        "Place the blue cube in the bowl.",
        _object_poses(("red", "blue", "green")),
    )

    assert matched
    assert target_color == "blue"
    assert active_colors == active_cube_colors_from_positions(poses)
    assert not green_matched
    assert green_target == "green"
    assert not three_cube_matched
    assert len(three_active_colors) == 3


class FakePolicy:
    """Return one fixed action without loading a VLM."""

    model_id = "test/smolvlm"
    model_revision = "test-revision"
    training_profile = "limited"
    chunk_size = 1
    replan_interval = 1
    ensemble_decay = 0.0

    def reset(self, instruction: str | None = None) -> None:
        """Accept the episode instruction."""

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a deterministic finite action."""
        return np.asarray([0.4, 0.0, 0.2, 0.0, 0.0], dtype=np.float32)


class FakeEnvironment:
    """Provide one deterministic terminal transition for trajectory tests."""

    def __init__(self) -> None:
        self.data = SimpleNamespace(time=0.0)
        self._state = np.zeros(12, dtype=np.float32)
        self._object_poses = _object_poses(("blue", "green"))

    def observe_images(self) -> dict[str, np.ndarray]:
        """Provide VLA-shaped camera observations without rendering."""
        return {
            "observation.images.front": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action and end the diagnostic episode."""
        self._state[0] += float(action[0]) / 100.0
        self.data.time = 0.05
        observation = {
            "observation.state": self._state.copy(),
            "object_poses": self._object_poses,
        }
        info = {
            "success": False,
            "failure_reason": "timeout",
            "collision_count": 0,
            "episode_time": self.data.time,
        }
        return observation, 0.0, False, True, info


def test_vla_episode_trajectory_stores_states_actions_and_reloads(
    tmp_path: Path,
) -> None:
    """Persist a replayable step with before/after robot and object state."""
    object_poses = _object_poses(("blue", "green"))
    trajectory = _run_episode(
        FakePolicy(),
        FakeEnvironment(),
        {
            "instruction": "Place the blue cube in the bowl.",
            "observation.state": np.zeros(12, dtype=np.float32),
            "object_poses": object_poses,
        },
        checkpoint=Path("checkpoints/fake.pt"),
        seed=1_000_000,
        level="L1",
        instruction="Place the blue cube in the bowl.",
        instruction_template="Place the {color} cube in the bowl.",
        target_color="blue",
        active_colors=active_cube_colors_from_positions(object_poses),
    )
    path = save_vla_trajectory(tmp_path / "episode.json", trajectory)
    reloaded = VLAEpisodeTrajectory.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    assert reloaded.result.episode_steps == 1
    assert len(reloaded.steps) == 1
    assert len(reloaded.steps[0].state) == 12
    assert len(reloaded.steps[0].action) == 5
    assert reloaded.steps[0].state != reloaded.steps[0].next_state
    assert reloaded.steps[0].object_positions_before == (
        reloaded.steps[0].object_positions_after
    )

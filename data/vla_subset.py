"""Episode-balanced frame selection for the limited T3-L VLA baseline profile."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from envs.pick_place_env import CUBE_COLORS, STAGING_POSITIONS

VLA_TARGET_COLORS = ("red", "blue")
VLA_ACTIVE_CUBE_COUNT = 2
VLA_FRAMES_PER_EPISODE = 48
_OBJECT_POSE_WIDTH = 7
_INACTIVE_POSITION_TOLERANCE_M = 0.02


class VLAFrameSelection(BaseModel):
    """Record the source episodes and selected frame indices for one split."""

    model_config = ConfigDict(extra="forbid")

    split: Literal["train", "validation"]
    frame_indices: list[int]
    episode_ids_by_color: dict[str, list[int]]
    frames_per_episode: int = Field(ge=1)
    active_cube_count: int = Field(ge=1)


def active_cube_count_from_positions(cube_positions: Mapping[str, Any]) -> int:
    """Count cubes outside their inactive staging positions."""
    return len(active_cube_colors_from_positions(cube_positions))


def active_cube_colors_from_positions(
    cube_positions: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return colors whose cubes are outside their inactive staging locations."""
    active_colors: list[str] = []
    for color, staging in zip(CUBE_COLORS, STAGING_POSITIONS, strict=True):
        value = cube_positions.get(f"cube_{color}")
        if isinstance(value, Mapping):
            value = value.get("position")
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"cube_{color} position must be a finite 3-D vector")
        if np.linalg.norm(position[:2] - staging[:2]) > _INACTIVE_POSITION_TOLERANCE_M:
            active_colors.append(color)
    return tuple(active_colors)


def select_limited_t3l_frames(
    dataset: Any,
    split: Literal["train", "validation"],
    *,
    episodes_per_color: int | None = None,
    frames_per_episode: int = VLA_FRAMES_PER_EPISODE,
    active_cube_count: int = VLA_ACTIVE_CUBE_COUNT,
) -> VLAFrameSelection:
    """Select balanced red/blue two-cube episodes and spaced frames.

    Training uses ``episodes_per_color`` episodes of each target color, or by
    default as many as the smaller color class provides.
    """
    if split not in {"train", "validation"}:
        raise ValueError("VLA subset split must be 'train' or 'validation'")
    if (
        (episodes_per_color is not None and episodes_per_color < 1)
        or frames_per_episode < 1
        or active_cube_count < 1
    ):
        raise ValueError("VLA subset sizes and active cube count must be positive")
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is None:
        raise TypeError("LeRobot dataset metadata must provide episode records")

    environment_view = dataset.select_columns(
        ["observation.environment_state"]
    ).with_format("numpy")
    environment_states = np.asarray(
        environment_view[:]["observation.environment_state"],
        dtype=np.float32,
    )
    if (
        environment_states.ndim != 2
        or environment_states.shape[1] != 4 * _OBJECT_POSE_WIDTH
        or not np.isfinite(environment_states).all()
    ):
        raise ValueError("T3-L environment states must be finite 28-D vectors")

    candidates: dict[str, list[tuple[int, int, int]]] = {
        color: [] for color in VLA_TARGET_COLORS
    }
    for episode in episodes:
        episode_id = int(episode["episode_index"])
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        task_labels = episode.get("tasks")
        if not isinstance(task_labels, (list, tuple)) or not task_labels:
            raise ValueError(f"T3-L episode {episode_id} has no task label")
        color = _task_color(task_labels[0])
        if color not in candidates:
            continue
        if start < 0 or stop <= start or stop > len(environment_states):
            raise ValueError(f"T3-L episode {episode_id} has invalid frame bounds")
        first_state = environment_states[start]
        cube_positions = {
            f"cube_{cube_color}": first_state[
                index * _OBJECT_POSE_WIDTH : index * _OBJECT_POSE_WIDTH + 3
            ]
            for index, cube_color in enumerate(CUBE_COLORS)
        }
        if active_cube_count_from_positions(cube_positions) == active_cube_count:
            candidates[color].append((episode_id, start, stop))

    for color, color_candidates in candidates.items():
        if not color_candidates:
            raise ValueError(
                f"No {split} episodes match target color {color!r} and "
                f"{active_cube_count} active cubes"
            )
    training_count = (
        min(len(color_candidates) for color_candidates in candidates.values())
        if episodes_per_color is None
        else episodes_per_color
    )
    selected_episodes: dict[str, list[tuple[int, int, int]]] = {}
    for color, color_candidates in candidates.items():
        if split == "train":
            if len(color_candidates) < training_count:
                raise ValueError(
                    f"Limited VLA profile needs {training_count} training "
                    f"episodes for {color}, but found {len(color_candidates)}"
                )
            candidate_indices = np.linspace(
                0,
                len(color_candidates) - 1,
                num=training_count,
                dtype=np.int64,
            )
            selected_episodes[color] = [
                color_candidates[int(index)] for index in candidate_indices
            ]
        else:
            selected_episodes[color] = color_candidates

    frame_indices: list[int] = []
    episode_ids_by_color: dict[str, list[int]] = {}
    for color, color_episodes in selected_episodes.items():
        episode_ids_by_color[color] = []
        for episode_id, start, stop in color_episodes:
            frame_count = stop - start
            sample_count = min(frames_per_episode, frame_count)
            offsets = np.linspace(
                0,
                frame_count - 1,
                num=sample_count,
                dtype=np.int64,
            )
            frame_indices.extend(start + int(offset) for offset in offsets)
            episode_ids_by_color[color].append(episode_id)

    return VLAFrameSelection(
        split=split,
        frame_indices=sorted(frame_indices),
        episode_ids_by_color=episode_ids_by_color,
        frames_per_episode=frames_per_episode,
        active_cube_count=active_cube_count,
    )


def _task_color(task_label: Any) -> str:
    """Extract exactly one target color from a T3-L task label."""
    if not isinstance(task_label, str) or not task_label.strip():
        raise ValueError("T3-L task labels must be non-empty strings")
    matches = [
        color
        for color in CUBE_COLORS
        if re.search(rf"\b{color}\b", task_label.lower()) is not None
    ]
    if len(matches) != 1:
        raise ValueError(f"T3-L task label must name one cube color: {task_label!r}")
    return matches[0]

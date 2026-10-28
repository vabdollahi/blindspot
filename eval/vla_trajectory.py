"""Validated state/action records for replayable VLA evaluation episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eval.schema import EpisodeResult


class VLATrajectoryStep(BaseModel):
    """Store one policy observation, action, simulator transition, and object state."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    state: list[float] = Field(min_length=12, max_length=12)
    action: list[float] = Field(min_length=5, max_length=5)
    next_state: list[float] = Field(min_length=12, max_length=12)
    object_positions_before: dict[str, list[float]]
    object_positions_after: dict[str, list[float]]
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    failure_reason: str | None
    collision_count: int = Field(ge=0)
    episode_time: float = Field(ge=0.0)


class VLAEpisodeTrajectory(BaseModel):
    """Store a reproducible limited-profile VLA rollout for fast replay."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task: Literal["T3-L"] = "T3-L"
    level: Literal["L0", "L1", "L2"]
    seed: int = Field(ge=1_000_000)
    instruction: str
    instruction_template: str
    target_color: Literal["red", "blue"]
    active_cube_colors: tuple[str, str]
    checkpoint: str
    model_id: str
    model_revision: str
    training_profile: Literal["limited"]
    git_commit: str
    result: EpisodeResult
    steps: list[VLATrajectoryStep]


def save_vla_trajectory(
    path: str | Path,
    trajectory: VLAEpisodeTrajectory,
) -> Path:
    """Atomically write one validated VLA episode trajectory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        trajectory.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination

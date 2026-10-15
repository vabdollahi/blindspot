"""Validated schemas for per-episode and aggregate evaluation results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EpisodeResult(BaseModel):
    """Store deterministic outcome and simulated timing for one episode."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    success: bool
    terminated: bool
    truncated: bool
    failure_reason: str | None
    episode_steps: int = Field(ge=0)
    episode_time: float = Field(ge=0.0)
    collision_count: int = Field(ge=0)
    total_reward: float
    instruction: str | None = None
    error: str | None = None


class EvaluationSummary(BaseModel):
    """Summarize successes, confidence bounds, and failure counts."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(gt=0)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    wilson_95: tuple[float, float]
    failure_breakdown: dict[str, int]


class EvaluationResult(BaseModel):
    """Represent one complete policy/task/level evaluation as JSON data."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    method: str
    task: str
    level: str
    obs_mode: Literal["state", "image"]
    seeds: list[int]
    git_commit: str
    config_hash: str
    summary: EvaluationSummary
    episodes: list[EpisodeResult]

    def json_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation of the result."""
        return self.model_dump(mode="json")

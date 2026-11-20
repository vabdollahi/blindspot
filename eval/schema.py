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


class MetricEstimate(BaseModel):
    """A point estimate with an optional bootstrap interval and per-seed values."""

    model_config = ConfigDict(extra="forbid")

    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    per_seed: list[float] = Field(default_factory=list)


class ProbeEncoder(BaseModel):
    """Identify one probed encoder and where its weights came from."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["random", "jepa", "supervised", "pretrained"]
    seeds: list[int]
    parameters: int = Field(ge=0)
    sources: list[str]


class ProbeRow(BaseModel):
    """Probe metrics for one encoder, readout, and probe model on one split."""

    model_config = ConfigDict(extra="forbid")

    encoder: str
    readout: Literal["latent", "global", "grid", "output"]
    probe: Literal["linear", "mlp", "trained"]
    split: Literal["validation", "test"]
    metrics: dict[str, MetricEstimate]


class TrainingCurvePoint(BaseModel):
    """One periodic validation record copied from a JEPA run."""

    model_config = ConfigDict(extra="forbid")

    step: int
    probe_score: float
    effective_rank: float | None = None
    latent_std: float | None = None
    persistence_ratio: float | None = None
    gate_magnitude: float | None = None


class JEPAProbeResult(BaseModel):
    """Phase 4 probe study results, written by scripts/evaluate_jepa.py."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task: Literal["T3"] = "T3"
    level: Literal["L1"] = "L1"
    git_commit: str
    train_datasets: list[str]
    validation_dataset: str
    test_dataset: str
    train_timesteps: int
    test_episodes: int
    resamples: int
    encoders: list[ProbeEncoder]
    rows: list[ProbeRow]
    h1_pass: bool | None
    phase6_encoder: str | None
    validation_probe_scores: dict[str, float] = Field(default_factory=dict)
    training_curves: dict[str, list[TrainingCurvePoint]]


class RolloutCurve(BaseModel):
    """Decoded error against horizon for one model, target, and phase."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["predictor", "persistence", "probe_floor"]
    target: Literal["cube_xyz", "tcp_xyz"]
    phase: Literal["all", "free_motion", "held", "after_release"]
    errors_cm: list[MetricEstimate]


class JEPARolloutResult(BaseModel):
    """Phase 4 predictor rollout results, written by scripts/evaluate_jepa.py."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task: Literal["T3"] = "T3"
    level: Literal["L1"] = "L1"
    git_commit: str
    runs: list[str]
    test_dataset: str
    horizons_seconds: list[float]
    windows: int
    curves: list[RolloutCurve]
    latent_mse_vs_persistence: list[MetricEstimate]
    h4_pass: bool

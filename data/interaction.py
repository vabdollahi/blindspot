"""Collect a broad LeRobot interaction corpus for world-model training."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import multiprocessing
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.seeding import (
    EVALUATION_SEED_START,
    EVALUATION_SEED_STOP,
    TRAIN_SEED_START,
    TRAIN_SEED_STOP,
    VALIDATION_SEED_START,
    VALIDATION_SEED_STOP,
    named_generators,
)
from data.generate import (
    DATASET_FPS,
    DATASET_IMAGE_SIZE,
    DATASET_SCHEMA_VERSION,
    _checkpoint_digest,
    _make_frame,
    _publish_staged_dataset,
    _require_lerobot,
    _select_dataset_name,
    _verify_dataset,
    clip_action,
    dataset_features,
    episode_task,
    maybe_perturb_action,
)
from envs.pick_place_env import PickPlaceEnv
from eval.policy import Policy
from motion.control import WORKSPACE_LOWER, WORKSPACE_UPPER, tcp_pose

TASKS = ("T1", "T2", "T3")
SeedSplit = Literal["train", "validation", "evaluation"]
SEED_RANGES: dict[str, tuple[int, int]] = {
    "train": (TRAIN_SEED_START, TRAIN_SEED_STOP),
    "validation": (VALIDATION_SEED_START, VALIDATION_SEED_STOP),
    "evaluation": (EVALUATION_SEED_START, EVALUATION_SEED_STOP),
}
REQUESTED_SOURCES = (
    "rl-expert",
    "rl-expert-noisy",
    "intermediate-ppo",
    "scripted-exploration",
)
SOURCE_IDS = {
    "rl-expert": 0,
    "rl-expert-failure": 1,
    "rl-expert-noisy": 2,
    "intermediate-ppo": 3,
    "scripted-exploration": 4,
}
ACTION_POSITION_BINS = (6, 6, 4)
ACTION_YAW_BINS = 12
MINIMUM_WORKSPACE_SPAN_FRACTION = 0.25
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKER_CONFIG: InteractionConfig | None = None
_WORKER_ENVIRONMENT: PickPlaceEnv | None = None
_WORKER_TASK: str | None = None
_WORKER_TARGET_BANK: list[np.ndarray] = []
_WORKER_POLICY_CACHE: dict[Path, Policy] = {}
_LOGGER = logging.getLogger(__name__)


class InteractionConfig(BaseModel):
    """Validate one reproducible world-model corpus collection request."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(default=2_000, ge=12, le=TRAIN_SEED_STOP)
    workers: int = Field(
        default_factory=lambda: min(8, os.cpu_count() or 1),
        ge=1,
    )
    tasks: tuple[Literal["T1", "T2", "T3"], ...] = TASKS
    seed_split: SeedSplit = "train"
    seed_start: int = Field(ge=0)
    large_noise_position_std: float = Field(default=0.05, gt=0.0, allow_inf_nan=False)
    large_noise_yaw_degrees: float = Field(default=15.0, gt=0.0, allow_inf_nan=False)
    reachable_target_count: int = Field(default=24, ge=4)
    max_attempts_per_target: int = Field(default=50, ge=1)

    @model_validator(mode="before")
    @classmethod
    def default_seed_start(cls, data: Any) -> Any:
        """Start at the first seed of the requested split unless one is given."""
        if isinstance(data, Mapping) and data.get("seed_start") is None:
            split = data.get("seed_split", "train")
            if split in SEED_RANGES:
                return {**data, "seed_start": SEED_RANGES[split][0]}
        return data

    @model_validator(mode="after")
    def validate_seed_range_and_tasks(self) -> InteractionConfig:
        """Keep every episode inside one reserved seed range, with known tasks."""
        start, stop = SEED_RANGES[self.seed_split]
        if self.seed_start < start or self.seed_start + self.episodes > stop:
            raise ValueError(
                f"Interaction seeds {self.seed_start}-"
                f"{self.seed_start + self.episodes - 1} must lie in the "
                f"{self.seed_split} range [{start}, {stop})"
            )
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be a non-empty list of distinct tasks")
        self.tasks = tuple(task for task in TASKS if task in self.tasks)
        return self


class CheckpointRecord(BaseModel):
    """Identify a frozen or intermediate policy used by the corpus."""

    task: Literal["T1", "T2", "T3"]
    kind: Literal["rl-expert", "intermediate-ppo"]
    checkpoint_dir: str
    sha256: str


class InteractionEpisodeRequest(BaseModel):
    """Describe one deterministic episode assigned to a worker."""

    model_config = ConfigDict(extra="forbid")

    order_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    task: Literal["T1", "T2", "T3"]
    requested_source: Literal[
        "rl-expert",
        "rl-expert-noisy",
        "intermediate-ppo",
        "scripted-exploration",
    ]
    checkpoint_dir: Path | None = None
    checkpoint_sha256: str | None = None


class InteractionEpisodeResult(BaseModel):
    """Carry one episode's transitions and outcome from its worker."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    order_index: int
    seed: int
    task: str
    requested_source: str
    source: str
    source_id: int
    checkpoint_dir: str | None = None
    checkpoint_sha256: str | None = None
    success: bool
    failure_reason: str | None = None
    frame_count: int
    frames: list[dict[str, Any]]


class InteractionEpisodeRecord(BaseModel):
    """Record the provenance and outcome of one saved corpus episode."""

    episode_index: int
    order_index: int
    seed: int
    task: str
    requested_source: str
    source: str
    source_id: int
    checkpoint_dir: str | None = None
    checkpoint_sha256: str | None = None
    success: bool
    failure_reason: str | None = None
    frame_count: int


class InteractionCorpusSummary(BaseModel):
    """Describe the corpus, checkpoint provenance, and coverage checks."""

    dataset: str
    schema_version: int = DATASET_SCHEMA_VERSION
    level: str = "L1"
    fps: int = DATASET_FPS
    image_size: int = DATASET_IMAGE_SIZE
    tasks: list[str] = list(TASKS)
    seed_split: SeedSplit = "train"
    total_episodes: int
    source_ids: dict[str, int]
    requested_source_counts: dict[str, int]
    source_counts: dict[str, int]
    task_counts: dict[str, int]
    success_counts: dict[str, int]
    failure_breakdown: dict[str, int]
    large_noise_position_std_m: float
    large_noise_yaw_std_degrees: float
    large_noise_probability: float = 1.0
    reachable_target_count_per_worker: int
    action_coverage: dict[str, Any]
    checkpoints: list[CheckpointRecord]
    feature_shapes: dict[str, list[int]]
    episodes: list[InteractionEpisodeRecord]


class RandomReachablePolicy:
    """Issue random commands from a bank of IK-verified reachable TCP targets."""

    def __init__(
        self,
        targets: list[np.ndarray],
        random: np.random.Generator,
    ) -> None:
        """Initialize an episode-local exploratory command sampler."""
        if not targets:
            raise ValueError("Exploration requires at least one reachable target")
        self._targets = targets
        self._random = random
        self._remaining_steps = 0
        self._action = np.zeros(5, dtype=np.float64)

    def reset(self, instruction: str | None = None) -> None:
        """Discard the previous target before a new exploratory episode."""
        del instruction
        self._remaining_steps = 0

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Choose a reachable TCP target and random gripper command periodically."""
        del obs
        if self._remaining_steps == 0:
            target_index = int(self._random.integers(0, len(self._targets)))
            gripper_open = float(self._random.integers(0, 2))
            self._action = np.array(
                [*self._targets[target_index], gripper_open],
                dtype=np.float64,
            )
            self._remaining_steps = int(self._random.integers(10, 41))
        self._remaining_steps -= 1
        return self._action.copy()


def default_dataset_name(config: InteractionConfig) -> str:
    """Name the full training corpus interaction-l1, and subsets by task and split."""
    if config.tasks == TASKS and config.seed_split == "train":
        return "interaction-l1"
    task_part = "-".join(task.lower() for task in config.tasks)
    return f"interaction-{task_part}-l1-{config.seed_split}"


def interaction_dataset_features() -> dict[str, dict[str, Any]]:
    """Return the demonstration schema plus corpus provenance and next success."""
    features = dataset_features()
    features["source_id"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["source_id"],
    }
    features["next.success"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["success"],
    }
    return features


def interaction_task_label(task: str, target_object: str) -> str:
    """Describe the requested task for LeRobot's per-frame task column."""
    if task == "T1":
        return "Reach the target point."
    return episode_task(task, target_object)


def _discover_intermediate_checkpoints(task: str) -> list[Path]:
    """Find valid saved PPO step checkpoints for one task."""
    task_root = _PROJECT_ROOT / "checkpoints" / "rl" / f"{task.lower()}-l1"
    candidates = sorted(task_root.glob("seed-*/checkpoints/step_*"))
    valid_checkpoints = [
        path
        for path in candidates
        if re.fullmatch(r"step_[0-9]+", path.name)
        and (path / "model.zip").is_file()
        and (path / "vecnormalize.pkl").is_file()
    ]
    if not valid_checkpoints:
        raise FileNotFoundError(
            f"No intermediate PPO checkpoints found for {task} under {task_root}"
        )
    return valid_checkpoints


def _load_checkpoint_catalog(
    tasks: tuple[str, ...] = TASKS,
) -> tuple[dict[str, Path], dict[str, list[Path]], dict[Path, str]]:
    """Resolve frozen and intermediate policy checkpoints for the given tasks."""
    experts: dict[str, Path] = {}
    intermediate: dict[str, list[Path]] = {}
    digests: dict[Path, str] = {}
    for task in tasks:
        expert = (
            _PROJECT_ROOT
            / "checkpoints"
            / "rl"
            / f"{task.lower()}-l1"
            / "seed-0"
            / "best"
        )
        experts[task] = expert
        intermediate[task] = _discover_intermediate_checkpoints(task)
        for checkpoint_dir in (expert, *intermediate[task]):
            normalized_path = checkpoint_dir.resolve()
            digests[normalized_path] = _checkpoint_digest(normalized_path)
        experts[task] = expert.resolve()
        intermediate[task] = [path.resolve() for path in intermediate[task]]
    return experts, intermediate, digests


def build_episode_requests(
    config: InteractionConfig,
    experts: dict[str, Path],
    intermediate: dict[str, list[Path]],
    checkpoint_digests: dict[Path, str],
) -> list[InteractionEpisodeRequest]:
    """Build a balanced task/source schedule on the configured seed range."""
    tasks = config.tasks
    requests: list[InteractionEpisodeRequest] = []
    for order_index in range(config.episodes):
        task = tasks[order_index % len(tasks)]
        source_index = (order_index // len(tasks)) % len(REQUESTED_SOURCES)
        source = REQUESTED_SOURCES[source_index]
        checkpoint_dir: Path | None = None
        checkpoint_sha256: str | None = None
        if source in {"rl-expert", "rl-expert-noisy"}:
            checkpoint_dir = experts[task]
            checkpoint_sha256 = checkpoint_digests[checkpoint_dir]
        elif source == "intermediate-ppo":
            task_checkpoints = intermediate[task]
            checkpoint_dir = task_checkpoints[
                (order_index // (len(tasks) * len(REQUESTED_SOURCES)))
                % len(task_checkpoints)
            ]
            checkpoint_sha256 = checkpoint_digests[checkpoint_dir]
        requests.append(
            InteractionEpisodeRequest(
                order_index=order_index,
                seed=config.seed_start + order_index,
                task=task,
                requested_source=source,
                checkpoint_dir=checkpoint_dir,
                checkpoint_sha256=checkpoint_sha256,
            )
        )
    return requests


def _build_reachable_target_bank(
    environment: PickPlaceEnv,
    *,
    count: int,
    max_attempts: int,
    seed: int,
) -> list[np.ndarray]:
    """Sample and verify reachable targets with the environment's Mink IK model."""
    random = named_generators(seed, ("exploration",))["exploration"]
    targets: list[np.ndarray] = []
    attempts = 0
    while len(targets) < count and attempts < count * max_attempts:
        attempts += 1
        position = random.uniform(WORKSPACE_LOWER, WORKSPACE_UPPER)
        yaw = float(random.uniform(-np.pi / 2, np.pi / 2))
        _, reachable = environment.ik.solve(
            tcp_pose(position, yaw),
            environment.ik.rest_q,
        )
        if reachable:
            targets.append(np.array([*position, yaw], dtype=np.float64))
    if len(targets) != count:
        raise RuntimeError(
            f"Found only {len(targets)}/{count} reachable exploratory targets "
            f"after {attempts} IK checks"
        )
    return targets


def _initialize_worker(
    task: str,
    config_data: dict[str, Any],
    needs_exploration: bool,
) -> None:
    """Create the simulation, renderer, and exploration bank in each worker."""
    global _WORKER_CONFIG, _WORKER_ENVIRONMENT, _WORKER_TASK
    global _WORKER_TARGET_BANK, _WORKER_POLICY_CACHE
    _WORKER_CONFIG = InteractionConfig.model_validate(config_data)
    _WORKER_TASK = task
    _WORKER_ENVIRONMENT = PickPlaceEnv(
        task,
        "L1",
        "state",
        render_mode="rgb_array",
        image_size=DATASET_IMAGE_SIZE,
    )
    _WORKER_POLICY_CACHE = {}
    if needs_exploration:
        task_index = TASKS.index(task)
        _WORKER_TARGET_BANK = _build_reachable_target_bank(
            _WORKER_ENVIRONMENT,
            count=_WORKER_CONFIG.reachable_target_count,
            max_attempts=_WORKER_CONFIG.max_attempts_per_target,
            seed=24_071 + task_index,
        )
    else:
        _WORKER_TARGET_BANK = []
    atexit.register(_close_worker)


def _close_worker() -> None:
    """Close the worker-owned MuJoCo renderer."""
    if _WORKER_ENVIRONMENT is not None:
        _WORKER_ENVIRONMENT.close()


def _load_worker_policy(checkpoint_dir: Path) -> Policy:
    """Load and cache one frozen or intermediate PPO checkpoint in a worker."""
    if checkpoint_dir not in _WORKER_POLICY_CACHE:
        from policies.rl import load_rl_policy

        _WORKER_POLICY_CACHE[checkpoint_dir] = load_rl_policy(checkpoint_dir)
    return _WORKER_POLICY_CACHE[checkpoint_dir]


def _make_interaction_frame(
    observation: dict[str, Any],
    images: dict[str, np.ndarray],
    clean_action: np.ndarray,
    executed_action: np.ndarray,
    task_label: str,
    *,
    next_success: bool,
    source_id: int,
) -> dict[str, Any]:
    """Add outcome and source labels to one transition frame."""
    frame = _make_frame(
        observation,
        images,
        clean_action,
        executed_action,
        task_label,
    )
    frame["source_id"] = np.asarray([source_id], dtype=np.int64)
    frame["next.success"] = np.asarray([int(next_success)], dtype=np.int64)
    return frame


def _run_attempt(request: InteractionEpisodeRequest) -> InteractionEpisodeResult:
    """Collect one complete episode and retain every transition, including failures."""
    if _WORKER_CONFIG is None or _WORKER_ENVIRONMENT is None or _WORKER_TASK is None:
        raise RuntimeError("Interaction worker was not initialized")
    if request.task != _WORKER_TASK:
        raise ValueError(
            f"Worker for {_WORKER_TASK} received a {request.task} episode"
        )
    environment = _WORKER_ENVIRONMENT
    generators = named_generators(
        request.seed,
        ("scene", "visual", "noise", "exploration"),
    )
    if request.requested_source == "scripted-exploration":
        policy: Policy = RandomReachablePolicy(
            _WORKER_TARGET_BANK,
            generators["exploration"],
        )
    else:
        if request.checkpoint_dir is None:
            raise ValueError(
                f"Source {request.requested_source} requires an RL checkpoint"
            )
        policy = _load_worker_policy(request.checkpoint_dir)

    observation, _ = environment.reset(seed=request.seed)
    instruction = observation.get("instruction")
    policy.reset(instruction if isinstance(instruction, str) else None)
    target_object = str(observation.get("target_object", ""))
    task_label = interaction_task_label(request.task, target_object)
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []

    while not terminated and not truncated:
        clean_action = clip_action(np.asarray(policy.act(observation), dtype=np.float64))
        if request.requested_source == "rl-expert-noisy":
            executed_action = maybe_perturb_action(
                clean_action,
                generators["noise"],
                probability=1.0,
                position_std=_WORKER_CONFIG.large_noise_position_std,
                yaw_std=float(np.deg2rad(_WORKER_CONFIG.large_noise_yaw_degrees)),
            )
        else:
            executed_action = clean_action.copy()
        executed_action = clip_action(executed_action)
        images = environment.observe_images()
        transition_observation = observation
        observation, _, terminated, truncated, info = environment.step(executed_action)
        frames.append(
            _make_interaction_frame(
                transition_observation,
                images,
                clean_action,
                executed_action,
                task_label,
                next_success=environment.success(),
                source_id=SOURCE_IDS[request.requested_source],
            )
        )

    success = bool(info.get("success", False))
    source = (
        "rl-expert-failure"
        if request.requested_source == "rl-expert" and not success
        else request.requested_source
    )
    source_id = SOURCE_IDS[source]
    for frame in frames:
        frame["source_id"][0] = source_id
    return InteractionEpisodeResult(
        order_index=request.order_index,
        seed=request.seed,
        task=request.task,
        requested_source=request.requested_source,
        source=source,
        source_id=source_id,
        checkpoint_dir=(
            str(request.checkpoint_dir)
            if request.checkpoint_dir is not None
            else None
        ),
        checkpoint_sha256=request.checkpoint_sha256,
        success=success,
        failure_reason=None if success else info.get("failure_reason"),
        frame_count=len(frames),
        frames=frames,
    )


def action_coverage_summary(actions: list[np.ndarray]) -> dict[str, Any]:
    """Summarize executed TCP target coverage and gripper-state counts."""
    if not actions:
        raise ValueError("Action coverage requires at least one executed action")
    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[1] != 5:
        raise ValueError("Executed actions must have shape (n, 5)")
    if not np.all(np.isfinite(action_array)):
        raise ValueError("Executed actions must contain only finite values")
    position_ranges = [
        (float(low), float(high))
        for low, high in zip(WORKSPACE_LOWER, WORKSPACE_UPPER, strict=True)
    ]
    position_histogram, _ = np.histogramdd(
        action_array[:, :3],
        bins=ACTION_POSITION_BINS,
        range=position_ranges,
    )
    yaw_histogram, _ = np.histogram(
        action_array[:, 3],
        bins=ACTION_YAW_BINS,
        range=(-np.pi / 2, np.pi / 2),
    )
    gripper_open = action_array[:, 4] >= 0.5
    observed_minimum = action_array[:, :3].min(axis=0)
    observed_maximum = action_array[:, :3].max(axis=0)
    workspace_span_fraction = (
        (observed_maximum - observed_minimum) / (WORKSPACE_UPPER - WORKSPACE_LOWER)
    )
    return {
        "executed_action_count": len(action_array),
        "workspace_lower": WORKSPACE_LOWER.tolist(),
        "workspace_upper": WORKSPACE_UPPER.tolist(),
        "observed_position_minimum": observed_minimum.tolist(),
        "observed_position_maximum": observed_maximum.tolist(),
        "workspace_span_fraction": workspace_span_fraction.tolist(),
        "position_bins": list(ACTION_POSITION_BINS),
        "position_histogram": position_histogram.astype(np.int64).tolist(),
        "yaw_bins": ACTION_YAW_BINS,
        "yaw_histogram": yaw_histogram.astype(np.int64).tolist(),
        "gripper_state_counts": {
            "closed": int(np.count_nonzero(~gripper_open)),
            "open": int(np.count_nonzero(gripper_open)),
        },
        "occupied_position_bins": int(np.count_nonzero(position_histogram)),
        "total_position_bins": int(position_histogram.size),
    }


def plot_action_coverage(
    coverage: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot workspace-position, yaw, and gripper action histograms."""
    position_counts = np.asarray(coverage["position_histogram"], dtype=np.int64)
    position_bins = tuple(int(value) for value in coverage["position_bins"])
    lower = np.asarray(coverage["workspace_lower"], dtype=np.float64)
    upper = np.asarray(coverage["workspace_upper"], dtype=np.float64)
    yaw_counts = np.asarray(coverage["yaw_histogram"], dtype=np.int64)
    gripper_counts = coverage["gripper_state_counts"]
    if position_counts.shape != position_bins or len(position_bins) != 3:
        raise ValueError("Position histogram shape does not match its bin counts")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("Workspace bounds must be increasing xyz vectors")
    if yaw_counts.ndim != 1 or len(yaw_counts) != coverage["yaw_bins"]:
        raise ValueError("Yaw histogram length does not match its bin count")
    if np.any(position_counts < 0) or np.any(yaw_counts < 0):
        raise ValueError("Coverage histograms cannot contain negative counts")
    if any(
        key not in gripper_counts or int(gripper_counts[key]) < 0
        for key in ("closed", "open")
    ):
        raise ValueError("Gripper histogram must contain non-negative open/closed counts")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_counts = position_counts.sum(axis=2).T
    xz_counts = position_counts.sum(axis=1).T
    yz_counts = position_counts.sum(axis=0).T
    projections = (x_counts, xz_counts, yz_counts)
    color_max = max(int(projection.max()) for projection in projections)
    color_max = max(color_max, 1)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    projection_specs = (
        (axes[0, 0], x_counts, (lower[0], upper[0], lower[1], upper[1]), "x (m)", "y (m)", "XY occupancy"),
        (axes[0, 1], xz_counts, (lower[0], upper[0], lower[2], upper[2]), "x (m)", "z (m)", "XZ occupancy"),
        (axes[0, 2], yz_counts, (lower[1], upper[1], lower[2], upper[2]), "y (m)", "z (m)", "YZ occupancy"),
    )
    image = None
    for axis, values, extent, xlabel, ylabel, title in projection_specs:
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=color_max,
        )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if image is None:
        raise RuntimeError("No position projections were created")
    figure.colorbar(
        image,
        ax=list(axes[0, :]),
        label="Executed actions (summed across the third axis)",
        shrink=0.85,
    )

    yaw_axis = axes[1, 0]
    yaw_edges_degrees = np.linspace(-90.0, 90.0, len(yaw_counts) + 1)
    yaw_axis.bar(
        yaw_edges_degrees[:-1],
        yaw_counts,
        width=np.diff(yaw_edges_degrees),
        align="edge",
    )
    yaw_axis.set(
        title="Yaw command histogram",
        xlabel="Yaw (degrees)",
        ylabel="Executed actions",
        xlim=(-90.0, 90.0),
    )

    gripper_axis = axes[1, 1]
    gripper_labels = ("closed", "open")
    gripper_values = [int(gripper_counts[label]) for label in gripper_labels]
    gripper_axis.bar(gripper_labels, gripper_values)
    gripper_axis.set(
        title="Gripper command counts",
        ylabel="Executed actions",
    )
    axes[1, 2].set_visible(False)
    figure.suptitle(
        f"Interaction action coverage: "
        f"{int(coverage['executed_action_count']):,} executed actions"
    )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)
    return output


def plot_action_coverage_from_manifest(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Load a corpus manifest and save its action-coverage histogram figure."""
    manifest = InteractionCorpusSummary.model_validate_json(
        Path(manifest_path).expanduser().read_text(encoding="utf-8")
    )
    if output_path is None:
        output_path = Path(manifest_path).expanduser().parent / "action-coverage.png"
    return plot_action_coverage(manifest.action_coverage, output_path)


def _validate_action_coverage(summary: dict[str, Any]) -> None:
    """Require both gripper states and broad coverage of each workspace axis."""
    gripper_counts = summary["gripper_state_counts"]
    if gripper_counts["closed"] == 0 or gripper_counts["open"] == 0:
        raise RuntimeError(
            "Interaction corpus does not contain both gripper states: "
            f"{gripper_counts}"
        )
    spans = summary["workspace_span_fraction"]
    if any(span < MINIMUM_WORKSPACE_SPAN_FRACTION for span in spans):
        raise RuntimeError(
            "Interaction corpus does not span enough of the workspace; "
            f"observed fractions={spans}"
        )


def _collect_task_episodes(
    executor: ProcessPoolExecutor,
    writer: Any,
    requests: list[InteractionEpisodeRequest],
    *,
    workers: int,
    episode_records: list[InteractionEpisodeRecord],
    executed_actions: list[np.ndarray],
    source_counts: Counter[str],
    task_counts: Counter[str],
    success_counts: Counter[str],
    failure_breakdown: Counter[str],
) -> None:
    """Write every task episode in deterministic request order."""
    pending: dict[Future[InteractionEpisodeResult], InteractionEpisodeRequest] = {}
    completed: dict[int, InteractionEpisodeResult] = {}
    next_to_submit = 0
    next_to_write = 0

    def submit_available() -> None:
        nonlocal next_to_submit
        while (
            len(pending) + len(completed) < workers
            and next_to_submit < len(requests)
        ):
            request = requests[next_to_submit]
            pending[executor.submit(_run_attempt, request)] = request
            next_to_submit += 1

    submit_available()
    while next_to_write < len(requests):
        next_order_index = requests[next_to_write].order_index
        if next_order_index not in completed:
            finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in finished:
                request = pending.pop(future)
                completed[request.order_index] = future.result()
            continue

        result = completed.pop(next_order_index)
        for frame in result.frames:
            writer.add_frame(frame)
            executed_actions.append(np.asarray(frame["action.executed"], dtype=np.float64))
        writer.save_episode(parallel_encoding=False)
        episode_records.append(
            InteractionEpisodeRecord(
                episode_index=len(episode_records),
                order_index=result.order_index,
                seed=result.seed,
                task=result.task,
                requested_source=result.requested_source,
                source=result.source,
                source_id=result.source_id,
                checkpoint_dir=result.checkpoint_dir,
                checkpoint_sha256=result.checkpoint_sha256,
                success=result.success,
                failure_reason=result.failure_reason,
                frame_count=result.frame_count,
            )
        )
        source_counts[result.source] += 1
        task_counts[result.task] += 1
        success_counts["success" if result.success else "failure"] += 1
        if not result.success:
            failure_breakdown[result.failure_reason or "unknown"] += 1
        next_to_write += 1
        if next_to_write % 25 == 0 or next_to_write == len(requests):
            _LOGGER.info(
                "%s interaction progress: %d/%d episodes saved",
                result.task,
                next_to_write,
                len(requests),
            )
        submit_available()


def _verify_interaction_dataset(
    dataset_class: Any,
    repo_id: str,
    root: Path,
    expected_episodes: int,
) -> None:
    """Verify LeRobot metadata, video decoding, source IDs, and success labels."""
    _verify_dataset(dataset_class, repo_id, root, expected_episodes)
    dataset = dataset_class(repo_id=repo_id, root=root, video_backend="pyav")
    for feature_name in ("source_id", "next.success"):
        if feature_name not in dataset.features:
            raise RuntimeError(
                f"Interaction dataset at {root} is missing {feature_name!r}"
            )
    sample = dataset[0]
    for feature_name in ("source_id", "next.success"):
        value = np.asarray(sample[feature_name])
        if value.size != 1:
            raise RuntimeError(
                f"Unexpected interaction feature shape for {feature_name}: "
                f"{value.shape}"
            )
    source_id = int(np.asarray(sample["source_id"]).item())
    if source_id not in SOURCE_IDS.values():
        raise RuntimeError(f"Unexpected interaction source ID {source_id}")
    next_success = int(np.asarray(sample["next.success"]).item())
    if next_success not in {0, 1}:
        raise RuntimeError(f"Unexpected next.success value {next_success}")


def generate_interaction_corpus(
    config: InteractionConfig,
    *,
    output_root: str | Path = _PROJECT_ROOT / "data" / "datasets",
    dataset_name: str | None = None,
) -> InteractionCorpusSummary:
    """Collect and verify the mixed-source L1 interaction corpus."""
    experts, intermediate, checkpoint_digests = _load_checkpoint_catalog(config.tasks)
    requests = build_episode_requests(
        config,
        experts,
        intermediate,
        checkpoint_digests,
    )
    requested_source_counts = Counter(
        request.requested_source for request in requests
    )
    checkpoint_records: list[CheckpointRecord] = []
    for task in config.tasks:
        checkpoint_groups: tuple[
            tuple[Literal["rl-expert", "intermediate-ppo"], list[Path]], ...
        ] = (
            ("rl-expert", [experts[task]]),
            ("intermediate-ppo", intermediate[task]),
        )
        for kind, checkpoint_dirs in checkpoint_groups:
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_records.append(
                    CheckpointRecord(
                        task=task,
                        kind=kind,
                        checkpoint_dir=str(checkpoint_dir),
                        sha256=checkpoint_digests[checkpoint_dir],
                    )
                )
    requested_name = dataset_name or default_dataset_name(config)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", requested_name):
        raise ValueError(
            "dataset_name must contain lowercase letters, digits, or hyphens"
        )
    output_directory = Path(output_root).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    normalized_name = _select_dataset_name(
        output_directory,
        requested_name,
        automatic_versioning=dataset_name is None,
    )
    final_root = output_directory / normalized_name
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{normalized_name}.", dir=output_directory)
    )
    staging_root = staging_parent / "dataset"
    dataset_class = _require_lerobot()
    repo_id = f"blindspot/{normalized_name}"
    writer: Any | None = None
    writer_finalized = False
    episode_records: list[InteractionEpisodeRecord] = []
    executed_actions: list[np.ndarray] = []
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    failure_breakdown: Counter[str] = Counter()
    task_requests = {
        task: [request for request in requests if request.task == task]
        for task in config.tasks
    }

    try:
        writer = dataset_class.create(
            repo_id=repo_id,
            root=staging_root,
            fps=DATASET_FPS,
            robot_type="franka_panda",
            features=interaction_dataset_features(),
            use_videos=True,
            streaming_encoding=True,
            encoder_threads=1,
        )
        context = multiprocessing.get_context("spawn")
        for task in config.tasks:
            requests_for_task = task_requests[task]
            with ProcessPoolExecutor(
                max_workers=config.workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(
                    task,
                    config.model_dump(mode="python"),
                    any(
                        request.requested_source == "scripted-exploration"
                        for request in requests_for_task
                    ),
                ),
            ) as executor:
                _collect_task_episodes(
                    executor,
                    writer,
                    requests_for_task,
                    workers=config.workers,
                    episode_records=episode_records,
                    executed_actions=executed_actions,
                    source_counts=source_counts,
                    task_counts=task_counts,
                    success_counts=success_counts,
                    failure_breakdown=failure_breakdown,
                )

        if writer is None:
            raise RuntimeError("LeRobot interaction writer was not initialized")
        writer.finalize()
        writer_finalized = True
        action_coverage = action_coverage_summary(executed_actions)
        _validate_action_coverage(action_coverage)
        if len(episode_records) != config.episodes:
            raise RuntimeError(
                f"Saved {len(episode_records)}/{config.episodes} corpus episodes"
            )
        if sum(source_counts.values()) != config.episodes:
            raise RuntimeError("Per-source counts do not match the saved episode count")

        summary = InteractionCorpusSummary(
            dataset=normalized_name,
            tasks=list(config.tasks),
            seed_split=config.seed_split,
            total_episodes=len(episode_records),
            source_ids=SOURCE_IDS,
            requested_source_counts=dict(sorted(requested_source_counts.items())),
            source_counts=dict(sorted(source_counts.items())),
            task_counts=dict(sorted(task_counts.items())),
            success_counts=dict(sorted(success_counts.items())),
            failure_breakdown=dict(sorted(failure_breakdown.items())),
            large_noise_position_std_m=config.large_noise_position_std,
            large_noise_yaw_std_degrees=config.large_noise_yaw_degrees,
            reachable_target_count_per_worker=config.reachable_target_count,
            action_coverage=action_coverage,
            checkpoints=checkpoint_records,
            feature_shapes={
                name: list(feature["shape"])
                for name, feature in interaction_dataset_features().items()
            },
            episodes=episode_records,
        )
        (staging_root / "collection.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        plot_action_coverage(
            summary.action_coverage,
            staging_root / "action-coverage.png",
        )
        _verify_interaction_dataset(
            dataset_class,
            repo_id,
            staging_root,
            config.episodes,
        )
        _publish_staged_dataset(staging_root, final_root)
        _LOGGER.info("Wrote interaction corpus to %s", final_root)
        return summary
    finally:
        if writer is not None and not writer_finalized:
            writer.finalize()
        if staging_parent.exists():
            shutil.rmtree(staging_parent)


def _parse_args() -> argparse.Namespace:
    """Parse interaction corpus collection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="Collect only these tasks (default: T1 T2 T3).",
    )
    parser.add_argument(
        "--seed-split",
        choices=tuple(SEED_RANGES),
        default="train",
        help="Seed range for every episode; held-out sets use validation or evaluation.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        help="First seed (default: the start of --seed-split's range).",
    )
    parser.add_argument("--large-noise-position-std", type=float, default=0.05)
    parser.add_argument("--large-noise-yaw-degrees", type=float, default=15.0)
    parser.add_argument("--reachable-target-count", type=int, default=24)
    parser.add_argument("--max-attempts-per-target", type=int, default=50)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "datasets",
    )
    parser.add_argument("--dataset-name")
    parser.add_argument(
        "--plot-coverage",
        type=Path,
        help="Plot an existing corpus manifest without collecting data.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional output PNG path for --plot-coverage.",
    )
    return parser.parse_args()


def main() -> None:
    """Collect the configured mixed-source interaction corpus."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    arguments = _parse_args()
    if arguments.plot_coverage is not None:
        output_path = plot_action_coverage_from_manifest(
            arguments.plot_coverage,
            arguments.plot_output,
        )
        print(output_path)
        return
    if arguments.plot_output is not None:
        raise ValueError("--plot-output requires --plot-coverage")
    config = InteractionConfig(
        episodes=arguments.episodes,
        workers=arguments.workers,
        tasks=tuple(arguments.tasks),
        seed_split=arguments.seed_split,
        seed_start=arguments.seed_start,
        large_noise_position_std=arguments.large_noise_position_std,
        large_noise_yaw_degrees=arguments.large_noise_yaw_degrees,
        reachable_target_count=arguments.reachable_target_count,
        max_attempts_per_target=arguments.max_attempts_per_target,
    )
    summary = generate_interaction_corpus(
        config,
        output_root=arguments.dataset_root,
        dataset_name=arguments.dataset_name,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()

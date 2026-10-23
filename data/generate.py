"""Collect scripted or PPO expert demonstrations into LeRobot datasets."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from common.seeding import named_generators
from envs.pick_place_env import (
    CUBE_COLORS,
    PickPlaceEnv,
    get_instruction_templates,
)
from eval.policy import Policy
from motion.control import ENVIRONMENT_TIMESTEP, WORKSPACE_LOWER, WORKSPACE_UPPER
from skills.expert import ScriptedExpert

DATASET_FPS = round(1.0 / ENVIRONMENT_TIMESTEP)
DATASET_IMAGE_SIZE = 256
DATASET_SCHEMA_VERSION = 2
TRAIN_SEED_START = 0
TRAIN_SEED_STOP = 100_000
VALIDATION_SEED_START = 100_000
VALIDATION_SEED_STOP = 110_000
OBJECT_POSE_NAMES = ("cube_red", "cube_blue", "cube_green", "bowl")
OBJECT_POSE_COMPONENTS = ("x", "y", "z", "qw", "qx", "qy", "qz")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKER_ENVIRONMENT: PickPlaceEnv | None = None
_WORKER_POLICY: Policy | None = None
_WORKER_CONFIG: CollectionConfig | None = None
_LOGGER = logging.getLogger(__name__)


class CollectionConfig(BaseModel):
    """Validate one seeded demonstration collection request."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["T2", "T3", "T3-L"]
    policy_source: Literal["rl", "scripted"] = "rl"
    checkpoint_dir: Path | None = None
    instruction_template_split: Literal["train", "held-out"] = "train"
    episodes: int = Field(default=400, ge=2)
    validation_fraction: float = Field(
        default=0.05, gt=0.0, lt=1.0, allow_inf_nan=False
    )
    workers: int = Field(default=1, ge=1)
    noise_position_std: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    noise_yaw_degrees: float = Field(default=3.0, ge=0.0, allow_inf_nan=False)
    noise_probability: float = Field(default=0.01, ge=0.0, le=1.0)
    max_attempts_per_split: int | None = Field(default=None, ge=1)
    verify_open_loop: bool = False


class EpisodeRollout(BaseModel):
    """Carry one worker's episode frames and outcome back to the writer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int
    success: bool
    target_object: str
    task_label: str
    failure_reason: str | None = None
    frame_count: int
    frames: list[dict[str, Any]] = Field(default_factory=list)
    open_loop_success: bool | None = None


class SplitSummary(BaseModel):
    """Record the accepted episodes and discarded rollout attempts."""

    episodes: int
    attempts: int
    discarded: int
    discard_rate: float
    seeds: list[int]
    failure_breakdown: dict[str, int]
    target_object_counts: dict[str, int]
    task_label_counts: dict[str, int]
    open_loop_successes: int | None = None
    open_loop_rate: float | None = None


class CollectionSummary(BaseModel):
    """Describe the generated train and validation dataset roots."""

    dataset: str
    task: str
    schema_version: int = DATASET_SCHEMA_VERSION
    level: str = "L1"
    policy_source: Literal["rl", "scripted"]
    checkpoint_dir: str | None = None
    checkpoint_sha256: str | None = None
    instruction_template_split: Literal["train", "held-out"] | None = None
    fps: int = DATASET_FPS
    image_size: int = DATASET_IMAGE_SIZE
    total_episodes: int
    train: SplitSummary
    validation: SplitSummary
    noise_position_std_m: float
    noise_yaw_std_degrees: float
    noise_event_probability: float
    feature_shapes: dict[str, list[int]]


def validation_episode_count(episodes: int, fraction: float) -> int:
    """Return a non-empty ceiling-rounded validation holdout size."""
    if episodes < 2:
        raise ValueError("At least two episodes are required for a holdout")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    return min(episodes - 1, max(1, int(np.ceil(episodes * fraction))))


def demo_noise_generator(seed: int) -> np.random.Generator:
    """Create a noise stream independent of the environment's scene and visual streams."""
    return named_generators(seed, ("scene", "visual", "noise"))["noise"]


def perturb_action(
    action: np.ndarray,
    random: np.random.Generator,
    *,
    position_std: float,
    yaw_std: float,
) -> np.ndarray:
    """Add seeded xyz and yaw noise while leaving the gripper command clean."""
    clean_action = np.asarray(action, dtype=np.float64)
    if clean_action.shape != (5,) or not np.all(np.isfinite(clean_action)):
        raise ValueError("Action must be a finite [x, y, z, yaw, g] vector")
    if position_std < 0.0 or yaw_std < 0.0:
        raise ValueError("Noise standard deviations must be non-negative")

    perturbed = clean_action.copy()
    perturbed[:3] += random.normal(0.0, position_std, size=3)
    perturbed[3] += random.normal(0.0, yaw_std)
    return clip_action(perturbed)


def clip_action(action: np.ndarray) -> np.ndarray:
    """Validate and clip an absolute TCP command to the environment limits."""
    command = np.asarray(action, dtype=np.float64)
    if command.shape != (5,) or not np.all(np.isfinite(command)):
        raise ValueError("Action must be a finite [x, y, z, yaw, g] vector")
    action_low = np.array([*WORKSPACE_LOWER, -np.pi / 2, 0.0])
    action_high = np.array([*WORKSPACE_UPPER, np.pi / 2, 1.0])
    return np.clip(command, action_low, action_high)


def maybe_perturb_action(
    action: np.ndarray,
    random: np.random.Generator,
    *,
    probability: float,
    position_std: float,
    yaw_std: float,
) -> np.ndarray:
    """Inject a reproducible Gaussian action disturbance with the given probability."""
    if not 0.0 <= probability <= 1.0 or not np.isfinite(probability):
        raise ValueError("Perturbation probability must be between zero and one")
    clean_action = np.asarray(action, dtype=np.float64)
    if random.random() >= probability:
        return clean_action.copy()
    return perturb_action(
        clean_action,
        random,
        position_std=position_std,
        yaw_std=yaw_std,
    )


def episode_task(
    task: str,
    target_object: str,
    instruction: str | None = None,
) -> str:
    """Build the natural-language task label stored by LeRobot."""
    if task == "T3-L":
        if instruction is None or not instruction.strip():
            raise ValueError("T3-L requires a non-empty language instruction")
        return instruction
    color = target_object.removeprefix("cube_")
    if color not in CUBE_COLORS:
        raise ValueError(f"Unsupported target cube {target_object!r}")
    destination = "target zone" if task == "T2" else "bowl"
    return f"Place the {color} cube in the {destination}."


def dataset_features() -> dict[str, dict[str, Any]]:
    """Return the fixed LeRobot schema used by policies in later phases."""
    state_names = [*(f"joint_{index}" for index in range(1, 8)), "gripper_width"]
    state_names.extend(("tcp_x", "tcp_y", "tcp_z", "tcp_yaw"))
    object_pose_names = [
        f"{object_name}.{component}"
        for object_name in OBJECT_POSE_NAMES
        for component in OBJECT_POSE_COMPONENTS
    ]
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (3, DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": state_names,
        },
        "observation.object_poses": {
            "dtype": "float32",
            "shape": (len(OBJECT_POSE_NAMES) * len(OBJECT_POSE_COMPONENTS),),
            "names": object_pose_names,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (len(OBJECT_POSE_NAMES) * len(OBJECT_POSE_COMPONENTS),),
            "names": object_pose_names,
        },
        "observation.target_pose": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "yaw"],
        },
        "action": {
            "dtype": "float32",
            "shape": (5,),
            "names": ["x", "y", "z", "yaw", "gripper_open"],
        },
        "action.executed": {
            "dtype": "float32",
            "shape": (5,),
            "names": ["x", "y", "z", "yaw", "gripper_open"],
        },
    }


def _make_frame(
    observation: dict[str, Any],
    images: dict[str, np.ndarray],
    action: np.ndarray,
    executed_action: np.ndarray,
    task_label: str,
) -> dict[str, Any]:
    object_poses = observation["object_poses"]
    flattened_poses = np.concatenate(
        [
            np.concatenate(
                (
                    np.asarray(object_poses[name]["position"], dtype=np.float32),
                    np.asarray(object_poses[name]["quaternion_wxyz"], dtype=np.float32),
                )
            )
            for name in OBJECT_POSE_NAMES
        ]
    )
    return {
        **images,
        "observation.state": np.asarray(
            observation["observation.state"], dtype=np.float32
        ),
        "observation.object_poses": flattened_poses,
        "observation.environment_state": flattened_poses.copy(),
        "observation.target_pose": np.asarray(
            observation["target_pose"], dtype=np.float32
        ),
        "action": np.asarray(action, dtype=np.float32),
        "action.executed": np.asarray(executed_action, dtype=np.float32),
        "task": task_label,
    }


def _initialize_worker(config_data: dict[str, Any]) -> None:
    """Create one state environment, renderer, and policy in each worker."""
    global _WORKER_CONFIG, _WORKER_ENVIRONMENT, _WORKER_POLICY
    _WORKER_CONFIG = CollectionConfig.model_validate(config_data)
    _WORKER_ENVIRONMENT = PickPlaceEnv(
        _WORKER_CONFIG.task,
        "L1",
        "state",
        render_mode="rgb_array",
        image_size=DATASET_IMAGE_SIZE,
    )
    if _WORKER_CONFIG.policy_source == "scripted":
        _WORKER_POLICY = ScriptedExpert(_WORKER_CONFIG.task)
    else:
        if _WORKER_CONFIG.checkpoint_dir is None:
            raise ValueError("RL policy collection requires a checkpoint_dir")
        from policies.rl import load_rl_policy

        _WORKER_POLICY = load_rl_policy(_WORKER_CONFIG.checkpoint_dir)
    atexit.register(_close_worker)


def _close_worker() -> None:
    """Release the worker-owned simulation and renderer at process exit."""
    if _WORKER_ENVIRONMENT is not None:
        _WORKER_ENVIRONMENT.close()


def _run_attempt(seed: int) -> EpisodeRollout:
    """Run one noisy expert episode and retain frames only when it succeeds."""
    if _WORKER_CONFIG is None or _WORKER_ENVIRONMENT is None or _WORKER_POLICY is None:
        raise RuntimeError("Demo worker was not initialized")

    environment = _WORKER_ENVIRONMENT
    policy = _WORKER_POLICY
    config = _WORKER_CONFIG
    reset_options: dict[str, Any] = {}
    if config.task == "T3-L":
        templates = get_instruction_templates(
            config.task, config.instruction_template_split
        )
        if not templates:
            raise ValueError(
                f"No {config.instruction_template_split} templates configured for T3-L"
            )
        reset_options["instruction_template"] = templates[seed % len(templates)]
    observation, _ = environment.reset(
        seed=seed,
        options=reset_options or None,
    )
    instruction = observation.get("instruction")
    policy.reset(instruction if isinstance(instruction, str) else None)
    target_object = str(observation["target_object"])
    task_label = episode_task(
        config.task,
        target_object,
        instruction if isinstance(instruction, str) else None,
    )
    random = demo_noise_generator(seed)
    frames: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    while not terminated and not truncated:
        clean_action = np.asarray(policy.act(observation), dtype=np.float64)
        images = environment.observe_images()
        executed_action = maybe_perturb_action(
            clean_action,
            random,
            probability=config.noise_probability,
            position_std=config.noise_position_std,
            yaw_std=float(np.deg2rad(config.noise_yaw_degrees)),
        )
        executed_action = clip_action(executed_action)
        frames.append(
            _make_frame(
                observation,
                images,
                clean_action,
                executed_action,
                task_label,
            )
        )
        observation, _, terminated, truncated, info = environment.step(executed_action)

    success = bool(info.get("success", False))
    open_loop_success: bool | None = None
    if success and config.verify_open_loop:
        open_loop_success = _replay_actions(environment, seed, frames)
    return EpisodeRollout(
        seed=seed,
        success=success,
        target_object=target_object,
        task_label=task_label,
        failure_reason=None if success else info.get("failure_reason"),
        frame_count=len(frames),
        frames=frames if success else [],
        open_loop_success=open_loop_success,
    )


def _replay_actions(
    environment: PickPlaceEnv,
    seed: int,
    frames: list[dict[str, Any]],
) -> bool:
    """Replay clean labels without feedback from the same seeded initial state."""
    observation, _ = environment.reset(seed=seed)
    del observation
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    for frame in frames:
        _, _, terminated, truncated, info = environment.step(frame["action"])
        if terminated or truncated:
            break
    return bool(terminated and not truncated and info.get("success", False))


def _collect_split(
    executor: ProcessPoolExecutor,
    writer: Any,
    *,
    split: Literal["train", "validation"],
    target_episodes: int,
    seed_start: int,
    seed_stop: int,
    config: CollectionConfig,
) -> SplitSummary:
    """Collect a split in seed order while bounding in-flight image memory."""
    available_seeds = seed_stop - seed_start
    attempt_limit = config.max_attempts_per_split or target_episodes * 20
    attempt_limit = min(attempt_limit, available_seeds)
    if attempt_limit < target_episodes:
        raise ValueError(f"Attempt limit for {split} is smaller than its target")

    pending: dict[Future[EpisodeRollout], int] = {}
    completed: dict[int, EpisodeRollout] = {}
    next_seed_to_submit = seed_start
    next_seed_to_process = seed_start
    submitted = 0
    successes: list[int] = []
    failures: dict[str, int] = {}
    target_object_counts: dict[str, int] = {}
    task_label_counts: dict[str, int] = {}
    open_loop_successes = 0
    progress_interval = max(1, min(50, target_episodes // 10))

    def submit_available() -> None:
        nonlocal next_seed_to_submit, submitted
        outstanding_limit = min(config.workers, target_episodes - len(successes))
        while (
            len(pending) + len(completed) < outstanding_limit
            and submitted < attempt_limit
        ):
            seed = next_seed_to_submit
            pending[executor.submit(_run_attempt, seed)] = seed
            next_seed_to_submit += 1
            submitted += 1

    submit_available()
    while len(successes) < target_episodes and (pending or completed):
        if next_seed_to_process in completed:
            rollout = completed.pop(next_seed_to_process)
            next_seed_to_process += 1
            if rollout.success:
                for frame in rollout.frames:
                    writer.add_frame(frame)
                writer.save_episode(parallel_encoding=False)
                successes.append(rollout.seed)
                target_object_counts[rollout.target_object] = (
                    target_object_counts.get(rollout.target_object, 0) + 1
                )
                task_label_counts[rollout.task_label] = (
                    task_label_counts.get(rollout.task_label, 0) + 1
                )
                if rollout.open_loop_success is not None:
                    open_loop_successes += int(rollout.open_loop_success)
            else:
                reason = rollout.failure_reason or "unknown"
                failures[reason] = failures.get(reason, 0) + 1
            attempts_processed = next_seed_to_process - seed_start
            if attempts_processed % progress_interval == 0:
                _LOGGER.info(
                    "%s progress: %d/%d kept after %d attempts",
                    split,
                    len(successes),
                    target_episodes,
                    attempts_processed,
                )
            submit_available()
            continue

        finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in finished:
            seed = pending.pop(future)
            completed[seed] = future.result()

    extra_rollouts = list(completed.values())
    for future, seed in pending.items():
        if not future.cancel():
            extra_rollouts.append(future.result())
    attempts = next_seed_to_process - seed_start + len(extra_rollouts)
    for rollout in extra_rollouts:
        if not rollout.success:
            reason = rollout.failure_reason or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
    discarded = attempts - len(successes)
    if len(successes) != target_episodes:
        raise RuntimeError(
            f"Only collected {len(successes)}/{target_episodes} successful "
            f"{split} episodes after {attempts} attempts; failures={failures}"
        )
    open_loop_trials = target_episodes if config.verify_open_loop else None
    open_loop_rate = (
        open_loop_successes / open_loop_trials if open_loop_trials else None
    )
    if open_loop_rate is not None and open_loop_rate < 0.95:
        raise RuntimeError(
            f"Noise-free open-loop replay success {open_loop_rate:.1%} is below 95%"
        )

    summary = SplitSummary(
        episodes=len(successes),
        attempts=attempts,
        discarded=discarded,
        discard_rate=discarded / attempts,
        seeds=successes,
        failure_breakdown=dict(sorted(failures.items())),
        target_object_counts=dict(sorted(target_object_counts.items())),
        task_label_counts=dict(sorted(task_label_counts.items())),
        open_loop_successes=open_loop_successes if open_loop_trials else None,
        open_loop_rate=open_loop_rate,
    )
    _LOGGER.info(
        "%s: kept %d/%d after %d attempts; discard rate %.1f%%",
        split,
        len(successes),
        target_episodes,
        attempts,
        summary.discard_rate * 100.0,
    )
    return summary


def _require_lerobot() -> Any:
    """Load the pinned LeRobot API only when collection is requested."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "Demo generation requires LeRobot 0.6.1; run "
            "`uv sync --extra imitation --extra rl`."
        ) from error
    return LeRobotDataset


def _verify_dataset(
    dataset_class: Any,
    repo_id: str,
    root: Path,
    expected_episodes: int,
) -> None:
    """Load the written dataset and validate its schema and first sample."""
    dataset = dataset_class(repo_id=repo_id, root=root, video_backend="pyav")
    if dataset.fps != DATASET_FPS or dataset.num_episodes != expected_episodes:
        raise RuntimeError(
            f"Unexpected dataset metadata at {root}: fps={dataset.fps}, "
            f"episodes={dataset.num_episodes}"
        )
    expected_shapes = {
        name: tuple(feature["shape"]) for name, feature in dataset_features().items()
    }
    actual_shapes = {
        name: tuple(feature["shape"])
        for name, feature in dataset.features.items()
        if name in expected_shapes
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected feature shapes at {root}: {actual_shapes}")
    for episode_index in range(min(expected_episodes, 5)):
        episode_dataset = dataset_class(
            repo_id=repo_id,
            root=root,
            episodes=[episode_index],
            video_backend="pyav",
        )
        sample = episode_dataset[0]
        for key in expected_shapes:
            if key not in sample:
                raise RuntimeError(f"Dataset sample at {root} is missing {key!r}")
        for key in ("observation.images.front", "observation.images.wrist"):
            image = np.asarray(sample[key])
            shape = tuple(image.shape)
            expected_image_shapes = {
                (3, DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE),
                (DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE, 3),
            }
            if shape not in expected_image_shapes:
                raise RuntimeError(f"Unexpected decoded image shape for {key}: {shape}")
            minimum_pixel_range = (
                10.0 if np.issubdtype(image.dtype, np.integer) else 10.0 / 255.0
            )
            if float(image.max()) - float(image.min()) <= minimum_pixel_range:
                raise RuntimeError(f"Decoded image for {key} is blank")


def _select_dataset_name(
    output_directory: Path,
    requested_name: str,
    *,
    automatic_versioning: bool,
) -> str:
    """Select an unused dataset name without replacing existing user data."""
    candidate = requested_name
    candidate_path = output_directory / candidate
    if not candidate_path.exists() and not candidate_path.is_symlink():
        return candidate
    if not automatic_versioning:
        raise FileExistsError(f"Dataset output already exists: {candidate_path}")

    run_number = 2
    while True:
        candidate = f"{requested_name}-run-{run_number:02d}"
        candidate_path = output_directory / candidate
        if not candidate_path.exists() and not candidate_path.is_symlink():
            return candidate
        run_number += 1


def _publish_staged_dataset(staging_root: Path, final_root: Path) -> None:
    """Move a verified dataset into place without replacing existing data."""
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(f"Dataset output appeared during collection: {final_root}")
    staging_root.rename(final_root)


def _checkpoint_digest(checkpoint_dir: Path) -> str:
    """Hash the model and normalization files used to reproduce an RL policy."""
    digest = hashlib.sha256()
    for filename in ("model.zip", "vecnormalize.pkl"):
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing RL checkpoint file: {path}")
        digest.update(filename.encode("utf-8"))
        with path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _default_dataset_name(config: CollectionConfig) -> str:
    """Return a compact canonical name for a task and instruction split."""
    task_name = "t3l" if config.task == "T3-L" else config.task.lower()
    held_out_suffix = (
        "-held-out"
        if config.task == "T3-L" and config.instruction_template_split == "held-out"
        else ""
    )
    return f"{task_name}-l1{held_out_suffix}"


def generate_dataset(
    config: CollectionConfig,
    *,
    output_root: str | Path = _PROJECT_ROOT / "data" / "datasets",
    dataset_name: str | None = None,
) -> CollectionSummary:
    """Generate train and held-out validation datasets for one task."""
    if config.verify_open_loop and (
        config.noise_position_std != 0.0 or config.noise_yaw_degrees != 0.0
    ):
        raise ValueError("Open-loop verification requires both noise values to be zero")
    if config.instruction_template_split == "held-out" and config.task != "T3-L":
        raise ValueError("The held-out instruction split is only available for T3-L")
    if config.task == "T3-L" and not get_instruction_templates(
        "T3-L", config.instruction_template_split
    ):
        raise ValueError(
            f"No {config.instruction_template_split} instruction templates configured"
        )
    if config.policy_source == "rl":
        checkpoint_task = "T3" if config.task == "T3-L" else config.task
        checkpoint_dir = (
            config.checkpoint_dir
            or _PROJECT_ROOT
            / "checkpoints"
            / "rl"
            / f"{checkpoint_task.lower()}-l1"
            / "seed-0"
            / "best"
        )
        checkpoint_dir = checkpoint_dir.expanduser().resolve()
        checkpoint_sha256 = _checkpoint_digest(checkpoint_dir)
        config = config.model_copy(update={"checkpoint_dir": checkpoint_dir})
    else:
        if config.checkpoint_dir is not None:
            raise ValueError("checkpoint_dir is only valid when policy_source='rl'")
        checkpoint_sha256 = None

    dataset_class = _require_lerobot()
    requested_name = dataset_name or _default_dataset_name(config)
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
    _LOGGER.info("Collecting %s demonstrations into %s", config.task, final_root)

    validation_episodes = validation_episode_count(
        config.episodes, config.validation_fraction
    )
    train_episodes = config.episodes - validation_episodes
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{normalized_name}.", dir=output_directory)
    )
    writers: dict[str, Any] = {}
    worker_count = min(config.workers, max(train_episodes, validation_episodes))
    worker_config = config.model_copy(update={"workers": worker_count})
    dataset_ids = {
        split: f"blindspot/{normalized_name}-{split}"
        for split in ("train", "validation")
    }

    try:
        for split in ("train", "validation"):
            writers[split] = dataset_class.create(
                repo_id=dataset_ids[split],
                root=staging_root / split,
                fps=DATASET_FPS,
                robot_type="franka_panda",
                features=dataset_features(),
                use_videos=True,
                streaming_encoding=True,
                encoder_threads=1,
            )

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(worker_config.model_dump(mode="python"),),
        ) as executor:
            train_summary = _collect_split(
                executor,
                writers["train"],
                split="train",
                target_episodes=train_episodes,
                seed_start=TRAIN_SEED_START,
                seed_stop=TRAIN_SEED_STOP,
                config=worker_config,
            )
            validation_summary = _collect_split(
                executor,
                writers["validation"],
                split="validation",
                target_episodes=validation_episodes,
                seed_start=VALIDATION_SEED_START,
                seed_stop=VALIDATION_SEED_STOP,
                config=worker_config,
            )

        for writer in writers.values():
            writer.finalize()

        summary = CollectionSummary(
            dataset=normalized_name,
            task=config.task,
            policy_source=config.policy_source,
            checkpoint_dir=(
                str(config.checkpoint_dir)
                if config.checkpoint_dir is not None
                else None
            ),
            checkpoint_sha256=checkpoint_sha256,
            instruction_template_split=(
                config.instruction_template_split if config.task == "T3-L" else None
            ),
            total_episodes=config.episodes,
            train=train_summary,
            validation=validation_summary,
            noise_position_std_m=config.noise_position_std,
            noise_yaw_std_degrees=config.noise_yaw_degrees,
            noise_event_probability=config.noise_probability,
            feature_shapes={
                name: list(feature["shape"])
                for name, feature in dataset_features().items()
            },
        )
        (staging_root / "collection.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        for split, expected_episodes in (
            ("train", train_episodes),
            ("validation", validation_episodes),
        ):
            _verify_dataset(
                dataset_class,
                dataset_ids[split],
                staging_root / split,
                expected_episodes,
            )
        _publish_staged_dataset(staging_root, final_root)
        _LOGGER.info("Wrote LeRobot datasets to %s", final_root)
        return summary
    finally:
        for writer in writers.values():
            writer.finalize()
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _parse_args() -> argparse.Namespace:
    """Parse the demo-generation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("T2", "T3", "T3-L", "all"), default="all")
    parser.add_argument(
        "--policy-source", choices=("rl", "scripted"), default="rl"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--instruction-template-split",
        choices=("train", "held-out"),
        default="train",
    )
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--noise-position-std", type=float, default=0.01)
    parser.add_argument("--noise-yaw-degrees", type=float, default=3.0)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    parser.add_argument("--max-attempts-per-split", type=int)
    parser.add_argument(
        "--dataset-root", type=Path, default=_PROJECT_ROOT / "data" / "datasets"
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--verify-open-loop", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run demonstration collection from command-line arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    arguments = _parse_args()
    tasks = ("T2", "T3") if arguments.task == "all" else (arguments.task,)
    if arguments.dataset_name is not None and len(tasks) > 1:
        raise ValueError("--dataset-name can only be used with one selected task")
    if arguments.checkpoint_dir is not None and len(tasks) > 1:
        raise ValueError("--checkpoint-dir can only be used with one selected task")
    for task in tasks:
        config = CollectionConfig(
            task=task,
            policy_source=arguments.policy_source,
            checkpoint_dir=arguments.checkpoint_dir,
            instruction_template_split=arguments.instruction_template_split,
            episodes=arguments.episodes,
            validation_fraction=arguments.validation_fraction,
            workers=arguments.workers,
            noise_position_std=arguments.noise_position_std,
            noise_yaw_degrees=arguments.noise_yaw_degrees,
            noise_probability=arguments.noise_probability,
            max_attempts_per_split=arguments.max_attempts_per_split,
            verify_open_loop=arguments.verify_open_loop,
        )
        summary = generate_dataset(
            config,
            output_root=arguments.dataset_root,
            dataset_name=arguments.dataset_name,
        )
        print(json.dumps(summary.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()

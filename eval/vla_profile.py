"""Profile-matched VLA evaluation with per-step action/state capture."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from data.vla_subset import VLA_TARGET_COLORS, active_cube_colors_from_positions
from envs.pick_place_env import PickPlaceEnv, get_instruction_templates
from eval.runner import _config_hash, _git_commit
from eval.schema import EpisodeResult, EvaluationResult, EvaluationSummary
from eval.stats import wilson_interval
from eval.vla_trajectory import (
    VLAEpisodeTrajectory,
    VLATrajectoryStep,
    save_vla_trajectory,
)

if TYPE_CHECKING:
    from policies.vla import VLAActionPolicy

_LIMITED_METHOD = "vla-prototype-limited-matched"
_EVALUATION_SEED_START = 1_000_000
_IMAGE_SIZE = 256


def matches_limited_vla_profile(
    instruction: str,
    object_poses: Mapping[str, Any],
) -> tuple[bool, str, tuple[str, ...]]:
    """Check that an episode targets red or blue and has exactly two active cubes."""
    target_color = _instruction_color(instruction)
    if target_color not in VLA_TARGET_COLORS:
        return False, target_color, ()
    active_colors = active_cube_colors_from_positions(object_poses)
    if target_color not in active_colors:
        raise ValueError("T3-L instruction target must be an active cube")
    return len(active_colors) == 2, target_color, active_colors


def evaluate_limited_vla(
    policy: VLAActionPolicy,
    checkpoint: Path,
    level: str,
    episodes: int,
    seed_start: int,
    max_candidates: int,
    *,
    results_dir: Path | None = None,
    trajectories_dir: Path | None = None,
) -> tuple[EvaluationResult, Path, Path, int]:
    """Evaluate only limited-profile scenes and save replayable episode histories."""
    if policy.training_profile != "limited":
        raise ValueError("Limited-scope evaluation requires a limited VLA checkpoint")
    if level not in {"L0", "L1", "L2"}:
        raise ValueError(f"Unsupported T3-L level {level!r}")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if (
        isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or seed_start < _EVALUATION_SEED_START
    ):
        raise ValueError(
            f"Limited evaluation seeds must start at {_EVALUATION_SEED_START} or later"
        )
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < episodes
    ):
        raise ValueError("max_candidates must be at least the requested episode count")

    templates = get_instruction_templates("T3-L", "train")
    if not templates:
        raise ValueError("T3-L has no training instruction templates")
    results_root = (
        results_dir
        if results_dir is not None
        else Path(__file__).resolve().parents[1] / "eval" / "results"
    )
    result_path = (results_root / _LIMITED_METHOD / f"T3-L-{level}.json").resolve()
    trajectory_root = (
        trajectories_dir
        if trajectories_dir is not None
        else result_path.parent / f"T3-L-{level}-trajectories"
    ).resolve()
    trajectory_root.mkdir(parents=True, exist_ok=True)

    environment = PickPlaceEnv(
        "T3-L",
        level,
        obs_mode="state",
        image_size=_IMAGE_SIZE,
    )
    evaluation_episodes: list[EpisodeResult] = []
    accepted_seeds: list[int] = []
    candidates_scanned = 0
    try:
        for seed in tqdm(
            range(seed_start, seed_start + max_candidates),
            total=max_candidates,
            desc="Finding limited-profile seeds",
            unit="seed",
        ):
            candidates_scanned += 1
            template = templates[seed % len(templates)]
            observation, _ = environment.reset(
                seed=seed,
                options={"instruction_template": template},
            )
            instruction = observation.get("instruction")
            object_poses = observation.get("object_poses")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("T3-L environment did not return its instruction")
            if not isinstance(object_poses, Mapping):
                raise TypeError("Limited VLA evaluation requires object poses")
            matched, target_color, active_colors = matches_limited_vla_profile(
                instruction,
                object_poses,
            )
            if not matched:
                continue

            print(
                f"Evaluating seed={seed} target={target_color} "
                f"active_cubes={','.join(active_colors)} "
                f"instruction={instruction!r}"
            )
            trajectory = _run_episode(
                policy,
                environment,
                observation,
                checkpoint=checkpoint,
                seed=seed,
                level=level,
                instruction=instruction,
                instruction_template=template,
                target_color=target_color,
                active_colors=active_colors,
            )
            trajectory_path = trajectory_root / f"seed-{seed}.json"
            save_vla_trajectory(trajectory_path, trajectory)
            evaluation_episodes.append(trajectory.result)
            accepted_seeds.append(seed)
            print(
                f"seed={seed} success={trajectory.result.success} "
                f"failure={trajectory.result.failure_reason} "
                f"steps={trajectory.result.episode_steps} "
                f"trajectory={trajectory_path}"
            )
            if len(evaluation_episodes) == episodes:
                break
    finally:
        environment.close()

    if len(evaluation_episodes) != episodes:
        raise RuntimeError(
            f"Found {len(evaluation_episodes)} of {episodes} matched episodes "
            f"after scanning {candidates_scanned} seeds; partial trajectories "
            f"are in {trajectory_root}"
        )

    success_count = sum(episode.success for episode in evaluation_episodes)
    failures: dict[str, int] = {}
    for episode in evaluation_episodes:
        if not episode.success:
            reason = episode.failure_reason or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
    summary = EvaluationSummary(
        episodes=len(evaluation_episodes),
        successes=success_count,
        success_rate=success_count / len(evaluation_episodes),
        wilson_95=wilson_interval(success_count, len(evaluation_episodes)),
        failure_breakdown=dict(sorted(failures.items())),
    )
    base_config_hash = _config_hash(
        "T3-L",
        level,
        "image",
        image_size=_IMAGE_SIZE,
    )
    limited_config_hash = hashlib.sha256(
        (
            f"{base_config_hash}:limited-red-blue-two-cubes:"
            "training-instruction-templates"
        ).encode()
    ).hexdigest()
    result = EvaluationResult(
        method=_LIMITED_METHOD,
        task="T3-L",
        level=level,
        obs_mode="image",
        seeds=accepted_seeds,
        git_commit=_git_commit(),
        config_hash=limited_config_hash,
        summary=summary,
        episodes=evaluation_episodes,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(result_path)
    return result, result_path, trajectory_root, candidates_scanned


def _run_episode(
    policy: VLAActionPolicy,
    environment: PickPlaceEnv,
    observation: dict[str, Any],
    *,
    checkpoint: Path,
    seed: int,
    level: str,
    instruction: str,
    instruction_template: str,
    target_color: str,
    active_colors: tuple[str, ...],
) -> VLAEpisodeTrajectory:
    """Collect one matched episode and save its transitions for replay."""
    trajectory_steps: list[VLATrajectoryStep] = []
    total_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    error_message: str | None = None
    policy.reset(instruction)
    observation.update(environment.observe_images())
    while not terminated and not truncated:
        state = _state_values(observation)
        object_positions_before = _object_positions(observation)
        try:
            action = np.asarray(policy.act(observation), dtype=np.float32)
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            break
        if action.shape != (5,) or not np.isfinite(action).all():
            error_message = "ValueError: VLA policy returned an invalid 5-D action"
            break
        next_observation, reward, terminated, truncated, info = environment.step(action)
        next_state = _state_values(next_observation)
        object_positions_after = _object_positions(next_observation)
        total_reward += float(reward)
        episode_steps += 1
        trajectory_steps.append(
            VLATrajectoryStep(
                index=episode_steps - 1,
                state=state,
                action=action.astype(np.float64).tolist(),
                next_state=next_state,
                object_positions_before=object_positions_before,
                object_positions_after=object_positions_after,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                success=bool(info.get("success", False)),
                failure_reason=info.get("failure_reason"),
                collision_count=int(info.get("collision_count", 0)),
                episode_time=float(info.get("episode_time", environment.data.time)),
            )
        )
        observation = next_observation
        if not terminated and not truncated:
            observation.update(environment.observe_images())

    success = bool(info.get("success", False)) and error_message is None
    failure_reason = info.get("failure_reason")
    if error_message is not None:
        failure_reason = "policy_error"
    elif not success and failure_reason is None:
        failure_reason = "timeout" if truncated else "unknown"
    result = EpisodeResult(
        seed=seed,
        success=success,
        terminated=terminated,
        truncated=truncated,
        failure_reason=failure_reason,
        episode_steps=episode_steps,
        episode_time=float(environment.data.time),
        collision_count=int(info.get("collision_count", 0)),
        total_reward=total_reward,
        instruction=instruction,
        error=error_message,
    )
    return VLAEpisodeTrajectory(
        level=level,
        seed=seed,
        instruction=instruction,
        instruction_template=instruction_template,
        target_color=target_color,
        active_cube_colors=active_colors,
        checkpoint=str(checkpoint),
        model_id=policy.model_id,
        model_revision=policy.model_revision,
        training_profile="limited",
        git_commit=_git_commit(),
        result=result,
        steps=trajectory_steps,
    )


def _instruction_color(instruction: str) -> str:
    """Extract the single cube color named by a T3-L instruction."""
    colors = re.findall(r"\b(red|blue|green)\b", instruction.lower())
    if len(colors) != 1:
        raise ValueError(f"T3-L instruction must name one cube color: {instruction!r}")
    return colors[0]


def _state_values(observation: Mapping[str, Any]) -> list[float]:
    """Return one finite 12-D proprioceptive state as JSON-compatible values."""
    state = np.asarray(observation.get("observation.state"), dtype=np.float64)
    if state.shape != (12,) or not np.isfinite(state).all():
        raise ValueError("T3-L observation.state must be a finite 12-D vector")
    return state.tolist()


def _object_positions(observation: Mapping[str, Any]) -> dict[str, list[float]]:
    """Serialize every observed object position for trajectory diagnostics."""
    poses = observation.get("object_poses")
    if not isinstance(poses, Mapping):
        raise TypeError("T3-L observations must include object poses")
    positions: dict[str, list[float]] = {}
    for name, pose in poses.items():
        if not isinstance(pose, Mapping):
            raise TypeError(f"Object pose {name!r} must be a mapping")
        position = np.asarray(pose.get("position"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Object pose {name!r} must have a finite 3-D position")
        positions[str(name)] = position.tolist()
    return positions

"""Parallel, seed-reproducible evaluation for Gymnasium policies."""

from __future__ import annotations

import atexit
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module, util
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from common.seeding import named_generators
from envs.pick_place_env import PickPlaceEnv
from eval.policy import Policy
from eval.schema import EpisodeResult, EvaluationResult, EvaluationSummary
from eval.stats import wilson_interval

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TASKS = {"T1", "T2", "T3", "T3-L", "T4", "T5"}
_LEVELS = {"L0", "L1", "L2"}
_WORKER_POLICY: Policy | None = None
_WORKER_ENV: PickPlaceEnv | None = None


def evaluate(
    policy: Policy,
    task: str,
    level: str,
    seeds: list[int] | tuple[int, ...],
    *,
    obs_mode: str = "state",
    workers: int | None = None,
    method: str | None = None,
    results_dir: str | Path | None = None,
) -> EvaluationResult:
    """Run seeded episodes in isolated workers and write a deterministic JSON result."""
    normalized_task = _normalize_task(task)
    if normalized_task not in _TASKS:
        raise ValueError(f"Unsupported task {task!r}")
    if level not in _LEVELS:
        raise ValueError(f"Unsupported level {level!r}")
    if obs_mode not in {"state", "image"}:
        raise ValueError("obs_mode must be 'state' or 'image'")
    normalized_obs_mode = cast(Literal["state", "image"], obs_mode)
    episode_seeds = _validate_seeds(seeds)
    method_name = _method_name(method or policy.__class__.__name__)
    worker_count = _worker_count(workers, len(episode_seeds))
    try:
        policy_payload = pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise TypeError(
            "Policy must be picklable for spawn-based evaluation"
        ) from error

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(policy_payload, normalized_task, level, normalized_obs_mode),
    ) as executor:
        episodes = list(executor.map(_run_episode, episode_seeds))

    success_count = sum(episode.success for episode in episodes)
    failures: dict[str, int] = {}
    for episode in episodes:
        if episode.success:
            continue
        reason = episode.failure_reason or "unknown"
        failures[reason] = failures.get(reason, 0) + 1
    summary = EvaluationSummary(
        episodes=len(episodes),
        successes=success_count,
        success_rate=success_count / len(episodes),
        wilson_95=wilson_interval(success_count, len(episodes)),
        failure_breakdown=dict(sorted(failures.items())),
    )
    result = EvaluationResult(
        method=method_name,
        task=normalized_task,
        level=level,
        obs_mode=normalized_obs_mode,
        seeds=list(episode_seeds),
        git_commit=_git_commit(),
        config_hash=_config_hash(normalized_task, level, normalized_obs_mode),
        summary=summary,
        episodes=episodes,
    )
    root = (
        Path(results_dir)
        if results_dir is not None
        else _PROJECT_ROOT / "eval" / "results"
    )
    result_path = root / method_name / f"{normalized_task}-{level}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(result_path)
    return result


def _initialize_worker(
    policy_payload: bytes,
    task: str,
    level: str,
    obs_mode: str,
) -> None:
    """Create one policy and one environment inside each spawned worker."""
    global _WORKER_ENV, _WORKER_POLICY
    _WORKER_POLICY = pickle.loads(policy_payload)
    _WORKER_ENV = PickPlaceEnv(task, level, obs_mode)
    atexit.register(_close_worker_environment)


def _close_worker_environment() -> None:
    """Release a worker's renderer and simulation when its process exits."""
    if _WORKER_ENV is not None:
        _WORKER_ENV.close()


def _run_episode(seed: int) -> EpisodeResult:
    """Record ordinary episode errors while propagating process-control exceptions."""
    if _WORKER_ENV is None or _WORKER_POLICY is None:
        raise RuntimeError("Evaluation worker was not initialized")

    environment = _WORKER_ENV
    policy = _WORKER_POLICY
    instruction: str | None = None
    episode_steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    error_message: str | None = None
    try:
        _seed_policy_rngs(seed)
        observation, _ = environment.reset(seed=seed)
        observation_instruction = observation.get("instruction")
        instruction = (
            observation_instruction
            if isinstance(observation_instruction, str) and observation_instruction
            else None
        )
        policy.reset(instruction)
        while not terminated and not truncated:
            action = np.asarray(policy.act(observation))
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += float(reward)
            episode_steps += 1
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"

    success = bool(info.get("success", False)) and error_message is None
    failure_reason = info.get("failure_reason")
    if error_message is not None:
        failure_reason = "policy_error"
    elif not success and failure_reason is None:
        failure_reason = "timeout" if truncated else "unknown"
    return EpisodeResult(
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


def _seed_policy_rngs(seed: int) -> None:
    """Seed Python, NumPy, and optional Torch RNGs from the policy stream."""
    policy_random = named_generators(seed, ("policy",))["policy"]
    policy_seed = int(policy_random.integers(0, 2**32, dtype=np.uint32))
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    if util.find_spec("torch") is not None:
        torch = import_module("torch")
        torch.manual_seed(policy_seed)
        if (
            getattr(torch, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            torch.mps.manual_seed(policy_seed)


def _validate_seeds(seeds: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Require a non-empty sequence of distinct non-negative integer seeds."""
    normalized: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("Evaluation seeds must be integers")
        if seed < 0:
            raise ValueError("Evaluation seeds must be non-negative")
        normalized.append(int(seed))
    if not normalized:
        raise ValueError("At least one evaluation seed is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Evaluation seeds must be unique")
    return tuple(normalized)


def _worker_count(workers: int | None, episodes: int) -> int:
    """Choose a bounded process count for the requested number of episodes."""
    if workers is None:
        requested = min(8, os.cpu_count() or 1)
    elif isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer or None")
    else:
        requested = workers
    if requested < 1:
        raise ValueError("workers must be positive")
    return min(requested, episodes)


def _normalize_task(task: str) -> str:
    """Normalize task spelling to the environment's canonical task ID."""
    normalized = task.strip().upper().replace("_", "-")
    return "T3-L" if normalized == "T3L" else normalized


def _method_name(method: str) -> str:
    """Convert a policy label to a safe stable results-directory name."""
    separated = re.sub(r"(?<!^)(?=[A-Z])", "-", method).lower().replace("_", "-")
    normalized = re.sub(r"[^a-z0-9.-]+", "-", separated).strip("-.")
    if not normalized:
        raise ValueError("method must contain at least one alphanumeric character")
    return normalized


def _git_commit() -> str:
    """Return the current repository commit, or unknown outside a Git checkout."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _config_hash(task: str, level: str, obs_mode: str) -> str:
    """Hash the task/level YAML inputs and observation-mode configuration."""
    digest = hashlib.sha256()
    config_dir = _PROJECT_ROOT / "envs" / "configs"
    for filename in ("tasks.yaml", "levels.yaml"):
        digest.update(filename.encode("utf-8"))
        digest.update((config_dir / filename).read_bytes())
    digest.update(
        json.dumps(
            {"task": task, "level": level, "obs_mode": obs_mode},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()

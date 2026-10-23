"""Train one task's Phase 1b pure-PPO RL expert and record a reproducible run."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from common.seeding import VALIDATION_SEED_START
from envs.rl_env import RL_TASKS
from policies.rl import (
    RLTrainingConfig,
    RLTrainingResult,
    resume_ppo_expert,
    train_ppo_expert,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RLRunRecord(BaseModel):
    """Validate the reproducibility record written beside an RL checkpoint."""

    model_config = ConfigDict(extra="forbid")

    task: str
    level: str
    config: RLTrainingConfig
    git_commit: str
    validation_seeds: list[int]
    hardware: dict[str, str | bool]
    result: RLTrainingResult
    resumed_from: str | None = None


def _parse_args() -> argparse.Namespace:
    """Parse one (task, level, seed) training invocation's options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=RL_TASKS)
    parser.add_argument("--level", default="L1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Continue training from an existing checkpoint directory "
        "(model.zip + vecnormalize.pkl) instead of starting a fresh policy. "
        "Only --n-envs/--total-timesteps/--eval-freq/--eval-episodes/--device/"
        "--grasped-start-probability/--destination-start-fraction apply; the "
        "checkpoint's own "
        "hyperparameters are reused unchanged.",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="Early-stop each update's epochs past this KL; 0 disables it.",
    )
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--use-sde",
        action="store_true",
        help="Use generalized state-dependent (temporally correlated) exploration.",
    )
    parser.add_argument("--sde-sample-freq", type=int, default=-1)
    parser.add_argument(
        "--grasped-start-probability",
        type=float,
        default=0.3,
        help="Fraction of object-task training episodes that start with the "
        "cube already grasped (curriculum; ignored for T1 and evaluation).",
    )
    parser.add_argument(
        "--destination-start-fraction",
        type=float,
        default=0.0,
        help="Fraction of grasped starts that begin partway along the path to "
        "the destination, held above it (reverse curriculum for long carries).",
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        help="Steps between resumable intermediate checkpoints; defaults to --eval-freq.",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=2,
        help="How many intermediate step_* checkpoints to retain; older ones are pruned.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _git_commit() -> str:
    """Return the current Git commit or 'unknown' outside a Git checkout."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_json(path: Path, payload: BaseModel) -> None:
    """Atomically persist a validated Pydantic run record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_output_dir(requested_dir: Path, *, automatic_versioning: bool) -> Path:
    """Select an unused checkpoint directory without replacing earlier runs."""
    if not requested_dir.exists() and not requested_dir.is_symlink():
        return requested_dir
    if not automatic_versioning:
        raise FileExistsError(f"Training output already exists: {requested_dir}")

    run_number = 2
    while True:
        candidate = requested_dir.with_name(
            f"{requested_dir.name}-run-{run_number:02d}"
        )
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        run_number += 1


def main() -> None:
    """Train one task/seed pure-PPO expert and save its checkpoint and run record."""
    arguments = _parse_args()
    config = RLTrainingConfig(
        n_envs=arguments.n_envs,
        n_steps=arguments.n_steps,
        batch_size=arguments.batch_size,
        n_epochs=arguments.n_epochs,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        clip_range=arguments.clip_range,
        log_std_init=arguments.log_std_init,
        target_kl=arguments.target_kl if arguments.target_kl > 0 else None,
        ent_coef=arguments.ent_coef,
        use_sde=arguments.use_sde,
        sde_sample_freq=arguments.sde_sample_freq,
        grasped_start_probability=arguments.grasped_start_probability,
        destination_start_fraction=arguments.destination_start_fraction,
        total_timesteps=arguments.total_timesteps,
        eval_freq=arguments.eval_freq,
        eval_episodes=arguments.eval_episodes,
        checkpoint_freq=arguments.checkpoint_freq,
        keep_last=arguments.keep_last,
        seed=arguments.seed,
        device=arguments.device,
    )
    validation_seeds = list(
        range(VALIDATION_SEED_START, VALIDATION_SEED_START + arguments.eval_episodes)
    )
    requested_output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir
        else _PROJECT_ROOT
        / "checkpoints"
        / "rl"
        / f"{arguments.task.lower()}-{arguments.level.lower()}"
        / f"seed-{arguments.seed}"
    )
    output_dir = _select_output_dir(
        requested_output_dir,
        automatic_versioning=arguments.output_dir is None,
    )
    # Created before training so checkpoints can be written during the run and an
    # existing directory fails fast.
    output_dir.mkdir(parents=True, exist_ok=False)

    if arguments.resume_from is not None:
        if not (arguments.resume_from / "model.zip").is_file():
            raise SystemExit(f"No model.zip found under {arguments.resume_from}")
        model, vec_env, result = resume_ppo_expert(
            arguments.resume_from,
            arguments.task,
            arguments.level,
            config,
            validation_seeds,
            output_dir=output_dir,
        )
    else:
        model, vec_env, result = train_ppo_expert(
            arguments.task,
            arguments.level,
            config,
            validation_seeds,
            output_dir=output_dir,
        )
    try:
        model.save(output_dir / "model.zip")
        vec_env.save(str(output_dir / "vecnormalize.pkl"))
    finally:
        vec_env.close()

    run_record = RLRunRecord(
        task=arguments.task,
        level=arguments.level,
        config=config,
        git_commit=_git_commit(),
        validation_seeds=validation_seeds,
        hardware={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": result.device,
            "mps_available": torch.backends.mps.is_available(),
        },
        result=result,
        resumed_from=str(arguments.resume_from) if arguments.resume_from else None,
    )
    _write_json(output_dir / "run.json", run_record)
    print(run_record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

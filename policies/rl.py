"""Pure-PPO training for the Phase 1b RL expert (roadmap 1b.2)."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.rl_env import (
    RL_TASKS,
    DeltaCommandTracker,
    PickPlaceRLEnv,
    make_rl_env,
)
from eval.policy import Policy

# Per-episode statistics (`PickPlaceRLEnv` info["rl_episode"]) logged as stage-funnel rates.
_FUNNEL_KEYS = ("reached", "grasped", "lifted", "placed")


class RLTrainingConfig(BaseModel):
    """Validate the pure-PPO hyperparameters from roadmap 1b.2."""

    model_config = ConfigDict(extra="forbid")

    n_envs: int = Field(default=8, ge=1)
    n_steps: int = Field(default=2048, ge=1)
    batch_size: int = Field(default=1024, ge=1)
    n_epochs: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0, allow_inf_nan=False)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0, allow_inf_nan=False)
    net_arch: tuple[int, ...] = (256, 256)
    log_std_init: float = Field(default=-0.5, allow_inf_nan=False)
    target_kl: float | None = Field(default=0.03, gt=0.0, allow_inf_nan=False)
    ent_coef: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    use_sde: bool = False
    sde_sample_freq: int = Field(default=-1, ge=-1)
    grasped_start_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    total_timesteps: int = Field(default=10_000_000, ge=1)
    eval_freq: int = Field(default=100_000, ge=1)
    eval_episodes: int = Field(default=20, ge=1)
    # None means "same as eval_freq" (see _default_checkpoint_freq).
    checkpoint_freq: int | None = Field(default=None, ge=1)
    keep_last: int = Field(default=2, ge=1)
    seed: int = Field(default=0, ge=0)
    device: str = "auto"

    @model_validator(mode="after")
    def _default_checkpoint_freq(self) -> RLTrainingConfig:
        """Keep checkpoint cadence in sync with eval cadence unless overridden."""
        if self.checkpoint_freq is None:
            self.checkpoint_freq = self.eval_freq
        return self


class RLValidationPoint(BaseModel):
    """Record one validation-seed success-rate reading during training."""

    model_config = ConfigDict(extra="forbid")

    timesteps: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)


class RLTrainingResult(BaseModel):
    """Summarize the learning curve and final validation performance."""

    model_config = ConfigDict(extra="forbid")

    task: str
    level: str
    device: str
    history: list[RLValidationPoint]
    final_success_rate: float = Field(ge=0.0, le=1.0)
    best_success_rate: float = Field(ge=0.0, le=1.0)


class _SuccessRateCallback(BaseCallback):
    """Roll out the current policy on fixed validation seeds every `eval_freq` steps."""

    def __init__(
        self,
        task: str,
        level: str,
        validation_seeds: Sequence[int],
        eval_freq: int,
        eval_episodes: int,
        output_dir: str | Path | None = None,
        checkpoint_freq: int = 100_000,
        keep_last: int = 2,
    ) -> None:
        """Open one reusable evaluation environment for periodic rollouts."""
        super().__init__()
        if not validation_seeds:
            raise ValueError("Validation seeds must be non-empty")
        self._eval_env = PickPlaceRLEnv(task, level)
        self._validation_seeds = list(validation_seeds)
        self._eval_freq = eval_freq
        self._eval_episodes = eval_episodes
        # None disables intermediate and best-checkpoint saving.
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._checkpoint_freq = checkpoint_freq
        self._keep_last = keep_last
        self._best_success_rate = -1.0
        # Anchored to the starting step count in _on_training_start, so resumed
        # runs keep their evaluation and checkpoint cadence.
        self._next_eval_at: int | None = None
        self._next_checkpoint_at: int | None = None
        self._finished_episodes: list[dict[str, float]] = []
        self.history: list[RLValidationPoint] = []

    def _on_training_start(self) -> None:
        """Anchor the first evaluation/checkpoint to this run's starting step count."""
        self._next_eval_at = self.num_timesteps + self._eval_freq
        self._next_checkpoint_at = self.num_timesteps + self._checkpoint_freq

    def _on_step(self) -> bool:
        """Collect finished episodes, log validation success, and checkpoint on schedule."""
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done and "rl_episode" in info:
                self._finished_episodes.append(info["rl_episode"])
        if self.num_timesteps >= self._next_eval_at:
            self._next_eval_at += self._eval_freq
            success_rate = self._evaluate()
            self.history.append(
                RLValidationPoint(timesteps=self.num_timesteps, success_rate=success_rate)
            )
            self.logger.record("eval/success_rate", success_rate)
            print(
                f"[{self.num_timesteps:,} steps] validation success rate: "
                f"{success_rate:.1%}"
            )
            if self._output_dir is not None and success_rate >= self._best_success_rate:
                self._best_success_rate = success_rate
                self._save_snapshot(self._output_dir / "best")
        if self._output_dir is not None and self.num_timesteps >= self._next_checkpoint_at:
            self._next_checkpoint_at += self._checkpoint_freq
            self._save_snapshot(
                self._output_dir / "checkpoints" / f"step_{self.num_timesteps}"
            )
            self._prune_old_checkpoints()
        return True

    def _on_rollout_end(self) -> None:
        """Log training success, the stage funnel, and reward terms for finished episodes."""
        episodes, self._finished_episodes = self._finished_episodes, []
        if not episodes:
            return
        normal = [episode for episode in episodes if not episode["start_grasped"]]
        grasped_start = [episode for episode in episodes if episode["start_grasped"]]
        self.logger.record("rollout/episodes", len(episodes))
        self.logger.record("rollout/grasped_start_fraction", len(grasped_start) / len(episodes))
        if grasped_start:
            self.logger.record(
                "rollout/grasped_start_success_rate",
                float(np.mean([episode["success"] for episode in grasped_start])),
            )
        if not normal:
            return
        for key in normal[0]:
            if key == "start_grasped":
                continue
            value = float(np.mean([episode[key] for episode in normal]))
            if key == "success":
                self.logger.record("rollout/success_rate", value)
            elif key in _FUNNEL_KEYS:
                self.logger.record(f"funnel/{key}", value)
            elif key.startswith("reward/"):
                self.logger.record(key, value)
            else:
                self.logger.record(f"rollout/{key}", value)

    def _save_snapshot(self, destination: Path) -> None:
        """Write model + VecNormalize stats to `destination`, in the same resumable layout as the final checkpoint."""
        staging = destination.with_name(destination.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        self.model.save(staging / "model.zip")
        normalizer = self.model.get_vec_normalize_env()
        if normalizer is not None:
            normalizer.save(str(staging / "vecnormalize.pkl"))
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)

    def _prune_old_checkpoints(self) -> None:
        """Keep only the `keep_last` most recent step_* checkpoint directories."""
        checkpoints_dir = self._output_dir / "checkpoints"
        step_dirs = sorted(
            (
                path
                for path in checkpoints_dir.glob("step_*")
                if path.is_dir() and not path.name.endswith(".tmp")
            ),
            key=lambda path: int(path.name.removeprefix("step_")),
        )
        for stale in step_dirs[: -self._keep_last]:
            shutil.rmtree(stale)

    def _evaluate(self) -> float:
        """Return the success fraction over the fixed validation seeds."""
        normalizer = self.model.get_vec_normalize_env()
        successes = 0
        for index in range(self._eval_episodes):
            seed = self._validation_seeds[index % len(self._validation_seeds)]
            observation, _ = self._eval_env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                normalized = (
                    normalizer.normalize_obs(observation[np.newaxis, :])[0]
                    if normalizer is not None
                    else observation
                )
                action, _ = self.model.predict(normalized, deterministic=True)
                observation, _, terminated, truncated, _ = self._eval_env.step(action)
            successes += int(terminated)
        return successes / self._eval_episodes

    def _on_training_end(self) -> None:
        """Release the evaluation environment's MuJoCo resources."""
        self._eval_env.close()


def _run_training(
    model: PPO,
    task: str,
    level: str,
    config: RLTrainingConfig,
    validation_seeds: Sequence[int],
    output_dir: str | Path | None = None,
) -> RLTrainingResult:
    """Run `model.learn()` with the success-rate callback and summarize the result.

    With an `output_dir`, PPO's training metrics (entropy, std, KL, explained
    variance) and the callback's rollout funnel are written to
    `output_dir/logs/progress.csv` and printed each rollout.
    """
    model.set_logger(
        configure(str(Path(output_dir) / "logs"), ["stdout", "csv"])
        if output_dir is not None
        else Logger(folder=None, output_formats=[])
    )
    callback = _SuccessRateCallback(
        task,
        level,
        validation_seeds,
        config.eval_freq,
        config.eval_episodes,
        output_dir=output_dir,
        checkpoint_freq=config.checkpoint_freq,
        keep_last=config.keep_last,
    )
    # Resumed models count total_timesteps as additional steps; fresh models
    # start from 0 either way.
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )

    if not callback.history:
        raise RuntimeError(
            "Training finished before a single validation evaluation; "
            "lower eval_freq or raise total_timesteps"
        )
    success_rates = [point.success_rate for point in callback.history]
    return RLTrainingResult(
        task=task,
        level=level,
        device=str(model.device),
        history=callback.history,
        final_success_rate=success_rates[-1],
        best_success_rate=max(success_rates),
    )


def _training_vec_env(task: str, level: str, config: RLTrainingConfig) -> SubprocVecEnv:
    """Build the spawn-based training workers, with the grasped-start curriculum."""
    return SubprocVecEnv(
        [
            make_rl_env(
                task,
                level,
                grasped_start_probability=config.grasped_start_probability,
            )
            for _ in range(config.n_envs)
        ],
        start_method="spawn",
    )


def train_ppo_expert(
    task: str,
    level: str,
    config: RLTrainingConfig,
    validation_seeds: Sequence[int],
    output_dir: str | Path | None = None,
) -> tuple[PPO, VecNormalize, RLTrainingResult]:
    """Train one task's pure-PPO expert from scratch, logging validation success.

    If `output_dir` is given, resumable intermediate checkpoints (the last
    `config.keep_last`, every `config.checkpoint_freq` steps) and a `best`
    snapshot (by validation success rate) are written under it as training runs,
    independent of the final checkpoint the caller saves once training finishes.
    """
    if task not in RL_TASKS:
        raise ValueError(
            f"Phase 1b RL supports only non-language tasks {RL_TASKS}, got {task!r}"
        )
    vec_env = VecNormalize(_training_vec_env(task, level, config))
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        target_kl=config.target_kl,
        ent_coef=config.ent_coef,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        policy_kwargs={
            "net_arch": list(config.net_arch),
            "log_std_init": config.log_std_init,
        },
        seed=config.seed,
        device=config.device,
        verbose=0,
    )
    result = _run_training(model, task, level, config, validation_seeds, output_dir)
    return model, vec_env, result


def resume_ppo_expert(
    resume_from: str | Path,
    task: str,
    level: str,
    config: RLTrainingConfig,
    validation_seeds: Sequence[int],
    output_dir: str | Path | None = None,
) -> tuple[PPO, VecNormalize, RLTrainingResult]:
    """Continue training an existing Phase 1b checkpoint for more steps.

    Only `config.n_envs`, `total_timesteps`, `eval_freq`, `eval_episodes`, `seed`,
    `grasped_start_probability`, and `device` apply; the learning rate, clip
    range, network architecture, and other hyperparameters come from the
    checkpoint, as with any `stable_baselines3` resume. Resuming is meant for
    extending a run under the same reward and observation; after changing
    either, train a fresh model (roadmap 1b.2).
    """
    if task not in RL_TASKS:
        raise ValueError(
            f"Phase 1b RL supports only non-language tasks {RL_TASKS}, got {task!r}"
        )
    resume_from = Path(resume_from)
    vec_env = VecNormalize.load(
        str(resume_from / "vecnormalize.pkl"),
        _training_vec_env(task, level, config),
    )
    vec_env.training = True
    model = PPO.load(resume_from / "model.zip", env=vec_env, device=config.device)
    result = _run_training(model, task, level, config, validation_seeds, output_dir)
    return model, vec_env, result


class RLPolicy:
    """Evaluation adapter for a frozen Phase 1b PPO + VecNormalize checkpoint.

    The PPO policy outputs normalized deltas; `DeltaCommandTracker` (the same
    one the training wrapper uses) turns them into absolute `[x, y, z, yaw, g]`
    commands, so this is a drop-in absolute-action `Policy`.
    """

    def __init__(self, model: PPO, normalizer: VecNormalize) -> None:
        """Pair a loaded PPO model with its saved observation-normalization stats."""
        self.model = model
        self.normalizer = normalizer
        self._tracker = DeltaCommandTracker()

    def reset(self, instruction: str | None = None) -> None:
        """Forget the previous episode's command target before a new episode."""
        self._tracker = DeltaCommandTracker()

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Predict an absolute action from the environment's oracle observation."""
        dict_observation = dict(obs)
        if not self._tracker.initialized:
            self._tracker.reset(dict_observation)
        flat = self._tracker.observe(dict_observation)
        normalized_obs = self.normalizer.normalize_obs(flat[np.newaxis, :])[0]
        normalized_action, _ = self.model.predict(normalized_obs, deterministic=True)
        return self._tracker.command(normalized_action)


def load_rl_policy(checkpoint_dir: str | Path, device: str = "cpu") -> Policy:
    """Load a frozen Phase 1b `model.zip` + `vecnormalize.pkl` checkpoint pair."""
    checkpoint_dir = Path(checkpoint_dir)
    model = PPO.load(checkpoint_dir / "model.zip", device=device)
    # VecNormalize.load() needs a venv only to check observation-space shape; every
    # task shares the same observation and [-1, 1] action spaces (envs/rl_env.py).
    template_env = DummyVecEnv([make_rl_env("T1", "L0")])
    try:
        normalizer = VecNormalize.load(
            str(checkpoint_dir / "vecnormalize.pkl"), template_env
        )
    finally:
        template_env.close()
    normalizer.training = False
    normalizer.norm_reward = False
    return RLPolicy(model, normalizer)

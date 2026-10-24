import importlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
rl = importlib.import_module("policies.rl")
RLTrainingConfig = rl.RLTrainingConfig
RLTrainingResult = rl.RLTrainingResult
train_ppo_expert = rl.train_ppo_expert
resume_ppo_expert = rl.resume_ppo_expert


def test_training_run_output_format_offsets_resumed_steps() -> None:
    """Record resumed TensorBoard values from the new run's local step zero."""
    training_run = Mock()
    output_format = rl._TrainingRunOutputFormat(training_run, step_offset=500)
    output_format.write(
        {
            "train/loss": 0.25,
            "train/n_updates": 12,
            "message": "skip",
        },
        {"train/n_updates": ("tensorboard",)},
        step=540,
    )

    training_run.log_metrics.assert_called_once_with(40, {"train/loss": 0.25})


def _tiny_config(**overrides: object) -> RLTrainingConfig:
    """Build a config small enough for one rollout to finish in a test."""
    defaults = {
        "n_envs": 2,
        "n_steps": 32,
        "batch_size": 16,
        "n_epochs": 1,
        "total_timesteps": 64,
        "eval_freq": 32,
        "eval_episodes": 2,
        "device": "cpu",
    }
    defaults.update(overrides)
    return RLTrainingConfig(**defaults)


def test_rejects_language_conditioned_tasks() -> None:
    """Reject T3-L and T5, which need language conditioning, per roadmap 1b."""
    with pytest.raises(ValueError, match="non-language"):
        train_ppo_expert("T3-L", "L0", _tiny_config(), validation_seeds=[100_000])


def test_train_ppo_expert_logs_a_learning_curve_and_saves_a_checkpoint(
    tmp_path: Path,
) -> None:
    """Run one tiny rollout and confirm the result, history, and checkpoint round-trip."""
    config = _tiny_config()
    validation_seeds = [100_000, 100_001]

    model, vec_env, result = train_ppo_expert("T1", "L0", config, validation_seeds)
    try:
        assert isinstance(result, RLTrainingResult)
        assert result.task == "T1"
        assert result.history, "expected at least one validation-seed reading"
        for point in result.history:
            assert point.timesteps > 0
            assert 0.0 <= point.success_rate <= 1.0
        assert result.final_success_rate == result.history[-1].success_rate
        assert result.best_success_rate == max(
            point.success_rate for point in result.history
        )

        model_path = tmp_path / "model.zip"
        stats_path = tmp_path / "vecnormalize.pkl"
        model.save(model_path)
        vec_env.save(str(stats_path))
    finally:
        vec_env.close()

    reloaded = sb3.PPO.load(model_path, device="cpu")
    observation = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = reloaded.predict(observation, deterministic=True)
    assert action.shape == model.action_space.shape
    assert stats_path.is_file()


def test_resume_ppo_expert_continues_from_an_existing_checkpoint(
    tmp_path: Path,
) -> None:
    """Resuming must advance the step count and schedule, not restart from zero."""
    config = _tiny_config()
    validation_seeds = [100_000, 100_001]

    model, vec_env, _ = train_ppo_expert("T1", "L0", config, validation_seeds)
    initial_timesteps = model.num_timesteps
    try:
        model.save(tmp_path / "model.zip")
        vec_env.save(str(tmp_path / "vecnormalize.pkl"))
    finally:
        vec_env.close()

    resumed_model, resumed_vec_env, resumed_result = resume_ppo_expert(
        tmp_path, "T1", "L0", config, validation_seeds
    )
    try:
        assert resumed_model.num_timesteps == initial_timesteps + config.total_timesteps
        assert resumed_result.history, "expected at least one validation-seed reading"
        for point in resumed_result.history:
            assert point.timesteps > initial_timesteps

        resumed_model.save(tmp_path / "resumed-model.zip")
        observation = np.zeros(resumed_model.observation_space.shape, dtype=np.float32)
        action, _ = resumed_model.predict(observation, deterministic=True)
        assert action.shape == resumed_model.action_space.shape
    finally:
        resumed_vec_env.close()


def test_train_ppo_expert_writes_intermediate_and_best_checkpoints(
    tmp_path: Path,
) -> None:
    """With an output_dir, periodic step_* checkpoints and a best/ snapshot appear."""
    config = _tiny_config(keep_last=1)
    validation_seeds = [100_000, 100_001]
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    model, vec_env, _ = train_ppo_expert(
        "T1", "L0", config, validation_seeds, output_dir=output_dir
    )
    try:
        checkpoints_dir = output_dir / "checkpoints"
        step_dirs = sorted(checkpoints_dir.glob("step_*"))
        assert step_dirs, "expected at least one intermediate checkpoint"
        assert len(step_dirs) <= config.keep_last
        for step_dir in step_dirs:
            assert (step_dir / "model.zip").is_file()
            assert (step_dir / "vecnormalize.pkl").is_file()

        best_dir = output_dir / "best"
        assert (best_dir / "model.zip").is_file()
        assert (best_dir / "vecnormalize.pkl").is_file()

        reloaded = sb3.PPO.load(step_dirs[-1] / "model.zip", device="cpu")
        observation = np.zeros(reloaded.observation_space.shape, dtype=np.float32)
        action, _ = reloaded.predict(observation, deterministic=True)
        assert action.shape == model.action_space.shape
    finally:
        vec_env.close()


def test_training_logs_the_stage_funnel_and_ppo_metrics(tmp_path: Path) -> None:
    """With an output_dir, finished episodes and PPO metrics land in progress.csv."""
    import csv

    config = _tiny_config(n_steps=64, batch_size=32, total_timesteps=256, eval_freq=128)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    model, vec_env, _ = train_ppo_expert(
        "T1", "L0", config, [100_000, 100_001], output_dir=output_dir
    )
    vec_env.close()

    with (output_dir / "logs" / "progress.csv").open(encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or ())
    assert {
        "rollout/success_rate",
        "rollout/return",
        "funnel/reached",
        "reward/reach",
        "eval/success_rate",
        "train/std",
        "train/approx_kl",
    } <= columns
    assert model.policy.log_std.detach().numpy() == pytest.approx(
        config.log_std_init, abs=0.1
    )


def test_shared_run_adapter_preserves_ppo_results_and_checkpoint_pair(
    tmp_path: Path,
) -> None:
    """Forward SB3 metrics to the shared run without changing PPO updates."""
    pytest.importorskip("tensorboard")
    import torch

    from common.training import TrainingRun, TrainingSettings

    config = _tiny_config(total_timesteps=128)
    validation_seeds = [100_000, 100_001]
    baseline_model, baseline_env, baseline_result = train_ppo_expert(
        "T1",
        "L0",
        config,
        validation_seeds,
    )
    baseline_state = {
        name: value.detach().clone()
        for name, value in baseline_model.policy.state_dict().items()
    }
    baseline_env.close()

    output_dir = tmp_path / "shared-run"
    settings = TrainingSettings(
        seed=config.seed,
        device="cpu",
        precision="fp32",
        max_grad_norm=0.5,
    )
    with TrainingRun(
        output_dir,
        config=config,
        dataset_id="online-mujoco/T1-L0",
        dataset_version="test-version",
        seeds={"train": [config.seed], "validation": validation_seeds},
        settings=settings,
    ) as training_run:
        model, vec_env, result = train_ppo_expert(
            "T1",
            "L0",
            config,
            validation_seeds,
            output_dir=output_dir,
            training_run=training_run,
        )
        try:
            model.save(output_dir / "model.zip")
            vec_env.save(str(output_dir / "vecnormalize.pkl"))
        finally:
            vec_env.close()

    assert result == baseline_result
    for name, expected in baseline_state.items():
        assert torch.equal(model.policy.state_dict()[name], expected)
    assert (output_dir / "model.zip").is_file()
    assert (output_dir / "vecnormalize.pkl").is_file()
    assert (output_dir / "run.json").is_file()
    assert list((output_dir / "tensorboard").glob("events.out.tfevents.*"))

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from scripts.train_rl import RLRunRecord

    event_accumulator = EventAccumulator(str(output_dir / "tensorboard"))
    event_accumulator.Reload()
    scalar_tags = set(event_accumulator.Tags()["scalars"])
    assert "eval/success_rate" in scalar_tags
    assert "train/learning_rate" in scalar_tags
    assert "train/loss" in scalar_tags
    record_data = training_run.record.model_dump(mode="python")
    record_data.update(
        {
            "task": "T1",
            "level": "L0",
            "validation_seeds": validation_seeds,
            "result": result,
            "resumed_from": None,
        }
    )
    run_record = RLRunRecord.model_validate(record_data)
    assert run_record.status == "completed"
    assert run_record.result == result


def test_rl_policy_emits_absolute_actions_tracking_its_own_target() -> None:
    """RLPolicy integrates deltas from observations alone, like the training wrapper."""
    from envs.pick_place_env import PickPlaceEnv

    config = _tiny_config()
    model, vec_env, _ = train_ppo_expert("T1", "L0", config, [100_000])
    vec_env.training = False
    policy = rl.RLPolicy(model, vec_env)
    environment = PickPlaceEnv("T1", "L0", "state")
    observation, _ = environment.reset(seed=100_000)
    policy.reset()
    for _ in range(5):
        action = policy.act(observation)
        assert environment.action_space.contains(action.astype(np.float32))
        assert action[4] in (0.0, 1.0)
        observation, *_ = environment.step(action)
        np.testing.assert_allclose(
            environment.control.target_position,
            policy._tracker.target_position,
            atol=1e-6,
        )
    environment.close()
    vec_env.close()


def test_load_rl_policy_round_trips_through_the_eval_harness(tmp_path: Path) -> None:
    """Train a tiny checkpoint, reload it, and run it through eval.runner.evaluate().

    This covers the main serialization risk in `load_rl_policy`/`RLPolicy`: the
    evaluation harness pickles the policy for spawn-based worker processes, so a
    frozen checkpoint's `VecNormalize` must not carry its MuJoCo-backed, generally
    unpicklable venv along with it.
    """
    from envs.pick_place_env import PickPlaceEnv
    from eval.runner import evaluate

    config = _tiny_config()
    validation_seeds = [100_000, 100_001]
    model, vec_env, _ = train_ppo_expert("T1", "L0", config, validation_seeds)
    try:
        model.save(tmp_path / "model.zip")
        vec_env.save(str(tmp_path / "vecnormalize.pkl"))
    finally:
        vec_env.close()

    policy = rl.load_rl_policy(tmp_path, device="cpu")
    probe_env = PickPlaceEnv("T1", "L0", "state")
    obs, _ = probe_env.reset(seed=100_000)
    action = policy.act(obs)
    assert action.shape == probe_env.action_space.shape
    assert probe_env.action_space.contains(action.astype(np.float32))
    probe_env.close()

    result = evaluate(
        policy,
        "T1",
        "L0",
        validation_seeds,
        obs_mode="state",
        workers=1,
        method="pure-ppo-test",
        results_dir=tmp_path / "results",
    )
    assert result.summary.episodes == len(validation_seeds)

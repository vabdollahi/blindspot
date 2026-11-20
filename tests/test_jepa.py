import importlib
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
frames = importlib.import_module("encoders.frames")
jepa = importlib.import_module("encoders.jepa")
probes = importlib.import_module("encoders.probes")
vit = importlib.import_module("encoders.vit")
training = importlib.import_module("encoders.training")
environment_module = importlib.import_module("envs.pick_place_env")

_TINY_ENCODER = vit.ViTConfig(width=16, depth=1, heads=2)
FRAME_STRIDE = frames.FRAME_STRIDE


def _tiny_jepa_config() -> object:
    """Return a CPU-sized JEPA with the production image sizes and latent shape."""
    return jepa.JEPAConfig(
        encoder=_TINY_ENCODER,
        latent_dim=8,
        projector_hidden=16,
        predictor_width=16,
        predictor_depth=2,
        predictor_heads=2,
        sigreg_directions=32,
        predictor_dropout=0.0,
    )


def _synthetic_cache(directory: Path, lengths: list[int], *, seed: int = 0) -> object:
    """Write a frame cache of random images, labels, and actions."""
    random = np.random.default_rng(seed)
    sources = [
        frames.EpisodeSource(
            index, 10 + index, "rl-expert" if index % 2 else "demonstration"
        )
        for index in range(len(lengths))
    ]

    def decode(source: object) -> dict[str, object]:
        count = lengths[source.episode_index]
        frame_count = FRAME_STRIDE * count - 1
        environment = np.zeros((count, 28), dtype=np.float32)
        environment[:, 0:3] = random.uniform(0.3, 0.6, (count, 3))
        environment[:, 3] = 1.0
        environment[:, 21:24] = (0.5, 0.1, 0.0)
        environment[:, 24] = 1.0
        state = random.normal(size=(count, 12)).astype(np.float32)
        actions = np.column_stack(
            (
                random.uniform(0.3, 0.6, (frame_count, 3)),
                random.uniform(-1.0, 1.0, frame_count),
                random.integers(0, 2, frame_count),
            )
        )
        return {
            "front": random.integers(
                0, 256, (count, frames.FRONT_SIZE, frames.FRONT_SIZE, 3), dtype=np.uint8
            ),
            "wrist": random.integers(
                0, 256, (count, frames.WRIST_SIZE, frames.WRIST_SIZE, 3), dtype=np.uint8
            ),
            "action_blocks": frames.action_blocks_for_episode(actions),
            "state": state,
            "environment_state": environment,
            "task_label": "Place the red cube in the bowl.",
            "frame_count": frame_count,
        }

    key = frames.FrameCacheKey(
        dataset_id="blindspot/synthetic",
        dataset_version="v3.0",
        collection_sha256="0" * 64,
        info_sha256="0" * 64,
        split=None,
        tasks=["T3"],
    )
    return frames.write_frame_cache(directory, key, sources, lengths, decode)


def test_sigreg_matches_known_answers_for_normal_and_collapsed_latents() -> None:
    """SIGReg is near its null value for N(0, I) and large for collapsed embeddings."""
    torch.manual_seed(0)
    directions = jepa.random_directions(64, 512, device=torch.device("cpu"))
    normal = jepa.sigreg(torch.randn(8192, 64), directions)
    assert abs(float(normal) - jepa.sigreg_null_expectation()) < 0.15

    count = 256
    constant = float(jepa.sigreg(torch.zeros(count, 64), directions))
    t = np.linspace(-4.0, 4.0, 200_001)
    weight = np.exp(-0.5 * t**2)
    expected = count * np.trapezoid(weight * (1.0 - weight) ** 2, t)
    assert constant == pytest.approx(expected, rel=1e-2)

    low_rank = torch.randn(count, 1) @ torch.randn(1, 64)
    rescaled = 5.0 * torch.randn(count, 64)
    for latents in (low_rank, rescaled):
        assert float(jepa.sigreg(latents, directions)) > 10.0 * float(normal)


def test_sigreg_gradients_are_finite_and_push_collapsed_latents_apart() -> None:
    """SIGReg has finite gradients even for a constant batch."""
    directions = jepa.random_directions(16, 64, device=torch.device("cpu"))
    latents = (0.01 * torch.randn(128, 16)).requires_grad_(True)
    jepa.sigreg(latents, directions).backward()
    assert torch.isfinite(latents.grad).all()
    assert latents.grad.abs().sum() > 0


def test_predictor_is_causal_in_time() -> None:
    """Changing later latents or actions never changes earlier predictions."""
    torch.manual_seed(1)
    config = _tiny_jepa_config()
    predictor = jepa.Predictor(config)
    for block in predictor.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.1)
    predictor.eval()
    latents = torch.randn(2, 3, config.latent_dim)
    actions = torch.randn(2, 3, 20)
    baseline, _ = predictor(latents, actions)
    changed_latents = latents.clone()
    changed_latents[:, 2] += 5.0
    changed_actions = actions.clone()
    changed_actions[:, 2] -= 5.0
    changed, _ = predictor(changed_latents, changed_actions)
    torch.testing.assert_close(changed[:, :2], baseline[:, :2])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_adaln_zero_predictor_ignores_actions_at_initialization() -> None:
    """Zero-initialized gates make the initial predictor independent of actions."""
    torch.manual_seed(2)
    predictor = jepa.Predictor(_tiny_jepa_config()).eval()
    latents = torch.randn(4, 3, 8)
    first, gates = predictor(latents, torch.randn(4, 3, 20))
    second, _ = predictor(latents, torch.randn(4, 3, 20))
    torch.testing.assert_close(first, second)
    assert float(gates) == 0.0


def test_action_blocks_hold_the_four_actions_after_each_kept_frame() -> None:
    """Block k holds the executed actions at frames 4k..4k+3, NaN past the end."""
    actions = np.arange(10 * 5, dtype=np.float32).reshape(10, 5)
    blocks = frames.action_blocks_for_episode(actions)
    assert blocks.shape == (3, 4, 5)
    np.testing.assert_array_equal(blocks[1], actions[4:8])
    np.testing.assert_array_equal(blocks[2, :2], actions[8:10])
    assert np.isnan(blocks[2, 2:]).all()
    assert frames.timesteps_for_frames(10) == 3


def test_action_normalization_round_trips_and_spans_unit_range() -> None:
    """Workspace corners map to +/-1, and denormalization inverts normalization."""
    lower = frames.WORKSPACE_LOWER
    upper = frames.WORKSPACE_UPPER
    raw = np.array(
        [
            [*lower, -math.pi / 2, 0.0],
            [*upper, math.pi / 2, 1.0],
            [*(lower + upper) / 2, 0.3, 0.5],
            [*lower, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    normalized = frames.normalize_action_blocks(raw)
    assert normalized.shape == (20,)
    torch.testing.assert_close(normalized[:5], -torch.ones(5))
    torch.testing.assert_close(normalized[5:10], torch.ones(5))
    torch.testing.assert_close(
        frames.denormalize_action_blocks(normalized), torch.as_tensor(raw)
    )


def test_frame_cache_round_trips_and_windows_stay_inside_episodes(
    tmp_path: Path,
) -> None:
    """Every window comes from one episode; the cache pickles without its arrays."""
    cache = _synthetic_cache(tmp_path / "cache", [5, 3, 7])
    reopened = frames.FrameCache(cache.directory)
    np.testing.assert_array_equal(reopened.front, cache.front)
    assert len(pickle.dumps(cache)) < 2_000
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(cache)).wrist, cache.wrist)

    dataset = frames.WindowDataset([cache], frames.all_episodes([cache]), window=4)
    assert len(dataset) == (5 - 3) + (3 - 3) + (7 - 3)
    episode_of_row = cache.episode_of_row()
    for start in dataset.starts[:, 1]:
        assert episode_of_row[start] == episode_of_row[start + 3]
    sample = dataset[0]
    assert sample["front"].shape == (4, frames.FRONT_SIZE, frames.FRONT_SIZE, 3)
    assert sample["actions"].shape == (3, 20)
    torch.testing.assert_close(
        sample["actions"], frames.normalize_action_blocks(cache.action_blocks[0:3])
    )


def test_holdout_split_is_reproducible_and_stratified(tmp_path: Path) -> None:
    """The same seed selects the same held-out episodes from every source."""
    cache = _synthetic_cache(tmp_path / "cache", [4] * 20)
    first = frames.split_holdout_episodes([cache], fraction=0.2, seed=3)
    second = frames.split_holdout_episodes([cache], fraction=0.2, seed=3)
    assert [ref.episode.seed for ref in first[1]] == [
        ref.episode.seed for ref in second[1]
    ]
    assert (
        sorted(ref.episode.source for ref in first[1])
        == ["demonstration"] * 2 + ["rl-expert"] * 2
    )
    assert len(first[0]) + len(first[1]) == 20


def test_batch_sampler_is_seeded_and_resumes_mid_epoch() -> None:
    """Resuming at step k yields exactly the batches an uninterrupted run would."""
    sampler = training.ResumableBatchSampler(10, 3, seed=5)
    stream = iter(sampler)
    uninterrupted = [next(stream) for _ in range(7)]
    resumed_stream = iter(training.ResumableBatchSampler(10, 3, seed=5, start_step=4))
    assert [next(resumed_stream) for _ in range(3)] == uninterrupted[4:]
    assert uninterrupted[:3] != uninterrupted[3:6]
    assert all(len(set(batch)) == 3 for batch in uninterrupted)


def test_resize_frames_accepts_channels_first_and_last() -> None:
    """Channels-first decoder output and channels-last arrays resize identically."""
    images = np.random.default_rng(0).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    last = frames.resize_frames(images, 160)
    first = frames.resize_frames(torch.from_numpy(images).permute(0, 3, 1, 2), 160)
    assert last.shape == (2, 160, 160, 3) and last.dtype == np.uint8
    np.testing.assert_array_equal(first, last)
    np.testing.assert_array_equal(frames.resize_frames(images, 256), images)


def test_target_object_is_parsed_from_the_task_label() -> None:
    """The named cube comes from the task string, never from images."""
    assert (
        frames.target_object_from_task("Place the green cube in the bowl.")
        == "cube_green"
    )
    with pytest.raises(ValueError):
        frames.target_object_from_task("Reach the target point.")


def test_held_and_in_bowl_labels_on_hand_built_poses(tmp_path: Path) -> None:
    """A lifted cube at the TCP is held; a cube inside the bowl band is in the bowl."""
    cache = _synthetic_cache(tmp_path / "cache", [3])
    rows = slice(0, 3)
    environment = cache.environment_state
    state = cache.state
    environment[rows, 21:24] = (0.5, 0.1, 0.0)
    environment[0, 0:3] = (0.4, 0.0, 0.02)
    state[0, 8:11] = (0.4, 0.0, 0.12)
    environment[1, 0:3] = (0.4, 0.0, 0.08)
    state[1, 8:11] = (0.4, 0.0, 0.085)
    environment[2, 0:3] = (0.52, 0.1, 0.026)
    state[2, 8:11] = (0.52, 0.1, 0.15)
    labels = probes.episode_labels(cache, 0, group=0)
    np.testing.assert_array_equal(labels.held, [False, True, False])
    np.testing.assert_array_equal(labels.in_bowl, [False, False, True])
    np.testing.assert_allclose(labels.tcp_xyz[1], (0.4, 0.0, 0.085), atol=1e-6)


def test_in_bowl_label_agrees_with_the_success_predicate_on_resting_cubes() -> None:
    """The shared bowl band matches the environment's at-rest T3 predicate."""
    environment = environment_module.PickPlaceEnv("T3", "L1", "state")
    try:
        environment.reset(seed=7)
        name = environment._target_object
        model, data = environment.model, environment.data
        bowl = data.xpos[model.body("bowl").id].copy()
        joint = model.body_jntadr[model.body(name).id]
        address = model.jnt_qposadr[joint]
        velocity_address = model.jnt_dofadr[joint]
        for offset, expected in (
            ((0.0, 0.0, 0.026), True),
            ((0.1, 0.0, 0.02), False),
            ((0.0, 0.0, 0.07), False),
        ):
            data.qpos[address : address + 3] = bowl + np.asarray(offset)
            data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[velocity_address : velocity_address + 6] = 0.0
            environment_module.mujoco.mj_forward(model, data)
            cube = data.xpos[model.body(name).id]
            assert environment._cube_in_bowl(name) == expected
            assert environment_module.cube_within_bowl(cube, bowl) == expected
    finally:
        environment.close()


def test_grouped_folds_never_split_an_episode() -> None:
    """Every episode's samples share one fold, and all folds are used."""
    groups = np.repeat(np.arange(12), 7)
    folds = probes.grouped_folds(groups, 5)
    for group in range(12):
        assert len(set(folds[groups == group])) == 1
    assert set(folds) == set(range(5))


def test_ridge_probe_recovers_a_linear_map() -> None:
    """Grouped-CV ridge recovers a noiseless linear target almost exactly."""
    random = np.random.default_rng(0)
    features = random.normal(size=(600, 12))
    weights = random.normal(size=(12, 3))
    targets = features @ weights + 1.5
    probe = probes.fit_ridge(features, targets, np.repeat(np.arange(30), 20))
    np.testing.assert_allclose(probe.predict(features[:5]), targets[:5], atol=0.05)


def test_logistic_probe_separates_a_linear_boundary() -> None:
    """The balanced logistic probe classifies a separable flag well."""
    random = np.random.default_rng(1)
    features = random.normal(size=(400, 4))
    labels = features[:, 0] + 0.5 * features[:, 1] > 0.8
    probe = probes.fit_logistic(features, labels, np.repeat(np.arange(20), 20))
    assert probes.balanced_accuracy(labels, probe.scores(features) > 0) > 0.95


def test_flag_metrics_match_hand_computed_values() -> None:
    """Balanced accuracy and AUROC match small hand-computed cases."""
    truth = np.array([True, True, False, False, False])
    assert probes.balanced_accuracy(
        truth, np.array([True, False, False, False, True])
    ) == pytest.approx(0.5 * (1 / 2 + 2 / 3))
    assert probes.auroc(truth, np.array([0.9, 0.4, 0.5, 0.1, 0.2])) == pytest.approx(
        5 / 6
    )
    assert probes.auroc(truth, np.array([1.0, 1.0, 1.0, 0.0, 0.0])) == pytest.approx(
        5 / 6
    )


def test_episode_bootstrap_matches_a_direct_resampling_loop() -> None:
    """Pooled-ratio episode bootstrap equals an explicit loop with the same draws."""
    sums = np.array([2.0, 9.0, 4.0, 1.0])
    counts = np.array([2.0, 3.0, 4.0, 1.0])
    low, high = probes.episode_bootstrap(sums, counts, resamples=200, seed=11)
    random = np.random.default_rng(11)
    estimates = []
    for _ in range(200):
        chosen = random.integers(0, 4, size=(1, 4))[0]
        estimates.append(sums[chosen].sum() / counts[chosen].sum())
    expected = np.quantile(estimates, [0.025, 0.975])
    assert (low, high) == pytest.approx(tuple(expected))
    identical = probes.episode_bootstrap(np.full(5, 3.0), np.full(5, 1.0), resamples=50)
    assert identical == pytest.approx((3.0, 3.0))


def test_jepa_training_resumes_exactly_from_a_checkpoint(tmp_path: Path) -> None:
    """Training 4 steps equals training 2, checkpointing, and resuming for 2 more."""
    pytest.importorskip("torch.utils.tensorboard")
    common_training = importlib.import_module("common.training")
    cache = _synthetic_cache(tmp_path / "cache", [6] * 6)
    train_episodes, _ = frames.split_holdout_episodes([cache], fraction=0.0, seed=0)

    def run(output: Path, stop_at: int, resume: bool = False) -> dict:
        config = training.JEPATrainingConfig(
            model=_tiny_jepa_config(),
            steps=4,
            batch_size=4,
            warmup_steps=1,
            eval_every=2,
            checkpoint_every=2,
            probe_timesteps=100,
            num_workers=0,
            device="cpu",
            precision="fp32",
        )
        settings = common_training.TrainingSettings(
            seed=0, device="cpu", precision="fp32"
        )
        common_training.seed_everything(0)
        with common_training.TrainingRun(
            output,
            config=config,
            dataset_id="blindspot/synthetic",
            dataset_version="v3.0",
            seeds={"train": [0]},
            settings=settings,
            resume_existing=resume,
            step_offset=2 if resume else 0,
        ) as training_run:
            model, result = training.train(
                config,
                run=training_run,
                train_caches=[cache],
                train_episodes=train_episodes,
                holdout_episodes=train_episodes[:2],
                validation_caches=[cache],
                validation_episodes=train_episodes,
                resume_checkpoint=str(output / "checkpoint.pt") if resume else None,
                stop_at=stop_at,
            )
        assert result.steps_completed == stop_at
        return {name: value.clone() for name, value in model.state_dict().items()}

    straight = run(tmp_path / "straight", 4)
    run(tmp_path / "resumed", 2)
    resumed = run(tmp_path / "resumed", 4, resume=True)
    for name, value in straight.items():
        torch.testing.assert_close(resumed[name], value, msg=name)


def test_cached_frames_match_freshly_decoded_frames(tmp_path: Path) -> None:
    """The 5 Hz cache equals LeRobot's own decode, resized, at every kept frame."""
    dataset_dir = Path(__file__).resolve().parents[1] / "data" / "datasets" / "t3-l1"
    if not (dataset_dir / "collection.json").is_file():
        pytest.skip("The t3-l1 demonstration dataset has not been generated")
    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    cache = frames.build_frame_cache(
        dataset_dir, tmp_path, tasks=("T3",), max_episodes=2
    )
    source = lerobot.LeRobotDataset(
        repo_id="blindspot/t3-l1-train",
        root=dataset_dir / "train",
        video_backend="pyav",
    )
    episode = cache.episodes[1]
    first_row = int(source.meta.episodes[1]["dataset_from_index"])
    for timestep in (0, episode.length - 1):
        sample = source[first_row + FRAME_STRIDE * timestep]
        for camera, size, cached in (
            ("front", frames.FRONT_SIZE, cache.front),
            ("wrist", frames.WRIST_SIZE, cache.wrist),
        ):
            decoded = (sample[frames.CAMERA_KEYS[camera]] * 255).round().to(torch.uint8)
            expected = frames.resize_frames(decoded[None], size)[0]
            np.testing.assert_array_equal(cached[episode.start + timestep], expected)
        np.testing.assert_allclose(
            cache.state[episode.start + timestep], sample["observation.state"].numpy()
        )


def test_parameter_count_includes_frozen_parameters() -> None:
    """Frozen encoders report their full size in the evaluation results."""
    model = jepa.JEPA(_tiny_jepa_config())
    trainable = jepa.count_parameters(model.encoder)
    model.encoder.requires_grad_(False)
    assert jepa.count_parameters(model.encoder) == trainable > 0


def test_supervised_output_decoding_inverts_training_standardization() -> None:
    """The ceiling's own output is read back in target units, flags as logits."""
    evaluation = importlib.import_module("encoders.evaluation")
    mean = np.arange(11, dtype=np.float64)
    std = np.linspace(0.5, 1.5, 11)
    run = evaluation.SupervisedRun(
        jepa.SupervisedEncoder(_tiny_jepa_config(), outputs=13), mean, std
    )
    targets = np.random.default_rng(0).normal(size=(4, 11))
    flags = np.array([[2.0, -1.0]] * 4)
    outputs = np.concatenate(((targets - mean) / std, flags), axis=1)
    regressors, classifiers = run.decoders()
    np.testing.assert_allclose(regressors["cube_xyz"](outputs), targets[:, 0:3])
    np.testing.assert_allclose(regressors["tcp_xyz"](outputs), targets[:, 7:10])
    np.testing.assert_allclose(regressors["gripper"](outputs), targets[:, 10:11])
    np.testing.assert_allclose(classifiers["in_bowl"](outputs), -1.0)

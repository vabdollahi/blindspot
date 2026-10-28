import importlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
vla = importlib.import_module("policies.vla")

HIDDEN_SIZE = 4
IMAGE_TOKENS = 2


def _observation(state: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return one valid VLA policy observation."""
    return {
        "observation.state": (
            np.zeros(12, dtype=np.float32) if state is None else state
        ),
        "observation.images.front": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
    }


class SyntheticVLADataset(torch.utils.data.Dataset):
    """Provide deterministic paired-camera instruction/action transitions."""

    def __init__(self, count: int = 24, *, color_dependent: bool = False) -> None:
        values = torch.linspace(-1.0, 1.0, count)
        self.states = torch.stack(
            [values**power for power in range(1, 13)],
            dim=1,
        )
        self.actions = torch.tanh(self.states[:, :5])
        self.actions[:, 4] = (self.states[:, 0] > 0.0).to(torch.float32)
        self.color_dependent = color_dependent
        if color_dependent:
            self.states.zero_()
            self.actions.zero_()
            self.actions[:, 0] = torch.tensor(
                [0.02 if index % 2 == 0 else -0.02 for index in range(count)]
            )
        self.episodes = torch.arange(count, dtype=torch.int64) // 4
        self.front = np.zeros((256, 256, 3), dtype=np.uint8)
        self.wrist = np.full((256, 256, 3), 32, dtype=np.uint8)

    def __len__(self) -> int:
        """Return the number of synthetic transitions."""
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one valid VLA sample."""
        color = "red" if index % 2 == 0 else "blue"
        return {
            "front": self.front,
            "wrist": self.wrist,
            "instruction": f"Place the {color} cube.",
            "state": self.states[index],
            "action": self.actions[index],
            "episode_index": self.episodes[index],
        }


class FakeEncoder:
    """Provide deterministic token-grid features without downloading a VLM."""

    hidden_size = HIDDEN_SIZE
    feature_layer = vla.DEFAULT_FEATURE_LAYER
    model_id = "test/smolvlm"
    model_revision = "test-revision"
    transformers_version = vla.TRANSFORMERS_VERSION
    model_dtype = "float32"
    device = torch.device("cpu")
    parameter_count = 10

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, front, wrist, instructions):
        """Encode image means into grids and the instruction color into context."""
        self.calls += 1
        grids = []
        contexts = []
        for front_image, wrist_image, instruction in zip(
            front, wrist, instructions, strict=True
        ):
            front_value = float(front_image.mean() / 255.0)
            wrist_value = float(wrist_image.mean() / 255.0)
            grids.append(
                [
                    [[front_value + token] * HIDDEN_SIZE for token in range(2)],
                    [[wrist_value - token] * HIDDEN_SIZE for token in range(2)],
                ]
            )
            red = float("red" in instruction)
            contexts.append([red, 1.0 - red, 0.5, 0.25])
        return vla.VLAFeatures(
            torch.tensor(grids, dtype=torch.float32),
            torch.tensor(contexts, dtype=torch.float32),
        )


class FailingEncoder(FakeEncoder):
    """Fail if a cached run tries to run the frozen VLM again."""

    def encode(self, front, wrist, instructions):
        """Reject encoding so tests prove cached features were reused."""
        raise AssertionError("cached features should have been reused")


class FakeProcessor:
    """Match the SmolVLM processor interface for unit tests."""

    image_token_id = 7

    def __init__(self) -> None:
        self.tokenizer = self
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, conversations, **kwargs):
        """Render batches containing two image markers and a text instruction."""
        del kwargs
        self.conversations = conversations
        prompts = []
        for conversation in conversations:
            content = conversation[0]["content"]
            instruction = next(
                item["text"] for item in content if item["type"] == "text"
            )
            prompts.append(f"<image><image>{instruction}")
        return prompts

    def __call__(self, *, text, images, **kwargs):
        """Return fake tokens with two image tokens for each camera."""
        self.calls.append(kwargs)
        if any(len(image_pair) != 2 for image_pair in images):
            raise ValueError("expected two images per sample")
        ids = [
            [1, 7, 7, 7, 7, 2 if "blue" in prompt else 3, 4 if "blue" in prompt else 5]
            for prompt in text
        ]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), len(ids[0]), dtype=torch.int64),
        }


class FakeBaseModel(torch.nn.Module):
    """Expose per-layer hidden states like SmolVLM2's multimodal base model."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, input_ids, **kwargs):
        """Expand token IDs and positions into deterministic layer states."""
        if not kwargs.get("output_hidden_states"):
            raise ValueError("expected hidden states for every layer")
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
        base = input_ids.to(torch.float32) + positions / 10.0
        hidden = base.unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE) * self.scale
        return SimpleNamespace(
            hidden_states=tuple(hidden * (layer + 1) for layer in range(3))
        )


class FakeBackbone(torch.nn.Module):
    """Expose the config and base-model attributes used by the adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBaseModel()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=HIDDEN_SIZE, num_hidden_layers=2),
        )


class FakeNumericRows:
    """Mimic the numeric LeRobot view used for action and state columns."""

    def __init__(self, rows: list[dict[str, np.ndarray]]) -> None:
        self.rows = rows

    def with_format(self, format_name: str):
        """Return the same lightweight view for the numpy format."""
        if format_name != "numpy":
            raise ValueError("expected numpy format")
        return self

    def __len__(self) -> int:
        """Return the number of numeric rows."""
        return len(self.rows)

    def __getitem__(self, index):
        """Return one numeric row."""
        return self.rows[index]


class FakeLeRobotDataset:
    """Mimic the LeRobot features consumed by VLAFrameDataset."""

    def __init__(self) -> None:
        self.meta = SimpleNamespace(
            info=SimpleNamespace(codebase_version="test-dataset-v1")
        )
        self.rows = []
        numeric = []
        for index in range(2):
            action = np.full(5, index, dtype=np.float32)
            self.rows.append(
                {
                    "observation.images.front": np.zeros((256, 256, 3), dtype=np.uint8),
                    "observation.images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
                    "task": f"Place the {'red' if index == 0 else 'blue'} cube.",
                    "action": action,
                    "action.executed": np.full(5, 99, dtype=np.float32),
                }
            )
            numeric.append(
                {
                    "observation.state": np.full(12, index, dtype=np.float32),
                    "action": action,
                    "episode_index": np.array(index, dtype=np.int64),
                }
            )
        self.numeric = FakeNumericRows(numeric)

    def select_columns(self, names):
        """Select numeric data columns for the trainer adapter."""
        if names != ["observation.state", "action", "episode_index"]:
            raise ValueError("unexpected numeric columns")
        return self.numeric

    def __len__(self) -> int:
        """Return the number of LeRobot rows."""
        return len(self.rows)

    def __getitem__(self, index: int):
        """Return one multimodal LeRobot row."""
        return self.rows[index]


def _training_config(**updates) -> object:
    """Return a small CPU training configuration for synthetic data."""
    settings = {
        "epochs": 12,
        "batch_size": 12,
        "encoding_batch_size": 6,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "device": "cpu",
        "seed": 17,
        "patience": 12,
    }
    settings.update(updates)
    return vla.VLAConfig(**settings)


def _policy(head, encoder) -> object:
    """Bind a head and fake encoder into the shared policy adapter."""
    return vla.VLAActionPolicy(
        head,
        encoder,
        model_id=encoder.model_id,
        model_revision=encoder.model_revision,
        head_state=head.state_dict(),
        image_tokens=head.image_tokens,
        device="cpu",
    )


def test_frozen_encoder_returns_image_token_grids_and_instruction_context() -> None:
    """Freeze the VLM, avoid image tiling, and split image tokens per camera."""
    backbone = FakeBackbone()
    processor = FakeProcessor()
    encoder = vla.FrozenSmolVLMEncoder(
        backbone,
        processor,
        model_id="test/smolvlm",
        model_revision="test-revision",
        transformers_version=vla.TRANSFORMERS_VERSION,
        device="cpu",
        feature_layer=1,
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    features = encoder.encode(
        [image, image],
        [image, image],
        ["Place the red cube.", "Place the blue cube."],
    )

    assert processor.calls[0]["do_image_splitting"] is False
    content_types = [item["type"] for item in processor.conversations[0][0]["content"]]
    assert content_types == ["text", "image", "image"]
    assert features.grid.shape == (2, 2, IMAGE_TOKENS, HIDDEN_SIZE)
    assert features.context.shape == (2, HIDDEN_SIZE)
    # Layer 1 doubles the token value; image tokens (id 7) sit at positions 1-4.
    assert torch.allclose(features.grid[0, 0, :, 0], torch.tensor([14.2, 14.4]))
    assert torch.allclose(features.grid[0, 1, :, 0], torch.tensor([14.6, 14.8]))
    assert not torch.equal(features.context[0], features.context[1])
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_frozen_encoder_rejects_a_missing_feature_layer() -> None:
    """Reject decoder layers that the VLM does not have."""
    with pytest.raises(ValueError, match="feature layer 3"):
        vla.FrozenSmolVLMEncoder(
            FakeBackbone(),
            FakeProcessor(),
            model_id="test/smolvlm",
            model_revision="test-revision",
            transformers_version=vla.TRANSFORMERS_VERSION,
            device="cpu",
            feature_layer=3,
        )


def test_vla_frame_dataset_uses_clean_action_and_task_text() -> None:
    """Keep instruction text and clean action separate from executed noise."""
    dataset = vla.VLAFrameDataset(FakeLeRobotDataset())

    sample = dataset[1]

    assert sample["instruction"] == "Place the blue cube."
    assert torch.equal(sample["action"], torch.ones(5))
    assert sample["front"].shape == (256, 256, 3)
    assert dataset.dataset_version == "test-dataset-v1"


def test_vla_frame_dataset_applies_limited_source_indices() -> None:
    """Samples use only the selected frame rows."""
    dataset = vla.VLAFrameDataset(FakeLeRobotDataset(), frame_indices=[1])

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["instruction"] == "Place the blue cube."
    assert torch.equal(sample["action"], torch.ones(5))


def test_vla_training_reduces_action_loss_with_frozen_encoder() -> None:
    """Cache frozen features once and train the small action head on clean labels."""
    dataset = SyntheticVLADataset()

    head, result, encoder = vla.train_vla(
        dataset,
        dataset,
        _training_config(),
        encoder=FakeEncoder(),
    )

    assert result.best_validation_l1 < result.initial_validation_l1
    assert result.optimizer_steps > 0
    assert result.encoding.encoded_samples == 2 * len(dataset)
    assert result.encoding.cached_samples == 0
    assert result.encoding.image_tokens_per_camera == IMAGE_TOKENS
    assert result.encoding.model_revision == encoder.model_revision
    assert 0.0 <= result.validation_gripper_accuracy <= 1.0
    assert result.gripper_positive_weight > 0.0
    assert head.image_tokens == IMAGE_TOKENS
    assert next(head.parameters()).device.type == "cpu"


def test_vla_head_normalizes_inputs_and_residual_targets_from_training_data() -> None:
    """Fit state, feature, and TCP-delta statistics on the training split."""
    dataset = SyntheticVLADataset()

    head, _, _ = vla.train_vla(
        dataset,
        dataset,
        _training_config(epochs=1),
        encoder=FakeEncoder(),
    )
    deltas = dataset.actions[:, :4] - dataset.states[:, 8:12]

    assert torch.allclose(head.state_mean, dataset.states.mean(dim=0))
    assert torch.allclose(head.delta_mean, deltas.mean(dim=0))
    assert torch.allclose(head.delta_std, deltas.std(dim=0, unbiased=False))
    assert torch.allclose(head.context_mean, torch.tensor([0.5, 0.5, 0.5, 0.25]))
    assert torch.all(head.context_std >= 1e-3)


def test_vla_feature_ablation_shows_reliance_on_instruction_features() -> None:
    """Report a larger error when the frozen-VLM inputs are replaced by means."""
    dataset = SyntheticVLADataset(count=32, color_dependent=True)

    _, result, _ = vla.train_vla(
        dataset,
        dataset,
        _training_config(epochs=40, batch_size=8, patience=40),
        encoder=FakeEncoder(),
    )

    assert result.validation_action_mae["x"] < 0.005
    assert result.ablated_validation_action_mae["x"] > 0.015
    assert result.ablated_validation_l1 > result.best_validation_l1


def test_vla_checkpoint_selection_uses_weighted_validation_objective() -> None:
    """Keep the epoch that best matches the action and binary-gripper losses."""
    dataset = SyntheticVLADataset()
    _, result, _ = vla.train_vla(
        dataset,
        dataset,
        _training_config(epochs=6, batch_size=6, seed=23, patience=6),
        encoder=FakeEncoder(),
    )
    validation_points = [(0, result.initial_validation_loss)]
    validation_points.extend(
        (metric.epoch, metric.validation_loss) for metric in result.history
    )
    expected_epoch, expected_loss = min(validation_points, key=lambda point: point[1])

    assert result.best_epoch == expected_epoch
    assert result.best_validation_loss == pytest.approx(expected_loss)


def test_vla_feasibility_limits_encoded_examples_and_optimizer_steps() -> None:
    """Honor the short-run caps used for device integration checks."""
    dataset = SyntheticVLADataset()
    config = vla.VLAConfig(
        epochs=2,
        batch_size=2,
        encoding_batch_size=2,
        device="cpu",
        max_encoding_samples=3,
        max_train_steps=2,
    )

    _, result, _ = vla.train_vla(
        dataset,
        dataset,
        config,
        encoder=FakeEncoder(),
    )

    assert result.encoding.encoded_samples == 6
    assert result.optimizer_steps == 2


def test_vla_max_steps_can_span_multiple_epochs() -> None:
    """Keep training until the total step ceiling is reached."""
    dataset = SyntheticVLADataset()

    _, result, _ = vla.train_vla(
        dataset,
        dataset,
        _training_config(
            epochs=4,
            batch_size=2,
            encoding_batch_size=2,
            seed=19,
            max_encoding_samples=5,
            max_train_steps=4,
        ),
        encoder=FakeEncoder(),
    )

    assert result.optimizer_steps == 4
    assert len(result.history) == 2


def test_vla_feature_cache_reuses_matching_encoded_splits(tmp_path) -> None:
    """Skip the frozen VLM when the data selection and encoder identity match."""
    dataset = SyntheticVLADataset(count=8)
    cache = vla.VLAFeatureCache(
        directory=tmp_path,
        split_keys={"train": {"frames": [0, 1]}, "validation": {"frames": [2]}},
    )
    encoder = FakeEncoder()
    _, first, _ = vla.train_vla(
        dataset,
        dataset,
        _training_config(epochs=1),
        encoder=encoder,
        feature_cache=cache,
    )

    _, cached, _ = vla.train_vla(
        dataset,
        dataset,
        _training_config(epochs=1),
        encoder=FailingEncoder(),
        feature_cache=cache,
    )
    changed = cache.model_copy(
        update={"split_keys": {"train": {"frames": [0]}, "validation": {"frames": [2]}}}
    )
    calls_before = encoder.calls
    vla.train_vla(
        dataset,
        dataset,
        _training_config(epochs=1),
        encoder=encoder,
        feature_cache=changed,
    )

    assert len(list(tmp_path.glob("*.pt"))) == 3
    assert first.encoding.cached_samples == 0
    assert cached.encoding.cached_samples == 2 * len(dataset)
    assert cached.encoding.encoding_seconds == pytest.approx(
        first.encoding.encoding_seconds
    )
    assert cached.initial_validation_l1 == pytest.approx(first.initial_validation_l1)
    assert encoder.calls > calls_before


def test_vla_feature_cache_rejects_a_hit_with_a_different_row_count(tmp_path) -> None:
    """Re-encode when a matching key no longer describes the split's rows."""
    cache = vla.VLAFeatureCache(
        directory=tmp_path,
        split_keys={"train": {"frames": "all"}, "validation": {"frames": "all"}},
    )
    encoder = FakeEncoder()
    vla.train_vla(
        SyntheticVLADataset(count=8),
        SyntheticVLADataset(count=8),
        _training_config(epochs=1),
        encoder=encoder,
        feature_cache=cache,
    )
    calls_before = encoder.calls

    _, result, _ = vla.train_vla(
        SyntheticVLADataset(count=12),
        SyntheticVLADataset(count=12),
        _training_config(epochs=1),
        encoder=encoder,
        feature_cache=cache,
    )

    assert encoder.calls > calls_before
    assert result.encoding.cached_samples == 0
    assert result.encoding.encoded_samples == 24


def test_vla_resume_keeps_the_head_unless_it_beats_the_recorded_best() -> None:
    """Use the recorded best loss as the threshold for a resumed run."""
    dataset = SyntheticVLADataset(count=12)
    encoder = FakeEncoder()
    config = _training_config(epochs=2, patience=4)
    head, first_result, encoder = vla.train_vla(
        dataset, dataset, config, encoder=encoder
    )
    original_state = {name: value.clone() for name, value in head.state_dict().items()}

    continuation, result, _ = vla.train_vla(
        dataset,
        dataset,
        config,
        encoder=encoder,
        initial_head=head,
        optimizer_step_offset=first_result.optimizer_steps,
        epoch_offset=len(first_result.history),
        initial_best_epoch=first_result.best_epoch,
        initial_best_loss=0.0,
    )

    assert result.best_epoch == first_result.best_epoch
    assert result.best_validation_loss == 0.0
    assert all(
        torch.equal(continuation.state_dict()[name], value)
        for name, value in original_state.items()
    )


def test_vla_training_warm_starts_and_offsets_history_and_steps() -> None:
    """Continue a saved action head with global epoch and optimizer-step counts."""
    dataset = SyntheticVLADataset(count=12)
    encoder = FakeEncoder()
    config = _training_config(epochs=2, patience=4)
    head, first_result, encoder = vla.train_vla(
        dataset,
        dataset,
        config,
        encoder=encoder,
    )
    continuation, continued_result, _ = vla.train_vla(
        dataset,
        dataset,
        config.model_copy(update={"epochs": 1}),
        encoder=encoder,
        initial_head=head,
        optimizer_step_offset=first_result.optimizer_steps,
        epoch_offset=len(first_result.history),
        initial_best_epoch=first_result.best_epoch,
    )

    assert continued_result.initial_validation_l1 == pytest.approx(
        first_result.best_validation_l1
    )
    assert continued_result.optimizer_steps == first_result.optimizer_steps + 1
    assert continued_result.history[0].epoch == len(first_result.history) + 1
    assert continued_result.best_epoch >= first_result.best_epoch
    assert next(continuation.parameters()).device.type == "cpu"


def test_vla_warm_start_rejects_different_training_statistics() -> None:
    """Refuse to continue a head whose state normalization came from other data."""
    encoder = FakeEncoder()
    head, _, _ = vla.train_vla(
        SyntheticVLADataset(count=12),
        SyntheticVLADataset(count=12),
        _training_config(epochs=1),
        encoder=encoder,
    )

    with pytest.raises(ValueError, match="normalization"):
        vla.train_vla(
            SyntheticVLADataset(count=16),
            SyntheticVLADataset(count=16),
            _training_config(epochs=1),
            encoder=encoder,
            initial_head=head,
            optimizer_step_offset=1,
            epoch_offset=1,
        )


def test_vla_policy_clips_finite_actions_and_requires_instruction() -> None:
    """Use the shared policy protocol and enforce absolute action limits."""
    encoder = FakeEncoder()
    head = vla.VLAActionHead(encoder.hidden_size, IMAGE_TOKENS)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.layers[-1].bias.copy_(torch.tensor([5.0, -5.0, 5.0, 5.0, 2.0]))
    policy = _policy(head, encoder)

    with pytest.raises(RuntimeError, match="instruction"):
        policy.act(_observation())
    policy.reset("Place the red cube.")
    action = policy.act(_observation())

    assert action.shape == (5,)
    assert action.dtype == np.float32
    assert np.all(action >= vla._ACTION_LOW)
    assert np.all(action <= vla._ACTION_HIGH)
    assert action[4] == 1.0
    with torch.no_grad():
        head.layers[-1].bias[4] = -2.0
    assert policy.act(_observation())[4] == 0.0
    assert action[4] == 1.0


def test_vla_policy_adds_the_predicted_delta_to_the_current_tcp_pose() -> None:
    """Emit an absolute action equal to the TCP pose plus the scaled delta."""
    encoder = FakeEncoder()
    head = vla.VLAActionHead(encoder.hidden_size, IMAGE_TOKENS)
    with torch.no_grad():
        for parameter in head.parameters():
            if parameter.ndim > 1:
                parameter.zero_()
        head.layers[-1].bias.copy_(torch.tensor([1.0, -1.0, 0.5, 0.0, -1.0]))
        head.delta_mean.copy_(torch.tensor([0.0, 0.0, 0.0, 0.02]))
        head.delta_std.copy_(torch.tensor([0.01, 0.01, 0.01, 0.05]))
    state = np.zeros(12, dtype=np.float32)
    state[8:12] = [0.5, 0.0, 0.2, 0.1]
    policy = _policy(head, encoder)
    policy.reset("Place the blue cube.")

    action = policy.act(_observation(state))

    assert action == pytest.approx([0.51, -0.01, 0.205, 0.12, 0.0])


def test_vla_checkpoint_loads_lazily_and_survives_policy_pickling(
    tmp_path,
    monkeypatch,
) -> None:
    """Restore the small head without pickling or downloading the frozen VLM."""
    encoder = FakeEncoder()
    head = vla.VLAActionHead(encoder.hidden_size, IMAGE_TOKENS)
    checkpoint = vla.save_vla_checkpoint(
        head,
        encoder.model_id,
        encoder.model_revision,
        tmp_path / "prototype.pt",
        training_profile="limited",
    )
    requested = {}

    def fake_loader(model_id, revision, *, device, feature_layer):
        requested["feature_layer"] = feature_layer
        return encoder

    monkeypatch.setattr(vla, "load_frozen_smolvlm", fake_loader)
    policy = pickle.loads(pickle.dumps(vla.load_vla_policy(checkpoint, device="cpu")))
    policy.reset("Place the blue cube.")

    assert policy.training_profile == "limited"
    assert policy.image_tokens == IMAGE_TOKENS
    action = policy.act(_observation())

    assert requested["feature_layer"] == vla.DEFAULT_FEATURE_LAYER
    assert action.shape == (5,)
    assert np.isfinite(action).all()


def test_vla_checkpoint_rejects_superseded_last_token_heads(tmp_path) -> None:
    """Refuse checkpoints from before the token-grid head with a clear message."""
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 2,
            "model_id": "test/smolvlm",
            "model_revision": "test-revision",
            "training_profile": "limited",
            "gripper_mode": "binary",
            "hidden_size": HIDDEN_SIZE,
            "state_dict": {},
        },
        legacy_checkpoint,
    )

    with pytest.raises(ValueError, match="superseded last-token head"):
        vla.load_vla_policy(legacy_checkpoint, device="cpu")


def test_load_vla_training_head_checks_checkpoint_identity(tmp_path) -> None:
    """Only warm-start with the same model, revision, profile, and feature layer."""
    encoder = FakeEncoder()
    head = vla.VLAActionHead(encoder.hidden_size, IMAGE_TOKENS)
    checkpoint = vla.save_vla_checkpoint(
        head,
        encoder.model_id,
        encoder.model_revision,
        tmp_path / "warm-start.pt",
        training_profile="limited",
    )
    identity = {
        "model_id": encoder.model_id,
        "model_revision": encoder.model_revision,
        "training_profile": "limited",
        "hidden_size": encoder.hidden_size,
        "feature_layer": vla.DEFAULT_FEATURE_LAYER,
    }

    loaded = vla.load_vla_training_head(checkpoint, **identity)

    assert all(
        torch.equal(loaded.state_dict()[name], value)
        for name, value in head.state_dict().items()
    )
    with pytest.raises(ValueError, match="training profile"):
        vla.load_vla_training_head(
            checkpoint, **{**identity, "training_profile": "full"}
        )
    with pytest.raises(ValueError, match="feature layer"):
        vla.load_vla_training_head(checkpoint, **{**identity, "feature_layer": 8})

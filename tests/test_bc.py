import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "torchvision"
)  # policies.bc needs it; the "rl" extra installs torch without it
bc = importlib.import_module("policies.bc")


class SyntheticBCDataset(torch.utils.data.Dataset):
    """Small deterministic transitions for CPU-only BC training checks."""

    def __init__(self, policy_type: str, count: int = 16) -> None:
        self.policy_type = policy_type
        values = torch.linspace(-1.0, 1.0, count)
        self.state = torch.stack([values**power for power in range(1, 13)], dim=1)
        self.object_poses = torch.stack(
            [values * (index + 1) / 28 for index in range(28)], dim=1
        )
        self.target_pose = torch.stack(
            (values, values.square(), -values, torch.zeros_like(values)), dim=1
        )
        self.action = torch.tanh(
            torch.stack(
                (
                    self.state[:, 0],
                    self.state[:, 1],
                    self.state[:, 2],
                    self.state[:, 3],
                    self.state[:, 4],
                ),
                dim=1,
            )
        )
        self.episode_index = torch.arange(count, dtype=torch.int64) // 4
        self.front = torch.zeros(count, 3, 128, 128)
        self.wrist = torch.zeros(count, 3, 128, 128)

    def __len__(self) -> int:
        """Return the number of synthetic transitions."""
        return len(self.state)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one structured sample in the BC trainer's batch format."""
        sample = {
            "state": self.state[index],
            "object_poses": self.object_poses[index],
            "target_pose": self.target_pose[index],
            "action": self.action[index],
            "episode_index": self.episode_index[index],
        }
        if self.policy_type == "image":
            sample["front"] = self.front[index]
            sample["wrist"] = self.wrist[index]
        return sample

    def normalization_features(self) -> torch.Tensor:
        """Provide state-only or concatenated state inputs for normalization."""
        if self.policy_type == "image":
            return self.state
        return torch.cat((self.state, self.object_poses, self.target_pose), dim=1)


def test_default_bc_training_reruns_use_numbered_directories(tmp_path) -> None:
    """Keep the existing run and skip numbered output directories already used."""
    from scripts.train_bc import _select_output_dir

    output_dir = tmp_path / "state"
    output_dir.mkdir()
    (tmp_path / "state-run-02").mkdir()

    assert (
        _select_output_dir(output_dir, automatic_versioning=True)
        == tmp_path / "state-run-03"
    )
    with pytest.raises(FileExistsError, match="state"):
        _select_output_dir(output_dir, automatic_versioning=False)


@pytest.mark.parametrize("policy_type", ["state", "image"])
def test_bc_models_train_and_reduce_validation_l1(policy_type: str) -> None:
    """Train each BC architecture and verify its held-out L1 improves."""
    dataset = SyntheticBCDataset(policy_type)
    config = bc.BCTrainingConfig(
        epochs=8,
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0,
        device="cpu",
        seed=7,
        patience=8,
        pretrained=False,
    )

    model, result = bc.train_behavior_cloning(policy_type, dataset, dataset, config)

    assert result.best_validation_l1 < result.initial_validation_l1
    assert result.history
    assert set(result.validation_action_mae) == set(bc.ACTION_COMPONENT_NAMES)
    assert len(result.validation_episode_l1) == 4
    assert next(model.parameters()).device.type == "cpu"


def test_shared_bc_run_preserves_results_and_checkpoint_format(tmp_path) -> None:
    """Adapt BC logging while preserving its training result and checkpoint."""
    pytest.importorskip("tensorboard")
    from scripts.train_bc import _train_bc_run

    dataset = SyntheticBCDataset("state")
    config = bc.BCTrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        device="cpu",
        seed=17,
        patience=2,
        pretrained=False,
    )
    expected_model, expected_result = bc.train_behavior_cloning(
        "state",
        dataset,
        dataset,
        config,
    )

    checkpoint_path, run_record = _train_bc_run(
        output_dir=tmp_path / "run",
        policy_type="state",
        train_data=dataset,
        validation_data=dataset,
        config=config,
        dataset_id="blindspot/synthetic",
        dataset_splits={
            "train": "blindspot/synthetic-train",
            "validation": "blindspot/synthetic-validation",
        },
        dataset_version="test-version",
        seeds={"train": [17], "validation": [100_000]},
        task="T2",
        level="L1",
        rollout_validation_seeds=None,
    )

    policy = bc.load_bc_policy(checkpoint_path, device="cpu")
    assert run_record.status == "completed"
    assert run_record.dataset_id == "blindspot/synthetic"
    assert run_record.result == expected_result
    assert checkpoint_path.name == "checkpoint.pt"
    for name, expected in expected_model.state_dict().items():
        assert torch.equal(policy.model.state_dict()[name], expected)
    saved_record = json.loads(
        (tmp_path / "run" / "run.json").read_text(encoding="utf-8")
    )
    assert saved_record["policy_type"] == "state"
    assert saved_record["status"] == "completed"
    assert saved_record["dataset_version"] == "test-version"
    assert saved_record["settings"]["precision"] == "fp32"
    assert list((tmp_path / "run" / "tensorboard").glob("events.out.tfevents.*"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_state_bc_validation_metrics_support_mps() -> None:
    """Accumulate per-action validation errors on CPU after MPS inference."""
    dataset = SyntheticBCDataset("state")
    config = bc.BCTrainingConfig(
        epochs=1,
        batch_size=8,
        device="mps",
        seed=7,
        pretrained=False,
    )

    _, result = bc.train_behavior_cloning("state", dataset, dataset, config)

    assert np.isfinite(result.best_validation_l1)
    assert set(result.validation_action_mae) == set(bc.ACTION_COMPONENT_NAMES)


def test_state_policy_obeys_policy_protocol_and_action_limits() -> None:
    """Adapt a state model to the shared mapping observation and 5-D action."""
    model = bc.StateBC()
    poses = {
        name: {
            "position": np.zeros(3, dtype=np.float32),
            "quaternion_wxyz": np.array([1, 0, 0, 0], dtype=np.float32),
        }
        for name in bc.OBJECT_POSE_NAMES
    }
    policy = bc.StateBCPolicy(model, device="cpu")
    policy.reset()

    action = policy.act(
        {
            "observation.state": np.zeros(12, dtype=np.float32),
            "object_poses": poses,
            "target_pose": np.zeros(4, dtype=np.float32),
        }
    )

    assert action.shape == (5,)
    assert action.dtype == np.float32
    assert np.all(action >= bc._ACTION_LOW)
    assert np.all(action <= bc._ACTION_HIGH)


def test_image_model_accepts_channel_last_uint8_frames() -> None:
    """Apply the same uint8 HWC conversion used by MuJoCo evaluation images."""
    model = bc.ImageBC(pretrained=False).eval()
    front = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
    wrist = torch.full((1, 128, 128, 3), 127, dtype=torch.uint8)

    with torch.inference_mode():
        output = model(front, wrist, torch.zeros(1, 12))

    assert output.shape == (1, 5)
    assert torch.isfinite(output).all()


def test_image_bc_freezes_resnets_but_trains_fusion_head() -> None:
    """Keep frozen camera encoders in eval mode while training the action head."""
    model = bc.ImageBC(pretrained=False).train()

    assert not model.front_encoder.training
    assert not model.wrist_encoder.training
    assert all(
        not parameter.requires_grad for parameter in model.front_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.wrist_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_image_policy_adapts_environment_images_and_state() -> None:
    """Accept uint8 HWC camera observations through the shared Policy interface."""
    policy = bc.ImageBCPolicy(bc.ImageBC(pretrained=False), device="cpu")
    action = policy.act(
        {
            "observation.state": np.zeros(12, dtype=np.float32),
            "observation.images.front": np.zeros((128, 128, 3), dtype=np.uint8),
            "observation.images.wrist": np.full((128, 128, 3), 127, dtype=np.uint8),
        }
    )

    assert action.shape == (5,)
    assert np.all(action >= bc._ACTION_LOW)
    assert np.all(action <= bc._ACTION_HIGH)


def test_bc_checkpoint_round_trips_without_pretrained_download(tmp_path) -> None:
    """Save and safely reload a state checkpoint for policy evaluation."""
    checkpoint_path = bc.save_bc_checkpoint(
        bc.StateBC(),
        "state",
        tmp_path / "state.pt",
    )

    policy = bc.load_bc_policy(checkpoint_path, device="cpu")

    assert isinstance(policy, bc.StateBCPolicy)

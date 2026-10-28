import json
import random
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

torch = pytest.importorskip("torch")
training = import_module("common.training")
nn = torch.nn
TrainingRun = training.TrainingRun
TrainingSettings = training.TrainingSettings
autocast_configuration = training.autocast_configuration
load_training_checkpoint = training.load_training_checkpoint
optimizer_step = training.optimizer_step
save_training_checkpoint = training.save_training_checkpoint
seed_everything = training.seed_everything


class TinyTrainingConfig(BaseModel):
    """Provide a small validated config for run-record tests."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = 2
    api_key: str = "must-not-be-recorded"


def test_cpu_auto_precision_records_fp32_fallback() -> None:
    """Use fp32 automatically on CPU without claiming mixed precision."""
    precision, dtype, fallback = autocast_configuration(torch.device("cpu"), "auto")
    assert precision == "fp32"
    assert dtype is None
    assert fallback is None


def test_optimizer_step_reports_non_finite_loss_with_step_number() -> None:
    """Stop on a NaN loss instead of updating model parameters."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).sum() * torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="loss.*step 7"):
        optimizer_step(loss, model, optimizer, step=7, max_grad_norm=1.0)


def test_checkpoint_resume_restores_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    """Resume one update identically after restoring model, optimizer, and RNG."""
    seed_everything(22)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    inputs = torch.randn(4, 2)
    loss = model(inputs).square().mean()
    optimizer_step(
        loss,
        model,
        optimizer,
        step=1,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    checkpoint_path = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=1,
        scheduler=scheduler,
        extra_state={"next_batch": 1},
    )

    expected_inputs = torch.randn(4, 2)
    expected_numpy = float(np.random.random())
    expected_python = random.random()
    expected_loss = model(expected_inputs).square().mean()
    optimizer_step(
        expected_loss,
        model,
        optimizer,
        step=2,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    seed_everything(999)
    resumed_model = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer,
        step_size=1,
        gamma=0.5,
    )
    resumed = load_training_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    assert resumed.step == 1
    assert resumed.extra_state == {"next_batch": 1}
    actual_inputs = torch.randn(4, 2)
    assert torch.equal(actual_inputs, expected_inputs)
    assert float(np.random.random()) == expected_numpy
    assert random.random() == expected_python
    actual_loss = resumed_model(actual_inputs).square().mean()
    optimizer_step(
        actual_loss,
        resumed_model,
        resumed_optimizer,
        step=2,
        max_grad_norm=1.0,
        scheduler=resumed_scheduler,
    )
    for name, expected in expected_parameters.items():
        assert torch.equal(resumed_model.state_dict()[name], expected)
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()


def test_training_run_writes_reproducibility_record_and_scalar_events(
    tmp_path: Path,
) -> None:
    """Write required metadata and scalar-only TensorBoard training metrics."""
    pytest.importorskip("tensorboard")
    output_dir = tmp_path / "run"
    settings = TrainingSettings(seed=10, device="cpu", precision="auto")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with TrainingRun(
        output_dir,
        config=TinyTrainingConfig(),
        dataset_id="tiny/train",
        dataset_version="v1",
        seeds={"train": [1, 2], "validation": [100_000]},
        settings=settings,
    ) as run:
        with run.autocast():
            loss = model(torch.ones(2, 2)).square().mean()
        run.optimize(loss, model, optimizer, step=1)
        run.log_metrics(1, {"validation/loss": 0.25})

    record = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["dataset_id"] == "tiny/train"
    assert record["dataset_version"] == "v1"
    assert isinstance(record["git_dirty"], bool)
    assert record["precision"] == "fp32"
    assert record["config"]["api_key"] == "[REDACTED]"
    event_files = list((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert event_files


def test_training_run_resumes_in_place_and_preserves_method_metadata(
    tmp_path: Path,
) -> None:
    """Append TensorBoard steps and preserve method-specific run metadata."""
    pytest.importorskip("tensorboard")
    output_dir = tmp_path / "run"
    settings = TrainingSettings(seed=10, device="cpu", precision="auto")
    seeds = {"train": [1, 2], "validation": [100_000]}
    with TrainingRun(
        output_dir,
        config=TinyTrainingConfig(),
        dataset_id="tiny/train",
        dataset_version="v1",
        seeds=seeds,
        settings=settings,
    ) as run:
        run.log_metrics(3, {"train/loss": 0.5})

    record_path = output_dir / "run.json"
    previous_record = json.loads(record_path.read_text(encoding="utf-8"))
    previous_started_at = previous_record["started_at"]
    previous_record["vla_metadata"] = {"warm_start_count": 0}
    record_path.write_text(json.dumps(previous_record), encoding="utf-8")
    with TrainingRun(
        output_dir,
        config=TinyTrainingConfig(epochs=3),
        dataset_id="tiny/train",
        dataset_version="v1",
        seeds=seeds,
        settings=settings,
        resume_existing=True,
        step_offset=3,
    ) as run:
        run.log_metrics(4, {"train/loss": 0.25})

    resumed_record = json.loads(record_path.read_text(encoding="utf-8"))
    assert resumed_record["status"] == "completed"
    assert resumed_record["started_at"] == previous_started_at
    assert resumed_record["vla_metadata"] == {"warm_start_count": 0}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(output_dir / "tensorboard"))
    events.Reload()
    assert [value.step for value in events.Scalars("train/loss")] == [3, 4]

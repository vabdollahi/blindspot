"""Verify single-process, DDP, and exact-resume training behavior."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional

from common.training import (
    TrainingRun,
    TrainingSettings,
    destroy_distributed,
    initialize_distributed,
    load_training_checkpoint,
    optimizer_step,
    save_training_checkpoint,
    seed_everything,
    wrap_distributed_model,
)

_STEP_COUNT = 6
_SEED = 2304


class SmokeConfig(BaseModel):
    """Record the toy training workload used by the infrastructure check."""

    model_config = ConfigDict(extra="forbid")

    steps: int = _STEP_COUNT
    batch_size: int = 8
    input_size: int = 3
    learning_rate: float = 0.01


def _training_data() -> tuple[Tensor, Tensor]:
    """Build one fixed batch shared by single- and multi-process runs."""
    generator = torch.Generator(device="cpu").manual_seed(_SEED + 1)
    inputs = torch.randn(8, 3, generator=generator)
    targets = inputs @ torch.tensor([[0.5], [-0.25], [0.75]]) + 0.1
    return inputs, targets


def _run_tiny_training(output_dir: Path) -> None:
    """Train the tiny deterministic model on this torchrun worker."""
    context = initialize_distributed(device="cpu", backend="gloo")
    seed_everything(_SEED)
    inputs, targets = _training_data()
    if context.is_distributed:
        shard_size = len(inputs) // context.world_size
        start = context.rank * shard_size
        end = start + shard_size
        local_inputs = inputs[start:end]
        local_targets = targets[start:end]
    else:
        local_inputs = inputs
        local_targets = targets

    raw_model = nn.Linear(3, 1)
    model = wrap_distributed_model(raw_model, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    history: list[float] = []
    config = SmokeConfig()
    settings = TrainingSettings(seed=_SEED, device="cpu", precision="auto")
    try:
        with TrainingRun(
            output_dir,
            config=config,
            dataset_id="training-smoke",
            dataset_version="1",
            seeds={"train": [_SEED]},
            settings=settings,
            distributed_context=context,
        ) as run:
            for step in range(1, _STEP_COUNT + 1):
                noise = torch.rand(()) * 0.001
                with run.autocast():
                    prediction = model(local_inputs)
                    loss = functional.mse_loss(prediction, local_targets + noise)
                reported_loss = loss.detach().clone()
                if context.is_distributed:
                    torch.distributed.all_reduce(reported_loss)
                    reported_loss /= context.world_size
                run.optimize(loss, model, optimizer, step=step)
                history.append(float(reported_loss))

            run.save_checkpoint(
                output_dir / "final.pt",
                model,
                optimizer,
                step=_STEP_COUNT,
            )
            if context.is_main_process:
                state = {
                    name: tensor.detach().cpu().tolist()
                    for name, tensor in raw_model.state_dict().items()
                }
                (output_dir / "history.json").write_text(
                    json.dumps({"losses": history, "state": state}, indent=2) + "\n",
                    encoding="utf-8",
                )
    finally:
        destroy_distributed()


def _resume_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
) -> float:
    """Consume the same random inputs and perform one deterministic update."""
    inputs = torch.rand(4, 3)
    targets = torch.randn(4, 1)
    target_offset = float(np.random.random() + random.random()) * 0.001
    loss = functional.mse_loss(model(inputs), targets + target_offset)
    loss_value = float(loss.detach())
    optimizer_step(
        loss,
        model,
        optimizer,
        step=step,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    return loss_value


def _resume_run(
    output_dir: Path, *, resume_from: Path | None
) -> tuple[list[float], dict[str, Tensor]]:
    """Run a continuous or checkpoint-resumed toy training segment."""
    seed_everything(_SEED if resume_from is None else _SEED + 999)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    start_step = 0
    if resume_from is not None:
        checkpoint = load_training_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler=scheduler,
        )
        start_step = checkpoint.step
    history: list[float] = []
    for step in range(start_step + 1, _STEP_COUNT + 1):
        history.append(_resume_step(model, optimizer, scheduler, step))
        if step == _STEP_COUNT // 2 and resume_from is None:
            save_training_checkpoint(
                output_dir / "midpoint.pt",
                model,
                optimizer,
                step=step,
                scheduler=scheduler,
                extra_state={"next_step": step + 1},
            )
    return history, {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }


def _verify_exact_resume(root: Path) -> None:
    """Require resumed updates and parameters to match an uninterrupted run."""
    continuous_history, continuous_state = _resume_run(
        root / "continuous",
        resume_from=None,
    )
    midpoint = root / "continuous" / "midpoint.pt"
    resumed_history, resumed_state = _resume_run(
        root / "split-resumed",
        resume_from=midpoint,
    )
    if len(continuous_history) != _STEP_COUNT:
        raise RuntimeError("Continuous training produced an incomplete loss curve")
    expected_history = continuous_history[_STEP_COUNT // 2 :]
    if resumed_history != expected_history:
        raise AssertionError(
            f"Checkpoint resume changed the loss curve: "
            f"{resumed_history!r} != {expected_history!r}"
        )
    for name, expected in continuous_state.items():
        if not torch.equal(resumed_state[name], expected):
            raise AssertionError(f"Checkpoint resume changed parameter {name!r}")
    if not midpoint.is_file():
        raise RuntimeError("The midpoint checkpoint was not saved")


def _read_history(path: Path) -> dict[str, Any]:
    """Read one verifier output file and validate its JSON root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _run_verification() -> None:
    """Compare a single-process run with torchrun DDP and exact resume."""
    with tempfile.TemporaryDirectory(prefix="blindspot-training-check-") as temporary:
        root = Path(temporary)
        single_dir = root / "single"
        ddp_dir = root / "ddp"
        _run_tiny_training(single_dir)
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            master_port = int(port_socket.getsockname()[1])
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--rdzv_backend=static",
                "--nnodes=1",
                "--nproc_per_node=2",
                "--master_addr=127.0.0.1",
                f"--master_port={master_port}",
                str(Path(__file__).resolve()),
                "--worker",
                "--output-dir",
                str(ddp_dir),
            ],
            check=True,
            timeout=120,
            env={**os.environ, "GLOO_SOCKET_IFNAME": "lo0"},
        )
        single = _read_history(single_dir / "history.json")
        ddp = _read_history(ddp_dir / "history.json")
        np.testing.assert_allclose(
            single["losses"],
            ddp["losses"],
            rtol=1e-6,
            atol=1e-7,
        )
        if single["state"].keys() != ddp["state"].keys():
            raise AssertionError("Single-process and DDP parameter keys differ")
        for name in single["state"]:
            np.testing.assert_allclose(
                single["state"][name],
                ddp["state"][name],
                rtol=1e-6,
                atol=1e-7,
            )
        _verify_exact_resume(root / "resume")

    print(
        "Training checks passed: 1-process and 2-process Gloo loss curves match "
        "within tolerance; checkpoint resume is bit-for-bit."
    )


def main() -> None:
    """Run as an orchestrator or as a torchrun DDP worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.output_dir is None:
            parser.error("--output-dir is required with --worker")
        _run_tiny_training(arguments.output_dir)
        return
    if arguments.output_dir is not None:
        parser.error("--output-dir is only used with --worker")
    _run_verification()


if __name__ == "__main__":
    main()

"""Shared reproducible PyTorch training utilities for model phases."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, distributed, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

Precision = Literal["auto", "fp32", "fp16", "bf16"]
ResolvedPrecision = Literal["fp32", "fp16", "bf16"]
_CHECKPOINT_FORMAT_VERSION = 1
_SENSITIVE_CONFIG_KEY = re.compile(
    r"(?:secret|password|credential|authorization|api[_-]?key|access[_-]?key|(?:^|_)token$)",
    re.IGNORECASE,
)


class TrainingSettings(BaseModel):
    """Validate runtime settings shared by model-specific training configs."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=0, ge=0)
    device: str = "auto"
    precision: Precision = "auto"
    max_grad_norm: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)


class TrainingRunRecord(BaseModel):
    """Persist reproducibility metadata and the run's final status."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    settings: TrainingSettings
    git_commit: str
    git_dirty: bool
    seeds: dict[str, list[int]]
    dataset_id: str
    dataset_version: str
    hardware: dict[str, str | bool | int]
    requested_precision: Precision
    precision: ResolvedPrecision
    precision_fallback: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    started_at: datetime
    finished_at: datetime | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class DistributedContext:
    """Describe the current process in a single- or multi-process run."""

    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        """Return whether this is the process responsible for shared outputs."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Return whether multiple processes participate in this run."""
        return self.world_size > 1


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Return the saved training step and optional caller-owned state."""

    step: int
    extra_state: dict[str, Any]


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve an explicit device or prefer MPS, CUDA, then CPU."""
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    selected = torch.device(device)
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and all available PyTorch random generators."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def initialize_distributed(
    *,
    device: str = "auto",
    backend: str | None = None,
) -> DistributedContext:
    """Initialize a torchrun process group, or describe a single process."""
    if distributed.is_initialized():
        rank = distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = str(distributed.get_backend())
        if device == "auto" and backend == "nccl":
            selected = torch.device("cuda", local_rank)
        elif device == "auto" and backend == "gloo":
            selected = torch.device("cpu")
        else:
            selected = resolve_device(device)
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=distributed.get_world_size(),
            backend=backend,
            device=selected,
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            f"Invalid distributed environment: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )

    if device == "auto" and world_size > 1:
        selected = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        selected = resolve_device(device)

    if world_size == 1:
        return DistributedContext(0, 0, 1, None, selected)
    if selected.type == "mps":
        raise RuntimeError(
            "Multi-process MPS training is unsupported; use CPU/Gloo or CUDA/NCCL"
        )

    selected_backend = backend or ("nccl" if selected.type == "cuda" else "gloo")
    if selected_backend == "nccl" and selected.type != "cuda":
        raise ValueError("The NCCL backend requires a CUDA device")
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
    distributed.init_process_group(
        backend=selected_backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return DistributedContext(rank, local_rank, world_size, selected_backend, selected)


def wrap_distributed_model(
    model: nn.Module,
    context: DistributedContext,
) -> nn.Module:
    """Move a model to its process device and wrap it in DDP when needed."""
    model = model.to(context.device)
    if not context.is_distributed:
        return model
    if not distributed.is_initialized():
        raise RuntimeError("initialize_distributed must run before wrapping a model")
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)


def make_distributed_sampler(
    dataset: Dataset[Any],
    context: DistributedContext,
    *,
    seed: int,
    shuffle: bool = True,
) -> Sampler[int] | None:
    """Create a rank-partitioned sampler, or return None for one process."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not context.is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
    )


def autocast_configuration(
    device: torch.device,
    requested: Precision,
) -> tuple[ResolvedPrecision, torch.dtype | None, str | None]:
    """Choose supported autocast precision and explain any fp32 fallback."""
    if requested == "auto":
        if device.type == "cuda":
            requested_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif device.type == "mps":
            requested_dtype = torch.float16
        else:
            return "fp32", None, None
    else:
        requested_dtype = {
            "fp32": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[requested]
        if requested_dtype is None:
            return "fp32", None, None

    requested_name: ResolvedPrecision = (
        "fp16" if requested_dtype == torch.float16 else "bf16"
    )
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        message = f"autocast is unavailable for device type {device.type!r}"
        if requested == "auto":
            return "fp32", None, message
        raise RuntimeError(message)
    if (
        requested_dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        message = "CUDA device does not support bfloat16 autocast"
        if requested == "auto":
            return "fp32", None, message
        raise RuntimeError(message)

    try:
        with torch.autocast(device_type=device.type, dtype=requested_dtype):
            probe = torch.ones((2, 2), device=device) @ torch.ones(
                (2, 2), device=device
            )
        if not torch.isfinite(probe).all().item():
            raise RuntimeError("autocast capability probe produced non-finite values")
    except (RuntimeError, TypeError, NotImplementedError) as error:
        if requested == "auto":
            return "fp32", None, f"{type(error).__name__}: {error}"
        raise RuntimeError(
            f"{requested_name} autocast is unavailable on {device}: {error}"
        ) from error
    return requested_name, requested_dtype, None


class TrainingRun:
    """Record and track one training run with safe lifecycle and metrics."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        config: BaseModel,
        dataset_id: str,
        dataset_version: str,
        seeds: Mapping[str, Sequence[int]],
        settings: TrainingSettings,
        distributed_context: DistributedContext | None = None,
        resume_existing: bool = False,
        step_offset: int = 0,
    ) -> None:
        """Create or resume a run directory, record, and TensorBoard writer."""
        if isinstance(step_offset, bool) or not isinstance(step_offset, int):
            raise TypeError("step_offset must be an integer")
        if step_offset < 0:
            raise ValueError("step_offset must be non-negative")
        if not resume_existing and step_offset != 0:
            raise ValueError("step_offset requires resume_existing=True")
        if not dataset_id.strip() or not dataset_version.strip():
            raise ValueError("dataset_id and dataset_version must be non-empty")
        if not seeds:
            raise ValueError("At least one named seed sequence is required")
        normalized_seeds: dict[str, list[int]] = {}
        for name, values in seeds.items():
            if not name or not values:
                raise ValueError(
                    "Each seed sequence needs a name and at least one seed"
                )
            if any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in values
            ):
                raise ValueError(
                    f"Seed sequence {name!r} must contain non-negative integers"
                )
            normalized_seeds[name] = list(values)

        git_commit = _git_commit()
        git_dirty = _git_dirty()

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TrainingRun requires TensorBoard; install the training extra with "
                "`uv sync --extra training`."
            ) from error

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.settings = settings
        self.distributed = distributed_context or initialize_distributed(
            device=settings.device
        )
        self.device = (
            self.distributed.device
            if settings.device == "auto"
            else resolve_device(settings.device)
        )
        self.precision, self.autocast_dtype, fallback = autocast_configuration(
            self.device, settings.precision
        )
        self._started_monotonic = time.monotonic()
        self._closed = False
        self._writer: Any | None = None
        self._step_offset = step_offset
        self._record_extra_fields: dict[str, Any] = {}
        resumed_started_at: datetime | None = None

        if resume_existing:
            record_path = self.output_dir / "run.json"
            tensorboard_dir = self.output_dir / "tensorboard"
            if not self.output_dir.is_dir():
                raise FileNotFoundError(
                    f"Cannot resume missing training directory: {self.output_dir}"
                )
            if not record_path.is_file() or not tensorboard_dir.is_dir():
                raise FileNotFoundError(
                    "Resuming a run requires its run.json and tensorboard directory"
                )
            previous_payload = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(previous_payload, dict):
                raise ValueError("Existing run.json must contain a JSON object")
            record_fields = TrainingRunRecord.model_fields
            previous_record = TrainingRunRecord.model_validate(
                {
                    name: previous_payload[name]
                    for name in record_fields
                    if name in previous_payload
                }
            )
            if previous_record.status == "running":
                raise RuntimeError(
                    f"Training run is already marked running: {self.output_dir}"
                )
            resumed_started_at = previous_record.started_at
            self._record_extra_fields = {
                name: value
                for name, value in previous_payload.items()
                if name not in record_fields
            }

        if self.distributed.is_main_process:
            if not resume_existing:
                self.output_dir.mkdir(parents=True, exist_ok=False)
            self._writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        self._barrier()
        self.record = TrainingRunRecord(
            config=_safe_config(config.model_dump(mode="json")),
            settings=settings,
            git_commit=git_commit,
            git_dirty=git_dirty,
            seeds=normalized_seeds,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            hardware=_hardware_info(self.device),
            requested_precision=settings.precision,
            precision=self.precision,
            precision_fallback=fallback,
            started_at=resumed_started_at or datetime.now(UTC),
        )
        if self.distributed.is_main_process:
            _write_json_atomic(
                self.output_dir / "run.json",
                self.record,
                extra_fields=self._record_extra_fields,
            )

    def __enter__(self) -> Self:
        """Return this run for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Persist completion or failure and close TensorBoard cleanly."""
        del exc_value, traceback
        self.close(error_type=exc_type.__name__ if exc_type is not None else None)
        return False

    def autocast(self) -> Any:
        """Return the selected autocast context, or a no-op fp32 context."""
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
        )

    def gradient_scaler(self) -> torch.amp.GradScaler | None:
        """Return a CUDA fp16 scaler when needed, otherwise None."""
        if self.device.type == "cuda" and self.precision == "fp16":
            return torch.amp.GradScaler("cuda")
        return None

    def optimize(
        self,
        loss: Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        scheduler: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> float:
        """Backpropagate, guard, clip, update, and log one training step.

        Every process in a distributed run must call this method at every step.
        """
        loss_for_logging = loss.detach().clone()
        if self.distributed.is_distributed:
            distributed.all_reduce(loss_for_logging)
            loss_for_logging /= self.distributed.world_size
        optimizer.zero_grad(set_to_none=True)
        loss_value = _finite_scalar(loss_for_logging, "loss", step)
        gradient_norm = optimizer_step(
            loss,
            model,
            optimizer,
            step=step,
            max_grad_norm=self.settings.max_grad_norm,
            scheduler=scheduler,
            scaler=scaler,
        )
        metrics: dict[str, float] = {
            "train/loss": loss_value,
            "train/gradient_norm": gradient_norm,
        }
        self.log_metrics(step, metrics, optimizer=optimizer)
        return gradient_norm

    def log_metrics(
        self,
        step: int,
        metrics: Mapping[str, float | int | Tensor],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Write finite scalar metrics, learning rates, and throughput to TensorBoard."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if step < self._step_offset:
            raise ValueError(
                f"step {step} is less than the resume offset {self._step_offset}"
            )
        if not self.distributed.is_main_process:
            return
        if self._closed or self._writer is None:
            raise RuntimeError("Cannot log to a closed training run")
        values: dict[str, float] = {}
        for name, value in metrics.items():
            if not name.strip():
                raise ValueError("Metric names must be non-empty")
            values[name] = _finite_scalar(value, name, step)
        values["train/steps_per_second"] = (step - self._step_offset) / max(
            time.monotonic() - self._started_monotonic, 1e-9
        )
        if optimizer is not None:
            for index, group in enumerate(optimizer.param_groups):
                values[f"train/learning_rate/group_{index}"] = float(group["lr"])
        for name, value in values.items():
            self._writer.add_scalar(name, value, step)
        self._writer.flush()

    def save_checkpoint(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        scheduler: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Save a resumable checkpoint, writing one shared file from rank zero."""
        return save_training_checkpoint(
            path,
            model,
            optimizer,
            step=step,
            scheduler=scheduler,
            scaler=scaler,
            extra_state=extra_state,
        )

    def close(self, *, error_type: str | None = None) -> None:
        """Finalize run metadata and release the TensorBoard writer."""
        if self._closed:
            return
        if self.distributed.is_main_process:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()
                self._writer = None
            self.record = self.record.model_copy(
                update={
                    "status": "failed" if error_type else "completed",
                    "finished_at": datetime.now(UTC),
                    "error_type": error_type,
                }
            )
            _write_json_atomic(
                self.output_dir / "run.json",
                self.record,
                extra_fields=self._record_extra_fields,
            )
        self._barrier()
        self._closed = True

    def _barrier(self) -> None:
        """Synchronize distributed processes around shared run files."""
        if self.distributed.is_distributed:
            distributed.barrier()


def optimizer_step(
    loss: Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    max_grad_norm: float,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one guarded optimizer step and return the pre-clip gradient norm."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive")
    optimizer.zero_grad(set_to_none=True)
    _finite_scalar(loss, "loss", step)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

    parameters = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    if not parameters:
        raise RuntimeError(f"No gradients were produced at step {step}")
    for parameter in parameters:
        if not torch.isfinite(parameter.grad).all().item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradient detected at step {step}")
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_grad_norm,
        error_if_nonfinite=False,
    )
    gradient_norm = _finite_scalar(norm, "gradient_norm", step)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if scheduler is not None:
        scheduler.step()
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(
                f"Non-finite model parameter detected at step {step}"
            )
    return gradient_norm


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path | None:
    """Atomically save model, optimizer, scheduler, scaler, and RNG state."""
    if isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    world_size = distributed.get_world_size() if distributed.is_initialized() else 1
    local_rng_state = _capture_rng_state()
    if world_size > 1:
        rng_states: list[dict[str, Any] | None] = [None] * world_size
        distributed.all_gather_object(rng_states, local_rng_state)
    else:
        rng_states = [local_rng_state]
    if rank != 0:
        return None

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "world_size": world_size,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_states": rng_states,
        "extra_state": dict(extra_state or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> TrainingCheckpoint:
    """Restore a checkpoint and its per-rank RNG state for exact continuation."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Training checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Training checkpoint root must be a mapping")
    if payload.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported training checkpoint format")
    if not isinstance(payload.get("step"), int) or payload["step"] < 0:
        raise ValueError("Training checkpoint has an invalid step")
    current_world_size = (
        distributed.get_world_size() if distributed.is_initialized() else 1
    )
    if payload.get("world_size") != current_world_size:
        raise ValueError(
            "Cannot resume with a different world size: "
            f"checkpoint={payload.get('world_size')}, current={current_world_size}"
        )
    _unwrap_model(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    _restore_optional_state("scheduler", scheduler, payload.get("scheduler"))
    _restore_optional_state("scaler", scaler, payload.get("scaler"))
    if restore_rng:
        rank = distributed.get_rank() if distributed.is_initialized() else 0
        rng_states = payload.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != current_world_size:
            raise ValueError("Training checkpoint has invalid per-rank RNG states")
        rng_state = rng_states[rank]
        if not isinstance(rng_state, dict):
            raise ValueError(
                f"Training checkpoint is missing RNG state for rank {rank}"
            )
        _restore_rng_state(rng_state)
    extra_state = payload.get("extra_state", {})
    if not isinstance(extra_state, dict):
        raise TypeError("Training checkpoint extra_state must be a mapping")
    return TrainingCheckpoint(step=payload["step"], extra_state=extra_state)


def destroy_distributed() -> None:
    """Destroy the active process group after distributed training completes."""
    if distributed.is_initialized():
        distributed.destroy_process_group()


def _safe_config(value: Any, key: str = "") -> Any:
    """Convert a JSON-mode config while redacting secret-shaped field names."""
    if _SENSITIVE_CONFIG_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(name): _safe_config(item, str(name)) for name, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_config(item) for item in value]
    return value


def _git_commit() -> str:
    """Return the checked-out commit hash or raise when the run is outside Git."""
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("Git returned an empty commit hash")
    return commit


def _git_dirty() -> bool:
    """Return whether the checkout has tracked or untracked working-tree changes."""
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def _hardware_info(device: torch.device) -> dict[str, str | bool | int]:
    """Collect the execution environment required to reproduce a training run."""
    hardware: dict[str, str | bool | int] = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 0,
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        hardware["accelerator"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        hardware["accelerator"] = "Apple Metal Performance Shaders"
    return hardware


def _write_json_atomic(
    path: Path,
    record: TrainingRunRecord,
    *,
    extra_fields: Mapping[str, Any] | None = None,
) -> None:
    """Persist a validated run record without exposing a partial JSON file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    if extra_fields:
        payload = record.model_dump(mode="json")
        payload.update(extra_fields)
        contents = json.dumps(payload, indent=2) + "\n"
    else:
        contents = record.model_dump_json(indent=2) + "\n"
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _finite_scalar(value: float | Tensor, name: str, step: int) -> float:
    """Convert a scalar metric to a finite Python float or report its step."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"Metric {name!r} at step {step} must be scalar")
        result = float(value.detach().item())
    else:
        result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite {name} detected at step {step}")
    return result


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return a DDP model's underlying module or the original model."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _restore_optional_state(
    name: str,
    component: Any | None,
    state: Any,
) -> None:
    """Restore an optional checkpoint component, requiring matching presence."""
    if component is None and state is None:
        return
    if component is None or state is None:
        raise ValueError(
            f"Checkpoint and current run disagree about the {name} component"
        )
    component.load_state_dict(state)


def _capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU, CUDA, and available MPS random states."""
    python_version, python_state, python_gaussian = random.getstate()
    numpy_name, numpy_keys, numpy_position, numpy_has_gaussian, numpy_gaussian = (
        np.random.get_state()
    )
    state: dict[str, Any] = {
        "python": {
            "version": python_version,
            "state": list(python_state),
            "gaussian": python_gaussian,
        },
        "numpy": {
            "name": numpy_name,
            "keys": numpy_keys.tolist(),
            "position": numpy_position,
            "has_gaussian": numpy_has_gaussian,
            "gaussian": numpy_gaussian,
        },
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [
            rng_state.tolist() for rng_state in torch.cuda.get_rng_state_all()
        ]
    mps_get_state = getattr(torch.mps, "get_rng_state", None)
    if torch.backends.mps.is_available() and callable(mps_get_state):
        state["mps"] = mps_get_state().cpu().tolist()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore all serialized global random generators."""
    python_state = state["python"]
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"]),
            python_state["gaussian"],
        )
    )
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["name"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["gaussian"]),
        )
    )
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if "cuda" in state:
        cuda_state = state["cuda"]
        if not isinstance(cuda_state, list) or not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(
            [torch.tensor(value, dtype=torch.uint8) for value in cuda_state]
        )
    if "mps" in state:
        mps_set_state = getattr(torch.mps, "set_rng_state", None)
        if not torch.backends.mps.is_available() or not callable(mps_set_state):
            raise RuntimeError(
                "Checkpoint contains MPS RNG state but MPS is unavailable"
            )
        mps_set_state(torch.tensor(state["mps"], dtype=torch.uint8))

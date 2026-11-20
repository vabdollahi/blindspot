"""Train the Phase 4 JEPA, or its supervised ceiling, on cached T3-L1 frames."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Sampler

from common.training import TrainingRun, load_training_checkpoint
from encoders.frames import (
    EpisodeRef,
    FrameCache,
    TimestepDataset,
    WindowDataset,
    window_starts,
)
from encoders.jepa import (
    JEPA,
    JEPAConfig,
    SupervisedEncoder,
    count_parameters,
    effective_rank,
    random_directions,
)
from encoders.probes import (
    FLAG_TARGETS,
    ProbeLabels,
    fit_ridge_targets,
    labels_for_episodes,
    per_sample_error,
)
from encoders.vit import global_readout

_PROBE_SCORE_TARGETS = ("cube_xyz", "tcp_xyz")
SUPERVISED_TARGETS = ("cube_xyz", "cube_yaw", "bowl_xy", "tcp_xyz", "gripper")


class JEPATrainingConfig(BaseModel):
    """Validate one JEPA or supervised-ceiling training run."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["jepa", "supervised"] = "jepa"
    model: JEPAConfig = Field(default_factory=JEPAConfig)
    train_datasets: list[str] = Field(
        default_factory=lambda: ["interaction-l1", "t3-l1"]
    )
    validation_dataset: str | None = "t3-l1-probe-val"
    tasks: list[str] = Field(default_factory=lambda: ["T3"])
    steps: int = Field(default=6_000, ge=1)
    batch_size: int = Field(default=64, ge=2)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    warmup_steps: int = Field(default=600, ge=0)
    weight_decay: float = Field(default=0.05, ge=0.0)
    holdout_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    eval_every: int = Field(default=1_000, ge=1)
    checkpoint_every: int = Field(default=1_000, ge=1)
    eval_windows: int = Field(default=1_024, ge=1)
    probe_timesteps: int = Field(default=4_000, ge=100)
    patience: int = Field(default=4, ge=1)
    num_workers: int = Field(default=4, ge=0)
    max_episodes_per_dataset: int | None = Field(default=None, ge=1)
    seed: int = Field(default=0, ge=0)
    device: str = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "fp32"


class JEPAEvaluation(BaseModel):
    """Record one periodic validation pass."""

    model_config = ConfigDict(extra="forbid")

    step: int
    probe_score: float
    cube_error_cm: float
    tcp_error_cm: float
    validation_loss: float
    prediction_loss: float | None = None
    sigreg: float | None = None
    persistence_ratio: float | None = None
    latent_std: float | None = None
    effective_rank: float | None = None
    gate_magnitude: float | None = None


class JEPATrainingResult(BaseModel):
    """Summarize a completed training run for its run record."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, int]
    steps_completed: int
    train_windows: int
    train_episodes: int
    holdout_episodes: int
    validation_episodes: int
    history: list[JEPAEvaluation]
    seconds_per_step: float
    peak_memory_mb: float | None
    best_step: int
    stopped_early: bool = False


class ResumableBatchSampler(Sampler[list[int]]):
    """Yield seeded, epoch-wise shuffled batches forever, starting at any step."""

    def __init__(
        self, size: int, batch_size: int, *, seed: int, start_step: int = 0
    ) -> None:
        """Index ``size`` samples; each epoch drops the last partial batch."""
        if size < batch_size:
            raise ValueError(f"Need at least {batch_size} samples, found {size}")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.batches_per_epoch = size // batch_size

    def epoch_order(self, epoch: int) -> NDArray[np.int64]:
        """Return the reproducible sample permutation of one epoch."""
        return np.random.default_rng([self.seed, epoch]).permutation(self.size)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches from ``start_step`` onward."""
        epoch, offset = divmod(self.start_step, self.batches_per_epoch)
        while True:
            order = self.epoch_order(epoch)
            for batch in range(offset, self.batches_per_epoch):
                start = batch * self.batch_size
                yield order[start : start + self.batch_size].tolist()
            epoch, offset = epoch + 1, 0


class _LabelledTimesteps(Dataset[dict[str, Tensor]]):
    """Pair timestep images with standardized supervised targets."""

    def __init__(self, images: TimestepDataset, targets: Tensor) -> None:
        """Require one target row per timestep."""
        if len(images) != len(targets):
            raise ValueError("Every timestep needs one target row")
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one timestep's images and target row."""
        return {**self.images[index], "target": self.targets[index]}


def supervised_targets(
    labels: ProbeLabels,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return continuous targets (N, 11) and flags (N, 2) for the supervised ceiling."""
    continuous = np.concatenate(
        [labels.regression_target(name) for name in SUPERVISED_TARGETS], axis=1
    )
    flags = np.stack([getattr(labels, name) for name in FLAG_TARGETS], axis=1)
    return continuous, flags.astype(np.float64)


def _sample_episode_rows(
    episodes: Sequence[EpisodeRef], count: int, seed: int
) -> tuple[list[EpisodeRef], NDArray[np.int64]]:
    """Choose up to ``count`` timesteps; return their episodes and positions in them."""
    rows = window_starts(episodes, 1)
    if len(rows) <= count:
        return list(episodes), np.arange(len(rows))
    chosen = np.sort(
        np.random.default_rng(seed).choice(len(rows), count, replace=False)
    )
    return list(episodes), chosen


class _ProbeSet:
    """A fixed set of timesteps, their images, and labels for validation probes."""

    def __init__(
        self,
        caches: Sequence[FrameCache],
        episodes: Sequence[EpisodeRef],
        count: int,
        seed: int,
    ) -> None:
        """Select timesteps reproducibly and load their labels."""
        episodes, chosen = _sample_episode_rows(episodes, count, seed)
        labels = labels_for_episodes(caches, episodes)
        self.labels = labels.subset(chosen)
        self.images = torch.utils.data.Subset(
            TimestepDataset(caches, episodes), chosen.tolist()
        )

    def features(
        self, encode: Any, device: torch.device, batch_size: int
    ) -> NDArray[np.float64]:
        """Encode every selected timestep with ``encode(front, wrist)``."""
        outputs = []
        loader = _loader(self.images, batch_size=batch_size)
        with torch.no_grad():
            for batch in loader:
                value = encode(batch["front"].to(device), batch["wrist"].to(device))
                outputs.append(value.float().cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)


def probe_validation(
    train_features: NDArray[np.float64],
    train_labels: ProbeLabels,
    validation_features: NDArray[np.float64],
    validation_labels: ProbeLabels,
) -> tuple[float, float, float]:
    """Fit ridge probes for cube and TCP position; return score and errors (cm).

    The score averages the validation MSE of the six position coordinates,
    each divided by its training variance (lower is better).
    """
    normalized = []
    errors = []
    probes = fit_ridge_targets(
        train_features,
        {
            target: train_labels.regression_target(target)
            for target in _PROBE_SCORE_TARGETS
        },
        train_labels.episode,
    )
    for target in _PROBE_SCORE_TARGETS:
        prediction = probes[target].predict(validation_features)
        truth = validation_labels.regression_target(target)
        variance = np.maximum(train_labels.regression_target(target).var(axis=0), 1e-12)
        normalized.extend(((prediction - truth) ** 2).mean(axis=0) / variance)
        errors.append(
            float(per_sample_error(target, prediction, validation_labels).mean())
        )
    return float(np.mean(normalized)), errors[0], errors[1]


def _learning_rate_lambda(config: JEPATrainingConfig) -> Any:
    """Return a linear-warmup, cosine-decay learning-rate multiplier."""

    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            config.steps - config.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return multiplier


def _optimizer(model: nn.Module, config: JEPATrainingConfig) -> torch.optim.Optimizer:
    """Create AdamW, without weight decay on biases, norms, and embeddings."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "embedding" in name or "token" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def _peak_memory_mb(device: torch.device) -> float | None:
    """Return driver-allocated accelerator memory in MiB, when measurable."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None


def _synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work so timings are accurate."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def train(
    config: JEPATrainingConfig,
    *,
    run: TrainingRun,
    train_caches: Sequence[FrameCache],
    train_episodes: Sequence[EpisodeRef],
    holdout_episodes: Sequence[EpisodeRef],
    validation_caches: Sequence[FrameCache],
    validation_episodes: Sequence[EpisodeRef],
    resume_checkpoint: str | None = None,
    stop_at: int | None = None,
) -> tuple[nn.Module, JEPATrainingResult]:
    """Train the configured model, evaluating and checkpointing periodically.

    ``stop_at`` ends the session early at that step (with an evaluation and a
    checkpoint) without changing the learning-rate schedule, so a run can be
    split into shorter sessions and resumed exactly.
    """
    device = run.device
    last_step = config.steps if stop_at is None else min(stop_at, config.steps)
    if config.mode == "jepa":
        model: nn.Module = JEPA(config.model)
        parameters = {
            "encoder": count_parameters(model.encoder),
            "projector": count_parameters(model.projector),
            "predictor": count_parameters(model.predictor),
        }
        dataset: Dataset[dict[str, Tensor]] = WindowDataset(
            train_caches, train_episodes, window=config.model.history + 1
        )
    else:
        train_labels = labels_for_episodes(train_caches, train_episodes)
        continuous, flags = supervised_targets(train_labels)
        target_mean = continuous.mean(axis=0)
        target_std = np.maximum(continuous.std(axis=0), 1e-6)
        targets = torch.as_tensor(
            np.concatenate(((continuous - target_mean) / target_std, flags), axis=1),
            dtype=torch.float32,
        )
        model = SupervisedEncoder(config.model, outputs=targets.shape[1])
        parameters = {
            "encoder": count_parameters(model.encoder),
            "projector": count_parameters(model.projector),
            "head": count_parameters(model.head),
        }
        dataset = _LabelledTimesteps(
            TimestepDataset(train_caches, train_episodes), targets
        )
    model = model.to(device)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _learning_rate_lambda(config)
    )

    history: list[JEPAEvaluation] = []
    step = 0
    train_seconds = 0.0
    best_step, best_loss, best_state = 0, float("inf"), None
    if resume_checkpoint is not None:
        checkpoint = load_training_checkpoint(
            resume_checkpoint, model, optimizer, scheduler=scheduler
        )
        step = checkpoint.step
        history = [
            JEPAEvaluation.model_validate(item)
            for item in checkpoint.extra_state["history"]
        ]
        train_seconds = float(checkpoint.extra_state["train_seconds"])
        best_step = int(checkpoint.extra_state["best_step"])
        best_loss = float(checkpoint.extra_state["best_loss"])

    probe_train = _ProbeSet(
        train_caches, train_episodes, config.probe_timesteps, config.seed
    )
    probe_validation_set = _ProbeSet(
        validation_caches, validation_episodes, config.probe_timesteps, config.seed + 1
    )
    holdout = (
        WindowDataset(train_caches, holdout_episodes, window=config.model.history + 1)
        if config.mode == "jepa" and holdout_episodes
        else None
    )
    supervised_validation: (
        tuple[Tensor, NDArray[np.float64], NDArray[np.float64]] | None
    ) = None
    if config.mode == "supervised":
        continuous, flags = supervised_targets(probe_validation_set.labels)
        supervised_validation = (
            torch.as_tensor(
                np.concatenate(
                    ((continuous - target_mean) / target_std, flags), axis=1
                ),
                dtype=torch.float32,
            ),
            target_mean,
            target_std,
        )

    loader = _loader(
        dataset,
        batch_sampler=ResumableBatchSampler(
            len(dataset), config.batch_size, seed=config.seed, start_step=step
        ),
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    peak_memory = _peak_memory_mb(device)
    stopped_early = False
    batches = iter(loader)
    model.train()
    while step < last_step:
        batch = next(batches)
        started = time.perf_counter()
        front = batch["front"].to(device, non_blocking=True)
        wrist = batch["wrist"].to(device, non_blocking=True)
        metrics: dict[str, float] = {}
        if isinstance(model, JEPA):
            directions = random_directions(
                config.model.latent_dim, config.model.sigreg_directions, device=device
            )
            with run.autocast():
                losses = model.losses(
                    front,
                    wrist,
                    batch["actions"].to(device, non_blocking=True),
                    directions,
                )
            loss = losses.total
            metrics = {
                "train/prediction_loss": float(losses.prediction.detach()),
                "train/sigreg": float(losses.sigreg.detach()),
                "train/gate_magnitude": float(losses.gate_magnitude),
            }
        else:
            with run.autocast():
                output = model(front, wrist)
            loss = _supervised_loss(output.float(), batch["target"].to(device))
        step += 1
        run.optimize(loss, model, optimizer, step=step, scheduler=scheduler)
        _synchronize(device)
        train_seconds += time.perf_counter() - started
        if metrics:
            run.log_metrics(step, metrics)

        if step % config.eval_every == 0 or step == last_step:
            evaluation = _evaluate(
                model,
                config,
                device=device,
                step=step,
                probe_train=probe_train,
                validation_set=probe_validation_set,
                holdout=holdout,
                supervised_validation=supervised_validation,
            )
            history.append(evaluation)
            run.log_metrics(
                step,
                {
                    f"validation/{name}": value
                    for name, value in evaluation.model_dump().items()
                    if name != "step" and value is not None
                },
            )
            memory = _peak_memory_mb(device)
            if memory is not None:
                peak_memory = max(peak_memory or 0.0, memory)
            if evaluation.validation_loss < best_loss:
                best_step, best_loss = step, evaluation.validation_loss
                if config.mode == "supervised":
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
            elif (
                config.mode == "supervised"
                and len(history) - _index_of(history, best_step) > config.patience
            ):
                stopped_early = True
            model.train()
        if step % config.checkpoint_every == 0 or step == last_step or stopped_early:
            run.save_checkpoint(
                run.output_dir / "checkpoint.pt",
                model,
                optimizer,
                step=step,
                scheduler=scheduler,
                extra_state={
                    "history": [item.model_dump() for item in history],
                    "train_seconds": train_seconds,
                    "best_step": best_step,
                    "best_loss": best_loss,
                },
            )
        if stopped_early:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if config.mode == "jepa":
        best_step = step
    result = JEPATrainingResult(
        parameters=parameters,
        steps_completed=step,
        train_windows=len(dataset),
        train_episodes=len(train_episodes),
        holdout_episodes=len(holdout_episodes),
        validation_episodes=len(validation_episodes),
        history=history,
        seconds_per_step=train_seconds / max(step, 1),
        peak_memory_mb=peak_memory,
        best_step=best_step,
        stopped_early=stopped_early,
    )
    return model, result


def _loader(dataset: Dataset[dict[str, Tensor]], **options: Any) -> DataLoader:
    """Create a DataLoader that never draws from the global PyTorch RNG.

    A DataLoader iterator otherwise seeds itself from the global generator, so
    resuming would shift the SIGReg direction samples that follow.
    """
    return DataLoader(dataset, generator=torch.Generator().manual_seed(0), **options)


def _index_of(history: Sequence[JEPAEvaluation], step: int) -> int:
    """Return the history position of ``step`` (or -1 before any evaluation)."""
    for index, item in enumerate(history):
        if item.step == step:
            return index
    return -1


def _supervised_loss(output: Tensor, target: Tensor) -> Tensor:
    """Return MSE on standardized continuous targets plus BCE on the two flags."""
    flags = len(FLAG_TARGETS)
    return functional.mse_loss(output[:, :-flags], target[:, :-flags]) + (
        functional.binary_cross_entropy_with_logits(
            output[:, -flags:], target[:, -flags:]
        )
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    config: JEPATrainingConfig,
    *,
    device: torch.device,
    step: int,
    probe_train: _ProbeSet,
    validation_set: _ProbeSet,
    holdout: WindowDataset | None,
    supervised_validation: tuple[Tensor, NDArray[np.float64], NDArray[np.float64]]
    | None,
) -> JEPAEvaluation:
    """Run probe validation and, for the JEPA, held-out prediction and collapse checks."""
    model.eval()
    encoder = model.encoder

    def encode(front: Tensor, wrist: Tensor) -> Tensor:
        return global_readout(encoder(front, wrist))

    score, cube_error, tcp_error = probe_validation(
        probe_train.features(encode, device, 256),
        probe_train.labels,
        validation_set.features(encode, device, 256),
        validation_set.labels,
    )
    if isinstance(model, SupervisedEncoder):
        if supervised_validation is None:
            raise ValueError("Supervised validation targets are missing")
        targets = supervised_validation[0]
        outputs = []
        loader = _loader(validation_set.images, batch_size=256)
        for batch in loader:
            outputs.append(
                model(batch["front"].to(device), batch["wrist"].to(device))
                .float()
                .cpu()
            )
        validation_loss = float(_supervised_loss(torch.cat(outputs), targets))
        return JEPAEvaluation(
            step=step,
            probe_score=score,
            cube_error_cm=cube_error,
            tcp_error_cm=tcp_error,
            validation_loss=validation_loss,
        )
    if not isinstance(model, JEPA):
        raise TypeError("Unsupported model type")
    if holdout is None:
        return JEPAEvaluation(
            step=step,
            probe_score=score,
            cube_error_cm=cube_error,
            tcp_error_cm=tcp_error,
            validation_loss=score,
        )
    generator = torch.Generator().manual_seed(config.seed)
    directions = random_directions(
        config.model.latent_dim,
        config.model.sigreg_directions,
        device=torch.device("cpu"),
        generator=generator,
    ).to(device)
    indices = np.random.default_rng(config.seed).permutation(len(holdout))[
        : config.eval_windows
    ]
    loader = _loader(
        torch.utils.data.Subset(holdout, indices.tolist()), batch_size=config.batch_size
    )
    totals = {"prediction": 0.0, "sigreg": 0.0, "persistence": 0.0, "gate": 0.0}
    latents = []
    batches = 0
    for batch in loader:
        losses = model.losses(
            batch["front"].to(device),
            batch["wrist"].to(device),
            batch["actions"].to(device),
            directions,
        )
        persistence = functional.mse_loss(
            losses.latents[:, :-1].float(), losses.latents[:, 1:].float()
        )
        totals["prediction"] += float(losses.prediction)
        totals["sigreg"] += float(losses.sigreg)
        totals["persistence"] += float(persistence)
        totals["gate"] += float(losses.gate_magnitude)
        latents.append(losses.latents.flatten(0, 1).float().cpu())
        batches += 1
    all_latents = torch.cat(latents)
    prediction = totals["prediction"] / batches
    return JEPAEvaluation(
        step=step,
        probe_score=score,
        cube_error_cm=cube_error,
        tcp_error_cm=tcp_error,
        validation_loss=prediction
        + config.model.sigreg_weight * totals["sigreg"] / batches,
        prediction_loss=prediction,
        sigreg=totals["sigreg"] / batches,
        persistence_ratio=prediction / max(totals["persistence"] / batches, 1e-12),
        latent_std=float(all_latents.std(dim=0).mean()),
        effective_rank=effective_rank(all_latents),
        gate_magnitude=totals["gate"] / batches,
    )

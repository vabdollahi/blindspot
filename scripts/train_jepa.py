"""Train the Phase 4 in-domain JEPA (or its supervised ceiling) on T3-L1 frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import ConfigDict

from common.training import TrainingRun, TrainingRunRecord, TrainingSettings
from encoders.frames import (
    FrameCache,
    all_episodes,
    build_frame_cache,
    split_holdout_episodes,
)
from encoders.jepa import JEPAConfig
from encoders.training import JEPATrainingConfig, JEPATrainingResult, train

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = _PROJECT_ROOT / "data" / "cache" / "frames"


class JEPARunRecord(TrainingRunRecord):
    """Validate the reproducibility record of a JEPA or supervised-ceiling run."""

    model_config = ConfigDict(extra="forbid")

    config: JEPATrainingConfig
    frame_caches: dict[Literal["train", "validation"], list[str]]
    result: JEPATrainingResult
    checkpoint: str


@dataclass(frozen=True)
class _Caches:
    """Hold the opened training and validation frame caches."""

    train: list[FrameCache]
    validation: list[FrameCache]


def _parse_args() -> argparse.Namespace:
    """Parse JEPA training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("jepa", "supervised"), default="jepa")
    parser.add_argument(
        "--train-datasets", nargs="+", default=["interaction-l1", "t3-l1"]
    )
    parser.add_argument(
        "--validation-dataset",
        default="t3-l1-probe-val",
        help="Held-out validation-seed dataset for probe validation; 'none' uses "
        "the training holdout (smoke runs only).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=_PROJECT_ROOT / "data" / "datasets"
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue an interrupted run in place from its checkpoint.pt.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6_000,
        help="Run length; the cosine schedule spans it (6,000 is about 55 min on an M3 Max).",
    )
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="End this session at that step; continue later with --resume.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--sigreg-weight", type=float, default=0.1, help="lambda")
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--probe-timesteps", type=int, default=4_000)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        help="Cache and train on only the first episodes of each dataset (smoke runs).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="fp32",
        help="fp32 measured faster than fp16 or bf16 autocast for this model on MPS.",
    )
    return parser.parse_args()


def _select_output_dir(requested: Path, automatic_versioning: bool) -> Path:
    """Select a fresh output directory without replacing earlier runs."""
    if not requested.exists():
        return requested
    if not automatic_versioning:
        raise FileExistsError(f"Training output already exists: {requested}")
    run_number = 2
    while (
        candidate := requested.with_name(f"{requested.name}-run-{run_number:02d}")
    ).exists():
        run_number += 1
    return candidate


def _load_caches(
    config: JEPATrainingConfig, data_root: Path, cache_root: Path
) -> _Caches:
    """Build or reuse the frame caches of every training and validation dataset."""
    train = [
        build_frame_cache(
            data_root / name,
            cache_root,
            tasks=config.tasks,
            max_episodes=config.max_episodes_per_dataset,
        )
        for name in config.train_datasets
    ]
    validation = (
        [
            build_frame_cache(
                data_root / config.validation_dataset,
                cache_root,
                tasks=config.tasks,
                max_episodes=config.max_episodes_per_dataset,
            )
        ]
        if config.validation_dataset
        else []
    )
    return _Caches(train=train, validation=validation)


def _default_output(config: JEPATrainingConfig) -> Path:
    """Return the default run directory for a mode, lambda, and seed."""
    root = _PROJECT_ROOT / "checkpoints" / "jepa" / "t3-l1"
    if config.mode == "supervised":
        return root / f"supervised-seed-{config.seed}"
    return root / f"lambda-{config.model.sigreg_weight:g}-seed-{config.seed}"


def main() -> None:
    """Train one run and write its checkpoint and run record."""
    arguments = _parse_args()
    resume_step: int | None = None
    if arguments.resume is not None:
        output_dir = arguments.resume.expanduser().resolve()
        previous = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
        if previous.get("status") == "running":
            raise RuntimeError(f"Run is still marked running: {output_dir}")
        config = JEPATrainingConfig.model_validate(previous["config"])
        checkpoint = torch.load(
            output_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        resume_step = int(checkpoint["step"])
        del checkpoint
    else:
        validation = (
            None
            if arguments.validation_dataset == "none"
            else arguments.validation_dataset
        )
        config = JEPATrainingConfig(
            mode=arguments.mode,
            model=JEPAConfig(sigreg_weight=arguments.sigreg_weight),
            train_datasets=arguments.train_datasets,
            validation_dataset=validation,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            warmup_steps=arguments.warmup_steps,
            weight_decay=arguments.weight_decay,
            eval_every=arguments.eval_every,
            checkpoint_every=arguments.checkpoint_every,
            probe_timesteps=arguments.probe_timesteps,
            patience=arguments.patience,
            num_workers=arguments.workers,
            max_episodes_per_dataset=arguments.max_episodes_per_dataset,
            seed=arguments.seed,
            device=arguments.device,
            precision=arguments.precision,
        )
        output_dir = _select_output_dir(
            (arguments.output_dir or _default_output(config)).expanduser().resolve(),
            automatic_versioning=arguments.output_dir is None,
        )
    caches = _load_caches(
        config,
        arguments.data_root.expanduser().resolve(),
        arguments.cache_root.expanduser().resolve(),
    )
    train_episodes, holdout_episodes = split_holdout_episodes(
        caches.train, fraction=config.holdout_fraction, seed=config.seed
    )
    if caches.validation:
        validation_caches = caches.validation
        validation_episodes = all_episodes(validation_caches)
    else:
        validation_caches, validation_episodes = caches.train, holdout_episodes
    dataset_ids = [cache.dataset_id for cache in caches.train]
    settings = TrainingSettings(
        seed=config.seed, device=config.device, precision=config.precision
    )
    with TrainingRun(
        output_dir,
        config=config,
        dataset_id="+".join(dataset_ids),
        dataset_version="+".join(
            cache.manifest.key.dataset_version for cache in caches.train
        ),
        seeds={
            "train": sorted({reference.episode.seed for reference in train_episodes}),
            "validation": sorted(
                {reference.episode.seed for reference in validation_episodes}
            ),
        },
        settings=settings,
        resume_existing=resume_step is not None,
        step_offset=resume_step or 0,
    ) as run:
        _, result = train(
            config,
            run=run,
            train_caches=caches.train,
            train_episodes=train_episodes,
            holdout_episodes=holdout_episodes,
            validation_caches=validation_caches,
            validation_episodes=validation_episodes,
            resume_checkpoint=(
                str(output_dir / "checkpoint.pt") if resume_step is not None else None
            ),
            stop_at=arguments.stop_after_steps,
        )
    record_data = run.record.model_dump(mode="python")
    record_data.update(
        {
            "config": config,
            "frame_caches": {
                "train": [str(cache.directory) for cache in caches.train],
                "validation": [str(cache.directory) for cache in caches.validation],
            },
            "result": result,
            "checkpoint": str(output_dir / "checkpoint.pt"),
        }
    )
    record = JEPARunRecord.model_validate(record_data)
    temporary = output_dir / "run.json.tmp"
    temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "run.json")
    print(record.model_dump_json(indent=2, exclude={"seeds"}))


if __name__ == "__main__":
    main()

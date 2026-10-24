"""Train and record a state or image behavior-cloning baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from common.training import TrainingRun, TrainingRunRecord, TrainingSettings
from eval.runner import evaluate
from eval.schema import EvaluationSummary
from policies.bc import (
    BCEpochMetrics,
    BCPolicyType,
    BCTrainingConfig,
    BCTrainingResult,
    LeRobotBCDataset,
    load_bc_policy,
    save_bc_checkpoint,
    train_behavior_cloning,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BCRunRecord(TrainingRunRecord):
    """Validate the reproducibility record written beside a BC checkpoint."""

    model_config = ConfigDict(extra="forbid")

    config: BCTrainingConfig
    policy_type: BCPolicyType
    dataset_splits: dict[Literal["train", "validation"], str]
    result: BCTrainingResult
    rollout_validation: EvaluationSummary | None = None


def _parse_args() -> argparse.Namespace:
    """Parse training options without silently selecting a dataset run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--policy", choices=("state", "image"), required=True)
    parser.add_argument(
        "--data-root", type=Path, default=_PROJECT_ROOT / "data" / "datasets"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--rollout-validation",
        action="store_true",
        help="Evaluate the checkpoint on held-out simulator validation seeds.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: BaseModel) -> None:
    """Atomically persist a validated Pydantic run record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_output_dir(
    requested_dir: Path,
    *,
    automatic_versioning: bool,
) -> Path:
    """Select an unused checkpoint directory without replacing earlier runs."""
    if not requested_dir.exists() and not requested_dir.is_symlink():
        return requested_dir
    if not automatic_versioning:
        raise FileExistsError(f"Training output already exists: {requested_dir}")

    run_number = 2
    while True:
        candidate = requested_dir.with_name(
            f"{requested_dir.name}-run-{run_number:02d}"
        )
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        run_number += 1


def _train_bc_run(
    *,
    output_dir: Path,
    policy_type: BCPolicyType,
    train_data: LeRobotBCDataset,
    validation_data: LeRobotBCDataset,
    config: BCTrainingConfig,
    dataset_id: str,
    dataset_splits: dict[Literal["train", "validation"], str],
    dataset_version: str,
    seeds: dict[Literal["train", "validation"], list[int]],
    task: str,
    level: str,
    rollout_validation_seeds: list[int] | None,
) -> tuple[Path, BCRunRecord]:
    """Train BC through the shared run lifecycle without changing its optimizer."""
    steps_per_epoch = (len(train_data) + config.batch_size - 1) // config.batch_size
    settings = TrainingSettings(
        seed=config.seed,
        device=config.device,
        precision="fp32",
        max_grad_norm=config.max_grad_norm,
    )
    rollout_summary = None
    with TrainingRun(
        output_dir,
        config=config,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        seeds=seeds,
        settings=settings,
    ) as run:

        def log_epoch(metrics: BCEpochMetrics, mean_gradient_norm: float) -> None:
            """Log completed BC epoch metrics without altering training updates."""
            run.log_metrics(
                metrics.epoch * steps_per_epoch,
                {
                    "train/loss": metrics.training_l1,
                    "validation/loss": metrics.validation_l1,
                    "train/gradient_norm": mean_gradient_norm,
                    "train/learning_rate": config.learning_rate,
                },
            )

        model, result = train_behavior_cloning(
            policy_type,
            train_data,
            validation_data,
            config,
            on_epoch_end=log_epoch,
        )
        checkpoint_path = save_bc_checkpoint(
            model,
            policy_type,
            output_dir / "checkpoint.pt",
        )
        if rollout_validation_seeds is not None:
            if not rollout_validation_seeds:
                raise ValueError(
                    "Rollout validation requires held-out validation seeds"
                )
            rollout = evaluate(
                load_bc_policy(checkpoint_path, device="cpu"),
                task,
                level,
                rollout_validation_seeds,
                obs_mode="image" if policy_type == "image" else "state",
                workers=1,
                method=f"{policy_type}-bc-validation",
                results_dir=output_dir / "rollout-validation",
            )
            rollout_summary = rollout.summary

    record_data = run.record.model_dump(mode="python")
    record_data.update(
        {
            "config": config,
            "policy_type": policy_type,
            "dataset_splits": dataset_splits,
            "result": result,
            "rollout_validation": rollout_summary,
        }
    )
    record = BCRunRecord.model_validate(record_data)
    _write_json(output_dir / "run.json", record)
    return checkpoint_path, record


def main() -> None:
    """Train one BC model from a Phase 2.1 train/validation dataset pair."""
    arguments = _parse_args()
    config = BCTrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.workers,
        device=arguments.device,
        seed=arguments.seed,
        patience=arguments.patience,
        pretrained=not arguments.no_pretrained,
    )
    dataset_root = arguments.data_root.expanduser().resolve() / arguments.dataset_name
    requested_output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir
        else _PROJECT_ROOT
        / "checkpoints"
        / "bc"
        / arguments.dataset_name
        / arguments.policy
    )
    output_dir = _select_output_dir(
        requested_output_dir,
        automatic_versioning=arguments.output_dir is None,
    )
    manifest_path = dataset_root / "collection.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 2.1 collection manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_root = dataset_root / "train"
    validation_root = dataset_root / "validation"
    if not train_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError(
            f"Expected train and validation datasets under {dataset_root}"
        )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "BC training requires the optional imitation dependencies; "
            "run `uv sync --extra imitation`."
        ) from error

    dataset_id = f"blindspot/{arguments.dataset_name}"
    train_source = LeRobotDataset(
        repo_id=f"{dataset_id}-train",
        root=train_root,
        video_backend="pyav",
    )
    validation_source = LeRobotDataset(
        repo_id=f"{dataset_id}-validation",
        root=validation_root,
        video_backend="pyav",
    )
    policy_type: BCPolicyType = arguments.policy
    train_data = LeRobotBCDataset(train_source, policy_type)
    validation_data = LeRobotBCDataset(validation_source, policy_type)
    validation_seeds = (
        manifest["validation"]["seeds"] if arguments.rollout_validation else None
    )
    _, run_record = _train_bc_run(
        output_dir=output_dir,
        policy_type=policy_type,
        train_data=train_data,
        validation_data=validation_data,
        config=config,
        dataset_id=dataset_id,
        dataset_splits={
            "train": f"{dataset_id}-train",
            "validation": f"{dataset_id}-validation",
        },
        dataset_version=str(train_source.meta.info.codebase_version),
        seeds={
            "train": manifest["train"]["seeds"],
            "validation": manifest["validation"]["seeds"],
        },
        task=str(manifest["task"]),
        level=str(manifest.get("level", "L1")),
        rollout_validation_seeds=validation_seeds,
    )
    print(run_record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

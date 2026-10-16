"""Train and record a state or image behavior-cloning baseline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from eval.runner import evaluate
from eval.schema import EvaluationSummary
from policies.bc import (
    BCPolicyType,
    BCTrainingConfig,
    BCTrainingResult,
    LeRobotBCDataset,
    load_bc_policy,
    save_bc_checkpoint,
    train_behavior_cloning,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BCRunRecord(BaseModel):
    """Validate the reproducibility record written beside a BC checkpoint."""

    model_config = ConfigDict(extra="forbid")

    policy_type: BCPolicyType
    config: BCTrainingConfig
    git_commit: str
    dataset_id: str
    dataset_splits: dict[Literal["train", "validation"], str]
    dataset_version: str
    seeds: dict[Literal["train", "validation"], list[int]]
    hardware: dict[str, str | bool]
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


def _git_commit() -> str:
    """Return the current Git commit or 'unknown' outside a Git checkout."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


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
    model, result = train_behavior_cloning(
        policy_type,
        train_data,
        validation_data,
        config,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = save_bc_checkpoint(
        model,
        policy_type,
        output_dir / "checkpoint.pt",
    )
    rollout_validation = None
    if arguments.rollout_validation:
        validation_seeds = manifest["validation"]["seeds"]
        if not validation_seeds:
            raise ValueError("Rollout validation requires held-out validation seeds")
        rollout = evaluate(
            load_bc_policy(checkpoint_path, device="cpu"),
            str(manifest["task"]),
            str(manifest.get("level", "L1")),
            validation_seeds,
            obs_mode="image" if policy_type == "image" else "state",
            workers=1,
            method=f"{policy_type}-bc-validation",
            results_dir=output_dir / "rollout-validation",
        )
        rollout_validation = rollout.summary
    run_record = BCRunRecord(
        policy_type=policy_type,
        config=config,
        git_commit=_git_commit(),
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
        hardware={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": result.device,
            "mps_available": torch.backends.mps.is_available(),
        },
        result=result,
        rollout_validation=rollout_validation,
    )
    _write_json(output_dir / "run.json", run_record)
    print(run_record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

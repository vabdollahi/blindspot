"""Train the frozen-backbone SmolVLM2 action head on T3-L demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.training import TrainingRun, TrainingRunRecord, TrainingSettings
from data.vla_subset import (
    VLA_ACTIVE_CUBE_COUNT,
    VLA_FRAMES_PER_EPISODE,
    VLA_TARGET_COLORS,
    VLAFrameSelection,
    select_limited_t3l_frames,
)
from policies.vla import (
    DEFAULT_FEATURE_LAYER,
    MODEL_ID,
    VLAConfig,
    VLAFeatureCache,
    VLAFrameDataset,
    VLATrainingResult,
    load_frozen_smolvlm,
    load_vla_training_head,
    save_vla_checkpoint,
    train_vla,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LimitedVLADatasetSelection(BaseModel):
    """Record the limited profile's selected episodes and source frames."""

    model_config = ConfigDict(extra="forbid")

    target_colors: tuple[str, str]
    active_cube_count: int
    episodes_per_color: int
    frames_per_episode: int
    train: VLAFrameSelection
    validation: VLAFrameSelection


class VLARunRecord(TrainingRunRecord):
    """Validate the reproducibility record for a VLA baseline training run."""

    model_config = ConfigDict(extra="forbid")

    config: VLAConfig
    dataset_splits: dict[Literal["train", "validation"], str]
    result: VLATrainingResult
    checkpoint: str
    dataset_selection: LimitedVLADatasetSelection | None = None
    warm_start_count: int = Field(default=0, ge=0)
    initialized_from: str | None = None


def _parse_args() -> argparse.Namespace:
    """Parse VLA training options and require a dataset name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    parser.add_argument(
        "--data-root", type=Path, default=_PROJECT_ROOT / "data" / "datasets"
    )
    parser.add_argument("--output-dir", type=Path)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume-from",
        type=Path,
        help="Warm-start in place from a VLA checkpoint and reset the optimizer.",
    )
    checkpoint_group.add_argument(
        "--initialize-from",
        type=Path,
        help="Start a fresh, versioned run initialized from an existing head.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--profile", choices=("full", "limited"), default="full")
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=DEFAULT_FEATURE_LAYER,
        help="SmolVLM2 decoder layer whose hidden states feed the action head.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Future actions predicted per query (10 = 0.5 s at 20 Hz; 1 = single step).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="Reuse or store frozen-VLM features here, keyed by data and encoder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Action-head epochs to run; additional epochs when using --resume-from.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoding-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-encoding-samples", type=int)
    parser.add_argument("--max-train-steps", type=int)
    return parser.parse_args()


def _write_json(path: Path, payload: BaseModel) -> None:
    """Atomically persist a validated VLA run record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_output_dir(requested_dir: Path, automatic_versioning: bool) -> Path:
    """Select a fresh output directory without replacing earlier runs."""
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


def _load_dataset_split(
    dataset_root: Path,
    split: Literal["train", "validation"],
    dataset_name: str,
) -> Any:
    """Open one LeRobot split."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "VLA training requires the optional VLA dependencies; "
            "run `uv sync --extra vla`."
        ) from error
    source = LeRobotDataset(
        repo_id=f"blindspot/{dataset_name}-{split}",
        root=dataset_root / split,
        video_backend="pyav",
    )
    return source


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one dataset metadata file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_seeds(
    source_seeds: list[int],
    selection: VLAFrameSelection,
) -> list[int]:
    """Map selected episode indices back to the collection's seed list."""
    episode_ids = sorted(
        {
            episode_id
            for color_episode_ids in selection.episode_ids_by_color.values()
            for episode_id in color_episode_ids
        }
    )
    if not episode_ids or any(
        episode_id < 0 or episode_id >= len(source_seeds) for episode_id in episode_ids
    ):
        raise ValueError("Selected VLA episode IDs do not match collection seeds")
    return [source_seeds[episode_id] for episode_id in episode_ids]


def _load_resume_record(checkpoint: Path) -> VLARunRecord:
    """Load the completed run record associated with a warm-start checkpoint."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    run_record_path = checkpoint.parent / "run.json"
    if not run_record_path.is_file():
        raise FileNotFoundError(
            f"Warm-start requires the run record beside the checkpoint: "
            f"{run_record_path}"
        )
    try:
        record = VLARunRecord.model_validate_json(
            run_record_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError(
            f"{run_record_path} is not a current VLA run record; runs saved before "
            "checkpoint format 3 used the superseded last-token head and must be "
            "retrained"
        ) from error
    if record.status == "running":
        raise RuntimeError(
            f"Cannot warm-start a run still marked running: {run_record_path}"
        )
    if Path(record.checkpoint).expanduser().resolve() != checkpoint:
        raise ValueError(
            f"Run record checkpoint does not match the requested resume path: "
            f"{record.checkpoint}"
        )
    return record


def _validate_resume_record(
    record: VLARunRecord,
    *,
    config: VLAConfig,
    dataset_id: str,
    dataset_version: str,
    training_seeds: list[int],
    validation_seeds: list[int],
    dataset_selection: LimitedVLADatasetSelection | None,
) -> None:
    """Reject warm starts when the model, profile, or selected data changed."""
    if record.dataset_id != dataset_id or record.dataset_version != dataset_version:
        raise ValueError("Warm-start dataset ID or version does not match the run")
    if record.config.profile != config.profile:
        raise ValueError("Warm-start profile does not match the run")
    if record.config.model_id != config.model_id:
        raise ValueError("Warm-start model ID does not match the run")
    if record.config.chunk_size != config.chunk_size:
        raise ValueError("Warm-start chunk size does not match the run")
    if config.model_revision not in {
        record.config.model_revision,
        record.result.encoding.model_revision,
    }:
        raise ValueError("Warm-start model revision does not match the run")
    if record.seeds != {
        "train": training_seeds,
        "validation": validation_seeds,
    }:
        raise ValueError("Warm-start dataset seeds do not match the run")
    if record.dataset_selection != dataset_selection:
        raise ValueError("Warm-start frame selection does not match the run")


def main() -> None:
    """Train and record one VLA baseline run from T3-L demonstrations."""
    arguments = _parse_args()
    config = VLAConfig(
        model_id=arguments.model_id,
        model_revision=arguments.model_revision,
        profile=arguments.profile,
        feature_layer=arguments.feature_layer,
        chunk_size=arguments.chunk_size,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        encoding_batch_size=arguments.encoding_batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        dropout=arguments.dropout,
        num_workers=arguments.workers,
        device=arguments.device,
        seed=arguments.seed,
        patience=arguments.patience,
        max_encoding_samples=arguments.max_encoding_samples,
        max_train_steps=arguments.max_train_steps,
    )
    dataset_root = arguments.data_root.expanduser().resolve() / arguments.dataset_name
    manifest_path = dataset_root / "collection.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 collection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "T3-L":
        raise ValueError(
            "The frozen-backbone VLA baseline trains only on T3-L datasets"
        )
    if manifest.get("instruction_template_split") != "train":
        raise ValueError("VLA training requires the T3-L train-template dataset")
    train_root = dataset_root / "train"
    validation_root = dataset_root / "validation"
    if not train_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError(
            f"Expected train and validation datasets under {dataset_root}"
        )
    train_data = _load_dataset_split(dataset_root, "train", arguments.dataset_name)
    train_source = train_data
    validation_source = _load_dataset_split(
        dataset_root, "validation", arguments.dataset_name
    )
    if len(train_source) == 0 or len(validation_source) == 0:
        raise ValueError("T3-L train and validation datasets must be non-empty")
    limited_selection = None
    if config.profile == "limited":
        if (
            config.max_encoding_samples is not None
            or config.max_train_steps is not None
        ):
            raise ValueError(
                "The limited profile already bounds data and training; omit "
                "--max-encoding-samples and --max-train-steps"
            )
        train_selection = select_limited_t3l_frames(train_source, "train")
        validation_selection = select_limited_t3l_frames(
            validation_source,
            "validation",
        )
        train_data = VLAFrameDataset(
            train_source,
            frame_indices=train_selection.frame_indices,
        )
        validation_data = VLAFrameDataset(
            validation_source,
            frame_indices=validation_selection.frame_indices,
        )
        limited_selection = LimitedVLADatasetSelection(
            target_colors=VLA_TARGET_COLORS,
            active_cube_count=VLA_ACTIVE_CUBE_COUNT,
            episodes_per_color=min(
                len(episode_ids)
                for episode_ids in train_selection.episode_ids_by_color.values()
            ),
            frames_per_episode=VLA_FRAMES_PER_EPISODE,
            train=train_selection,
            validation=validation_selection,
        )
        training_seeds = _selected_seeds(
            manifest["train"]["seeds"],
            train_selection,
        )
        validation_seeds = _selected_seeds(
            manifest["validation"]["seeds"],
            validation_selection,
        )
    else:
        train_data = VLAFrameDataset(train_source)
        validation_data = VLAFrameDataset(validation_source)
        training_seeds = manifest["train"]["seeds"]
        validation_seeds = manifest["validation"]["seeds"]
    dataset_id = f"blindspot/{arguments.dataset_name}"
    dataset_version = train_data.dataset_version
    previous_run = (
        _load_resume_record(arguments.resume_from.expanduser().resolve())
        if arguments.resume_from
        else None
    )
    initialization_run = (
        _load_resume_record(arguments.initialize_from.expanduser().resolve())
        if arguments.initialize_from
        else None
    )
    source_run = previous_run or initialization_run
    if source_run is not None:
        _validate_resume_record(
            source_run,
            config=config,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            training_seeds=training_seeds,
            validation_seeds=validation_seeds,
            dataset_selection=limited_selection,
        )
    output_name = (
        "limited-red-blue-two-cube" if config.profile == "limited" else "prototype"
    )
    if previous_run is not None:
        resume_checkpoint = arguments.resume_from.expanduser().resolve()
        if (
            arguments.output_dir is not None
            and arguments.output_dir.expanduser().resolve() != resume_checkpoint.parent
        ):
            raise ValueError(
                "When resuming, --output-dir must match the checkpoint directory"
            )
        output_dir = resume_checkpoint.parent
    else:
        requested_output = (
            arguments.output_dir.expanduser().resolve()
            if arguments.output_dir
            else _PROJECT_ROOT
            / "checkpoints"
            / "vla"
            / arguments.dataset_name
            / output_name
        )
        output_dir = _select_output_dir(
            requested_output,
            automatic_versioning=arguments.output_dir is None,
        )
    encoder_revision = (
        source_run.result.encoding.model_revision
        if source_run is not None
        else config.model_revision
    )
    if (
        source_run is not None
        and source_run.config.feature_layer != config.feature_layer
    ):
        raise ValueError("Warm-start feature layer does not match the run")
    encoder = load_frozen_smolvlm(
        config.model_id,
        encoder_revision,
        device=config.device,
        feature_layer=config.feature_layer,
    )
    head_source = arguments.resume_from or arguments.initialize_from
    initial_head = (
        load_vla_training_head(
            head_source.expanduser().resolve(),
            model_id=encoder.model_id,
            model_revision=encoder.model_revision,
            training_profile=config.profile,
            hidden_size=encoder.hidden_size,
            feature_layer=config.feature_layer,
            chunk_size=config.chunk_size,
        )
        if head_source is not None
        else None
    )
    feature_cache = (
        VLAFeatureCache(
            directory=arguments.feature_cache_dir.expanduser().resolve(),
            split_keys={
                split: {
                    "dataset_id": dataset_id,
                    "split": f"{dataset_id}-{split}",
                    "collection_sha256": _file_sha256(manifest_path),
                    "info_sha256": _file_sha256(
                        dataset_root / split / "meta" / "info.json"
                    ),
                    "source_seeds": manifest[split]["seeds"],
                    "frame_indices": (
                        selection.frame_indices if selection is not None else "all"
                    ),
                }
                for split, selection in (
                    (
                        "train",
                        limited_selection.train if limited_selection else None,
                    ),
                    (
                        "validation",
                        limited_selection.validation if limited_selection else None,
                    ),
                )
            },
        )
        if arguments.feature_cache_dir is not None
        else None
    )
    settings = TrainingSettings(
        seed=config.seed, device=config.device, precision="fp32"
    )
    with TrainingRun(
        output_dir,
        config=config,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        seeds={
            "train": training_seeds,
            "validation": validation_seeds,
        },
        settings=settings,
        resume_existing=previous_run is not None,
        step_offset=(
            previous_run.result.optimizer_steps if previous_run is not None else 0
        ),
    ) as run:
        head, result, encoder = train_vla(
            train_data,
            validation_data,
            config,
            encoder=encoder,
            run=run,
            initial_head=initial_head,
            optimizer_step_offset=(
                previous_run.result.optimizer_steps if previous_run is not None else 0
            ),
            epoch_offset=(
                max(
                    (metrics.epoch for metrics in previous_run.result.history),
                    default=previous_run.result.best_epoch,
                )
                if previous_run is not None
                else 0
            ),
            initial_best_epoch=(
                previous_run.result.best_epoch if previous_run is not None else 0
            ),
            initial_best_loss=(
                previous_run.result.best_validation_loss
                if previous_run is not None
                else None
            ),
            feature_cache=feature_cache,
        )
        if previous_run is not None:
            previous_best_loss = previous_run.result.best_validation_loss
            current_best_loss = result.best_validation_loss
            improved = current_best_loss < previous_best_loss
            result = result.model_copy(
                update={
                    "initial_validation_l1": (
                        previous_run.result.initial_validation_l1
                    ),
                    "initial_validation_loss": (
                        previous_run.result.initial_validation_loss
                    ),
                    "best_validation_l1": (
                        result.best_validation_l1
                        if improved
                        else previous_run.result.best_validation_l1
                    ),
                    "best_validation_loss": min(
                        current_best_loss,
                        previous_best_loss,
                    ),
                    "best_epoch": (
                        result.best_epoch
                        if improved
                        else previous_run.result.best_epoch
                    ),
                    "optimizer_steps": result.optimizer_steps,
                    "history": [
                        *previous_run.result.history,
                        *result.history,
                    ],
                    "validation_action_mae": (
                        result.validation_action_mae
                        if improved
                        else previous_run.result.validation_action_mae
                    ),
                    "validation_episode_l1": (
                        result.validation_episode_l1
                        if improved
                        else previous_run.result.validation_episode_l1
                    ),
                    "validation_gripper_accuracy": (
                        result.validation_gripper_accuracy
                        if improved
                        else previous_run.result.validation_gripper_accuracy
                    ),
                    "validation_gripper_bce": (
                        result.validation_gripper_bce
                        if improved
                        else previous_run.result.validation_gripper_bce
                    ),
                    "ablated_validation_l1": (
                        result.ablated_validation_l1
                        if improved
                        else previous_run.result.ablated_validation_l1
                    ),
                    "ablated_validation_action_mae": (
                        result.ablated_validation_action_mae
                        if improved
                        else previous_run.result.ablated_validation_action_mae
                    ),
                    "validation_chunk_l1": (
                        result.validation_chunk_l1
                        if improved
                        else previous_run.result.validation_chunk_l1
                    ),
                    "gripper_positive_weight": result.gripper_positive_weight,
                }
            )
        checkpoint_path = save_vla_checkpoint(
            head,
            encoder.model_id,
            encoder.model_revision,
            output_dir / "checkpoint.pt",
            training_profile=config.profile,
            feature_layer=config.feature_layer,
        )

    record_data = run.record.model_dump(mode="python")
    record_data.update(
        {
            "config": config,
            "dataset_splits": {
                "train": f"{dataset_id}-train",
                "validation": f"{dataset_id}-validation",
            },
            "result": result,
            "checkpoint": str(checkpoint_path),
            "dataset_selection": limited_selection,
            "warm_start_count": (
                previous_run.warm_start_count + 1 if previous_run is not None else 0
            ),
            "initialized_from": (
                str(arguments.initialize_from.expanduser().resolve())
                if initialization_run is not None
                else (
                    previous_run.initialized_from if previous_run is not None else None
                )
            ),
        }
    )
    record = VLARunRecord.model_validate(record_data)
    _write_json(output_dir / "run.json", record)
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

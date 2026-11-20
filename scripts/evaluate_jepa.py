"""Probe the Phase 4 encoders on T3-L1 and roll out the JEPA predictor."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from common.training import resolve_device
from encoders.evaluation import (
    TIMESTEP_SECONDS,
    ProbeOutcome,
    RolloutTotals,
    SplitData,
    load_jepa_run,
    load_supervised_run,
    probe_seed,
    rollout_seed,
    score_predictions,
    summarize,
    summarize_rollouts,
    training_curve,
)
from encoders.frames import EpisodeRef, FrameCache, all_episodes, build_frame_cache
from encoders.jepa import count_parameters
from encoders.pretrained import ViTFeatureExtractor, extract_features, load_dinov2
from encoders.probes import ProbeLabels, labels_for_episodes
from encoders.training import probe_validation
from encoders.vit import TwoViewViT
from eval.schema import (
    JEPAProbeResult,
    JEPARolloutResult,
    MetricEstimate,
    ProbeEncoder,
    ProbeRow,
    TrainingCurvePoint,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Readouts and probe models per encoder kind (docs/jepa.md, Evaluation A).
_ROWS = {
    "random": (("global", "linear"),),
    "jepa": (
        ("latent", "linear"),
        ("global", "linear"),
        ("global", "mlp"),
        ("grid", "linear"),
    ),
    "supervised": (("global", "linear"), ("output", "trained")),
    "pretrained": (("global", "linear"), ("grid", "linear")),
}


def _parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--supervised-run", type=Path)
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--dinov2-revision", default="main")
    parser.add_argument(
        "--train-datasets", nargs="+", default=["interaction-l1", "t3-l1"]
    )
    parser.add_argument("--validation-dataset", default="t3-l1-probe-val")
    parser.add_argument("--test-dataset", default="t3-l1-probe-test")
    parser.add_argument(
        "--data-root", type=Path, default=_PROJECT_ROOT / "data" / "datasets"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=_PROJECT_ROOT / "data" / "cache" / "frames"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_PROJECT_ROOT / "eval" / "results" / "jepa"
    )
    parser.add_argument(
        "--probe-train-stride",
        type=int,
        default=1,
        help="Fit probes on every n-th training timestep (1 = all, 5 Hz).",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes-per-dataset", type=int)
    return parser.parse_args()


def _caches(names: list[str], arguments: argparse.Namespace) -> list[FrameCache]:
    """Build or reuse the frame caches of the named datasets."""
    return [
        build_frame_cache(
            arguments.data_root / name,
            arguments.cache_root,
            tasks=("T3",),
            max_episodes=arguments.max_episodes_per_dataset,
        )
        for name in names
    ]


def _git_commit() -> str:
    """Return the checked-out commit hash."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stride_rows(labels: ProbeLabels, stride: int) -> np.ndarray:
    """Return every ``stride``-th timestep of each episode."""
    if stride == 1:
        return np.arange(len(labels))
    keep = np.zeros(len(labels), dtype=bool)
    for episode in np.unique(labels.episode):
        rows = np.flatnonzero(labels.episode == episode)
        keep[rows[::stride]] = True
    return np.flatnonzero(keep)


def main() -> None:
    """Evaluate every requested encoder and write the results JSON files."""
    arguments = _parse_args()
    device = resolve_device(arguments.device)
    train_caches = _caches(arguments.train_datasets, arguments)
    validation_caches = _caches([arguments.validation_dataset], arguments)
    test_caches = _caches([arguments.test_dataset], arguments)
    splits: dict[str, tuple[list[FrameCache], list[EpisodeRef]]] = {
        "train": (train_caches, all_episodes(train_caches)),
        "validation": (validation_caches, all_episodes(validation_caches)),
        "test": (test_caches, all_episodes(test_caches)),
    }
    labels = {name: labels_for_episodes(*split) for name, split in splits.items()}
    train_rows = _stride_rows(labels["train"], arguments.probe_train_stride)
    labels["train"] = labels["train"].subset(train_rows)
    episode_counts = {name: len(split[1]) for name, split in splits.items()}

    encoder_specs: list[tuple[str, str, list[tuple[int, object, str]]]] = []
    if arguments.random_seeds:
        encoder_specs.append(
            (
                "random-vit-t16",
                "random",
                [(seed, None, f"seed {seed}") for seed in arguments.random_seeds],
            )
        )
    if arguments.jepa_runs:
        encoder_specs.append(
            (
                "jepa-vit-t16",
                "jepa",
                [
                    (index, path, str(path))
                    for index, path in enumerate(arguments.jepa_runs)
                ],
            )
        )
    if arguments.supervised_run is not None:
        encoder_specs.append(
            (
                "supervised-vit-t16",
                "supervised",
                [(0, arguments.supervised_run, str(arguments.supervised_run))],
            )
        )
    if not arguments.skip_dinov2:
        encoder_specs.append(
            ("dinov2-vits14", "pretrained", [(0, None, "facebook/dinov2-small")])
        )

    rows: list[ProbeRow] = []
    encoders: list[ProbeEncoder] = []
    curves: dict[str, list[TrainingCurvePoint]] = {}
    rollout_totals = RolloutTotals()
    rollout_windows = 0
    rollout_runs: list[str] = []
    selection_scores: dict[str, list[float]] = {}
    for name, kind, members in encoder_specs:
        outcomes = {
            (readout, probe, split): ProbeOutcome()
            for readout, probe in _ROWS[kind]
            for split in ("validation", "test")
        }
        readouts = sorted({readout for readout, _ in _ROWS[kind]})
        parameters = 0
        sources = []
        for seed, source, description in members:
            jepa_model = None
            if kind == "random":
                torch.manual_seed(seed)
                encoder = TwoViewViT().to(device).eval()
                extractor = ViTFeatureExtractor(name, encoder)
                parameters = count_parameters(encoder)
            elif kind == "jepa":
                jepa_model, _, record = load_jepa_run(Path(source).resolve(), device)
                extractor = ViTFeatureExtractor(name, jepa_model.encoder, jepa_model)
                parameters = count_parameters(jepa_model.encoder) + count_parameters(
                    jepa_model.projector
                )
                curves[str(source)] = training_curve(record)
            elif kind == "supervised":
                supervised = load_supervised_run(
                    Path(source).resolve(),
                    device,
                    train_caches=train_caches,
                    train_datasets=arguments.train_datasets,
                )
                extractor = ViTFeatureExtractor(
                    name, supervised.model.encoder, supervised=supervised.model
                )
                parameters = count_parameters(supervised.model)
            else:
                extractor = load_dinov2(arguments.dinov2_revision, device=device)
                parameters = count_parameters(extractor.model)
                description = f"facebook/dinov2-small@{extractor.revision}"
            sources.append(description)
            data = {}
            for split, (caches, episodes) in splits.items():
                features = extract_features(
                    extractor,
                    caches,
                    episodes,
                    device=device,
                    batch_size=arguments.batch_size,
                    readouts=readouts,
                )
                if split == "train":
                    features = {
                        key: value[train_rows] for key, value in features.items()
                    }
                data[split] = SplitData(features, labels[split])
            selection_scores.setdefault(name, []).append(
                probe_validation(
                    data["train"].features["global"].astype(np.float64),
                    data["train"].labels,
                    data["validation"].features["global"].astype(np.float64),
                    data["validation"].labels,
                )[0]
            )
            for readout, probe in _ROWS[kind]:
                print(f"Probing {name} ({description}) {readout}/{probe}", flush=True)
                if probe == "trained":
                    regressors, classifiers = supervised.decoders()
                    score_predictions(
                        {
                            split: outcomes[(readout, probe, split)]
                            for split in ("validation", "test")
                        },
                        readout,
                        regressors,
                        classifiers,
                        {split: data[split] for split in ("validation", "test")},
                        episodes=episode_counts,
                    )
                    continue
                probe_seed(
                    {
                        split: outcomes[(readout, probe, split)]
                        for split in ("validation", "test")
                    },
                    readout,
                    probe,
                    data["train"],
                    {split: data[split] for split in ("validation", "test")},
                    episodes=episode_counts,
                    seed=seed,
                )
            if jepa_model is not None:
                rollout_windows = rollout_seed(
                    rollout_totals,
                    jepa_model,
                    data["train"].features["latent"],
                    data["train"].labels,
                    data["test"].features["latent"],
                    data["test"].labels,
                    test_caches,
                    splits["test"][1],
                    device=device,
                )
                rollout_runs.append(str(source))
            del data, extractor, jepa_model
            gc.collect()
        encoders.append(
            ProbeEncoder(
                name=name,
                kind=kind,
                seeds=[member[0] for member in members],
                parameters=parameters,
                sources=sources,
            )
        )
        for (readout, probe, split), outcome in outcomes.items():
            rows.append(
                ProbeRow(
                    encoder=name,
                    readout=readout,
                    probe=probe,
                    split=split,
                    metrics=summarize(outcome, resamples=arguments.resamples),
                )
            )

    def metric(
        encoder: str, split: str, key: str = "cube_xyz_cm"
    ) -> MetricEstimate | None:
        for row in rows:
            if (row.encoder, row.readout, row.probe, row.split) == (
                encoder,
                "global",
                "linear",
                split,
            ):
                return row.metrics.get(key)
        return None

    jepa_test, random_test = (
        metric("jepa-vit-t16", "test"),
        metric("random-vit-t16", "test"),
    )
    h1 = (
        jepa_test.ci_high < random_test.ci_low
        if jepa_test is not None and random_test is not None
        else None
    )
    # The Phase 6 encoder uses the lambda-selection criterion on validation episodes.
    scores = {name: float(np.mean(values)) for name, values in selection_scores.items()}
    candidates = [name for name in ("jepa-vit-t16", "dinov2-vits14") if name in scores]
    phase6 = min(candidates, key=scores.__getitem__) if len(candidates) == 2 else None

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    probe_result = JEPAProbeResult(
        git_commit=commit,
        train_datasets=arguments.train_datasets,
        validation_dataset=arguments.validation_dataset,
        test_dataset=arguments.test_dataset,
        train_timesteps=len(train_rows),
        test_episodes=episode_counts["test"],
        resamples=arguments.resamples,
        encoders=encoders,
        rows=rows,
        h1_pass=h1,
        phase6_encoder=phase6,
        validation_probe_scores=scores,
        training_curves=curves,
    )
    (output_dir / "T3-L1-probes.json").write_text(
        probe_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'T3-L1-probes.json'}")
    if rollout_runs:
        rollout_curves, latent, h4 = summarize_rollouts(
            rollout_totals, resamples=arguments.resamples
        )
        rollout = JEPARolloutResult(
            git_commit=commit,
            runs=rollout_runs,
            test_dataset=arguments.test_dataset,
            horizons_seconds=[
                TIMESTEP_SECONDS * step for step in range(1, len(latent) + 1)
            ],
            windows=rollout_windows,
            curves=rollout_curves,
            latent_mse_vs_persistence=latent,
            h4_pass=h4,
        )
        (output_dir / "T3-L1-rollout.json").write_text(
            rollout.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {output_dir / 'T3-L1-rollout.json'}")
    print(json.dumps({"h1_pass": h1, "phase6_encoder": phase6}, indent=2))


if __name__ == "__main__":
    main()

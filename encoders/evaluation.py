"""Phase 4 evaluation: probe every encoder and roll out the JEPA predictor."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from encoders.frames import (
    EpisodeRef,
    FrameCache,
    normalize_action_blocks,
    split_holdout_episodes,
)
from encoders.jepa import JEPA, SupervisedEncoder
from encoders.probes import (
    FLAG_TARGETS,
    REGRESSION_TARGETS,
    MLPProbe,
    ProbeLabels,
    auroc,
    balanced_accuracy,
    episode_bootstrap,
    fit_logistic,
    fit_ridge_targets,
    labels_for_episodes,
    per_episode_totals,
    per_sample_error,
)
from encoders.training import SUPERVISED_TARGETS, JEPATrainingConfig, supervised_targets
from eval.schema import MetricEstimate, RolloutCurve, TrainingCurvePoint

ROLLOUT_HORIZON = 5
ROLLOUT_TARGETS = ("cube_xyz", "tcp_xyz")
PHASES = ("free_motion", "held", "after_release")
TIMESTEP_SECONDS = 0.2


@dataclass
class SplitData:
    """Features and labels of one split, for one encoder seed."""

    features: dict[str, NDArray[np.float32]]
    labels: ProbeLabels


@dataclass
class ProbeOutcome:
    """Per-seed, per-episode error totals for every metric of one probe row."""

    sums: dict[str, list[NDArray[np.float64]]] = field(default_factory=dict)
    counts: dict[str, list[NDArray[np.float64]]] = field(default_factory=dict)
    flags: dict[str, list[float]] = field(default_factory=dict)

    def add_error(
        self, name: str, sums: NDArray[np.float64], counts: NDArray[np.float64]
    ) -> None:
        """Append one seed's per-episode error totals."""
        self.sums.setdefault(name, []).append(sums)
        self.counts.setdefault(name, []).append(counts)

    def add_flag(self, name: str, value: float) -> None:
        """Append one seed's flag metric."""
        self.flags.setdefault(name, []).append(value)


def load_jepa_run(
    run_dir: Path, device: torch.device
) -> tuple[JEPA, JEPATrainingConfig, dict]:
    """Load a trained JEPA from its run directory."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = JEPATrainingConfig.model_validate(record["config"])
    if config.mode != "jepa":
        raise ValueError(f"{run_dir} is not a JEPA run")
    model = JEPA(config.model)
    payload = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), config, record


@dataclass(frozen=True)
class SupervisedRun:
    """A trained supervised ceiling and the target scaling it was trained with."""

    model: SupervisedEncoder
    target_mean: NDArray[np.float64]
    target_std: NDArray[np.float64]

    def decoders(self) -> tuple[dict[str, Predictor], dict[str, Predictor]]:
        """Return per-target readers of the model's own output (the ``output`` readout).

        Continuous outputs are standardized targets in ``SUPERVISED_TARGETS``
        order and are mapped back to target units; the last columns are the
        flag logits.
        """
        widths = {
            "cube_xyz": 3,
            "cube_yaw": 2,
            "bowl_xy": 2,
            "tcp_xyz": 3,
            "gripper": 1,
        }
        columns, start = {}, 0
        for name in SUPERVISED_TARGETS:
            columns[name] = slice(start, start + widths[name])
            start += widths[name]
        if start != len(self.target_mean):
            raise ValueError("Supervised target layout does not match its statistics")

        def regressor(name: str) -> Predictor:
            def predict(outputs: NDArray[np.floating]) -> NDArray[np.float64]:
                values = np.asarray(outputs, dtype=np.float64)[:, :start]
                return (values * self.target_std + self.target_mean)[:, columns[name]]

            return predict

        def classifier(index: int) -> Predictor:
            return lambda outputs: np.asarray(outputs, dtype=np.float64)[
                :, start + index
            ]

        return (
            {name: regressor(name) for name in SUPERVISED_TARGETS},
            {name: classifier(index) for index, name in enumerate(FLAG_TARGETS)},
        )


def load_supervised_run(
    run_dir: Path,
    device: torch.device,
    *,
    train_caches: Sequence[FrameCache],
    train_datasets: Sequence[str],
) -> SupervisedRun:
    """Load the supervised ceiling and rebuild its target standardization.

    The statistics come from the run's own training episodes (the same seeded
    holdout split), so its outputs are decoded exactly as they were trained.
    """
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = JEPATrainingConfig.model_validate(record["config"])
    if config.mode != "supervised":
        raise ValueError(f"{run_dir} is not a supervised run")
    if list(train_datasets) != list(config.train_datasets):
        raise ValueError(
            f"Evaluation train datasets {list(train_datasets)} differ from the "
            f"supervised run's {config.train_datasets}"
        )
    payload = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    outputs = int(payload["model"]["head.weight"].shape[0])
    model = SupervisedEncoder(config.model, outputs)
    model.load_state_dict(payload["model"])
    train_episodes, _ = split_holdout_episodes(
        train_caches, fraction=config.holdout_fraction, seed=config.seed
    )
    continuous, _ = supervised_targets(
        labels_for_episodes(train_caches, train_episodes)
    )
    if outputs != continuous.shape[1] + len(FLAG_TARGETS):
        raise ValueError("Supervised head size does not match its targets")
    return SupervisedRun(
        model.to(device).eval(),
        continuous.mean(axis=0),
        np.maximum(continuous.std(axis=0), 1e-6),
    )


def training_curve(record: dict) -> list[TrainingCurvePoint]:
    """Extract the collapse and probe diagnostics logged during a JEPA run."""
    history = record.get("result", {}).get("history", [])
    return [
        TrainingCurvePoint(
            step=item["step"],
            probe_score=item["probe_score"],
            effective_rank=item.get("effective_rank"),
            latent_std=item.get("latent_std"),
            persistence_ratio=item.get("persistence_ratio"),
            gate_magnitude=item.get("gate_magnitude"),
        )
        for item in history
    ]


def probe_seed(
    outcomes: Mapping[str, ProbeOutcome],
    readout: str,
    probe: str,
    train: SplitData,
    evaluations: Mapping[str, SplitData],
    *,
    episodes: Mapping[str, int],
    seed: int,
) -> None:
    """Fit every target's probe once on ``train`` and score each evaluation split."""
    x_train = train.features[readout]
    groups = train.labels.episode
    targets = {
        name: train.labels.regression_target(name) for name in REGRESSION_TARGETS
    }
    if probe == "linear":
        ridge = fit_ridge_targets(x_train, targets, groups)
        regressors = {name: ridge[name].predict for name in REGRESSION_TARGETS}
        classifiers = {
            name: fit_logistic(x_train, getattr(train.labels, name), groups).scores
            for name in FLAG_TARGETS
        }
    else:
        regressors = {
            name: MLPProbe(
                x_train, targets[name], groups, classification=False, seed=seed
            ).predict
            for name in REGRESSION_TARGETS
        }
        classifiers = {
            name: _first_column(
                MLPProbe(
                    x_train,
                    getattr(train.labels, name),
                    groups,
                    classification=True,
                    seed=seed,
                ).predict
            )
            for name in FLAG_TARGETS
        }
    score_predictions(
        outcomes, readout, regressors, classifiers, evaluations, episodes=episodes
    )


Predictor = Callable[[NDArray[np.floating]], NDArray[np.float64]]


def score_predictions(
    outcomes: Mapping[str, ProbeOutcome],
    readout: str,
    regressors: Mapping[str, Predictor],
    classifiers: Mapping[str, Predictor],
    evaluations: Mapping[str, SplitData],
    *,
    episodes: Mapping[str, int],
) -> None:
    """Score per-target predictions from one readout on every evaluation split."""
    for split, data in evaluations.items():
        outcome = outcomes[split]
        features = data.features[readout]
        labels = data.labels
        count = episodes[split]
        for target, predict in regressors.items():
            errors = per_sample_error(target, predict(features), labels)
            unit = (
                "deg" if target == "cube_yaw" else "mm" if target == "gripper" else "cm"
            )
            outcome.add_error(
                f"{target}_{unit}", *per_episode_totals(errors, labels.episode, count)
            )
            if target == "cube_xyz":
                for name, mask in (("resting", ~labels.held), ("held", labels.held)):
                    outcome.add_error(
                        f"cube_xyz_{name}_cm",
                        *_masked_totals(errors, mask, labels.episode, count),
                    )
        for target, score in classifiers.items():
            truth = getattr(labels, target)
            scores = score(features)
            outcome.add_flag(
                f"{target}_balanced_accuracy", balanced_accuracy(truth, scores > 0)
            )
            outcome.add_flag(f"{target}_auroc", auroc(truth, scores))


def _first_column(
    predict: Callable[[NDArray[np.floating]], NDArray[np.float64]],
) -> Callable[[NDArray[np.floating]], NDArray[np.float64]]:
    """Return a scorer that keeps the single logit column of an MLP classifier."""
    return lambda features: predict(features)[:, 0]


def _masked_totals(
    errors: NDArray[np.float64],
    mask: NDArray[np.bool_],
    episode: NDArray[np.int64],
    episodes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-episode error totals over only the masked samples."""
    return per_episode_totals(errors[mask], episode[mask], episodes)


def summarize(
    outcome: ProbeOutcome, *, resamples: int, seed: int = 0
) -> dict[str, MetricEstimate]:
    """Pool seeds and episodes into point estimates with episode-bootstrap CIs."""
    metrics: dict[str, MetricEstimate] = {}
    for name, sums in outcome.sums.items():
        top = np.stack(sums)
        bottom = np.stack(outcome.counts[name])
        keep = bottom.sum(axis=0) > 0
        top, bottom = top[:, keep], bottom[:, keep]
        if not keep.any():
            continue
        low, high = episode_bootstrap(top, bottom, resamples=resamples, seed=seed)
        metrics[name] = MetricEstimate(
            value=float(top.sum() / bottom.sum()),
            ci_low=low,
            ci_high=high,
            per_seed=[
                float(t.sum() / b.sum()) for t, b in zip(top, bottom, strict=True)
            ],
        )
    for name, values in outcome.flags.items():
        metrics[name] = MetricEstimate(
            value=float(np.mean(values)), per_seed=list(values)
        )
    return metrics


def phases(labels: ProbeLabels) -> NDArray[np.str_]:
    """Label each timestep free motion, held, or after release (held earlier)."""
    result = np.empty(len(labels), dtype=object)
    for episode in np.unique(labels.episode):
        rows = np.flatnonzero(labels.episode == episode)
        held = labels.held[rows]
        was_held = np.maximum.accumulate(held)
        result[rows] = np.where(
            held, "held", np.where(was_held, "after_release", "free_motion")
        )
    return result.astype(str)


@dataclass
class RolloutTotals:
    """Accumulate per-seed, per-episode rollout error totals."""

    errors: dict[
        tuple[str, str, str, int], list[tuple[NDArray[np.float64], NDArray[np.float64]]]
    ] = field(default_factory=dict)
    latent: dict[int, list[tuple[NDArray[np.float64], NDArray[np.float64]]]] = field(
        default_factory=dict
    )


def rollout_seed(
    totals: RolloutTotals,
    model: JEPA,
    train_latents: NDArray[np.float32],
    train_labels: ProbeLabels,
    test_latents: NDArray[np.float32],
    test_labels: ProbeLabels,
    test_caches: Sequence[FrameCache],
    test_episodes: Sequence[EpisodeRef],
    *,
    device: torch.device,
) -> int:
    """Roll the predictor out on every test window and add decoded errors."""
    history = model.config.history
    length = history + ROLLOUT_HORIZON
    probes = fit_ridge_targets(
        train_latents,
        {target: train_labels.regression_target(target) for target in ROLLOUT_TARGETS},
        train_labels.episode,
    )
    offsets = np.cumsum([0] + [reference.episode.length for reference in test_episodes])
    windows, rows, blocks = [], [], []
    for position, reference in enumerate(test_episodes):
        cache = test_caches[reference.cache_index]
        for start in range(reference.episode.length - length + 1):
            windows.append(offsets[position] + start)
            row = reference.episode.start + start
            rows.append(row)
            blocks.append(cache.action_blocks[row : row + length - 1])
    if not windows:
        raise ValueError("No test episode is long enough for a rollout window")
    starts = np.asarray(windows)
    actions = normalize_action_blocks(np.stack(blocks)).to(device)
    latents = torch.as_tensor(test_latents, device=device)
    context = latents[starts[:, None] + np.arange(history)]
    predicted = []
    for batch in range(0, len(starts), 512):
        predicted.append(
            model.rollout(
                context[batch : batch + 512], actions[batch : batch + 512]
            ).cpu()
        )
    predictions = torch.cat(predicted).numpy()
    phase = phases(test_labels)
    episodes = len(test_episodes)
    last = starts + history - 1
    for horizon in range(1, ROLLOUT_HORIZON + 1):
        target_index = last + horizon
        truth_labels = test_labels.subset(target_index)
        group = truth_labels.episode
        sources = {
            "predictor": predictions[:, horizon - 1],
            "persistence": test_latents[last],
            "probe_floor": test_latents[target_index],
        }
        for model_name, values in sources.items():
            for target in ROLLOUT_TARGETS:
                errors = per_sample_error(
                    target, probes[target].predict(values), truth_labels
                )
                for phase_name in ("all", *PHASES):
                    mask = (
                        np.ones(len(errors), bool)
                        if phase_name == "all"
                        else phase[target_index] == phase_name
                    )
                    key = (model_name, target, phase_name, horizon)
                    totals.errors.setdefault(key, []).append(
                        _masked_totals(errors, mask, group, episodes)
                    )
        squared = np.sum(
            (predictions[:, horizon - 1] - test_latents[target_index]) ** 2, axis=1
        )
        baseline = np.sum(
            (test_latents[last] - test_latents[target_index]) ** 2, axis=1
        )
        totals.latent.setdefault(horizon, []).append(
            (
                np.bincount(group, weights=squared, minlength=episodes),
                np.bincount(group, weights=baseline, minlength=episodes),
            )
        )
    return len(starts)


def summarize_rollouts(
    totals: RolloutTotals, *, resamples: int
) -> tuple[list[RolloutCurve], list[MetricEstimate], bool]:
    """Turn rollout totals into curves, latent ratios, and the H4 verdict."""

    def estimate(
        parts: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> MetricEstimate:
        top = np.stack([part[0] for part in parts])
        bottom = np.stack([part[1] for part in parts])
        keep = bottom.sum(axis=0) > 0
        if not keep.any():
            return MetricEstimate(value=float("nan"))
        top, bottom = top[:, keep], bottom[:, keep]
        low, high = episode_bootstrap(top, bottom, resamples=resamples)
        return MetricEstimate(
            value=float(top.sum() / bottom.sum()),
            ci_low=low,
            ci_high=high,
            per_seed=[
                float(t.sum() / max(b.sum(), 1e-12))
                for t, b in zip(top, bottom, strict=True)
            ],
        )

    curves = []
    for model_name in ("predictor", "persistence", "probe_floor"):
        for target in ROLLOUT_TARGETS:
            for phase_name in ("all", *PHASES):
                curves.append(
                    RolloutCurve(
                        model=model_name,
                        target=target,
                        phase=phase_name,
                        errors_cm=[
                            estimate(
                                totals.errors[(model_name, target, phase_name, horizon)]
                            )
                            for horizon in range(1, ROLLOUT_HORIZON + 1)
                        ],
                    )
                )
    latent = [
        estimate(totals.latent[horizon]) for horizon in range(1, ROLLOUT_HORIZON + 1)
    ]

    def curve(model_name: str, target: str, phase_name: str) -> list[MetricEstimate]:
        return next(
            item.errors_cm
            for item in curves
            if (item.model, item.target, item.phase) == (model_name, target, phase_name)
        )

    h4 = all(
        curve("predictor", target, phase_name)[horizon - 1].value
        < curve("persistence", target, phase_name)[horizon - 1].value
        for target, phase_name in (("tcp_xyz", "all"), ("cube_xyz", "held"))
        for horizon in range(2, ROLLOUT_HORIZON + 1)
    )
    return curves, latent, h4

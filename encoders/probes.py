"""Probe labels, grouped-CV linear and MLP probes, metrics, and episode bootstrap."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from data.generate import OBJECT_POSE_COMPONENTS, OBJECT_POSE_NAMES
from encoders.frames import EpisodeRef, FrameCache
from envs.pick_place_env import cube_within_bowl

# The cube rests with its center 0.02 m above the table; held means 1 cm above that.
HELD_MIN_HEIGHT = 0.03
HELD_MAX_TCP_DISTANCE = 0.03
RIDGE_ALPHAS = tuple(float(value) for value in np.logspace(-2, 5, 15))
LOGISTIC_PENALTIES = (1e-3, 1e-2, 1e-1)
LOGISTIC_MAX_SAMPLES = 20_000
CV_FOLDS = 5
WIDE_FEATURES = 2048

_POSE_WIDTH = len(OBJECT_POSE_COMPONENTS)
_TCP_SLICE = slice(8, 11)
_GRIPPER_INDEX = 7

REGRESSION_TARGETS = ("cube_xyz", "cube_yaw", "bowl_xy", "tcp_xyz", "gripper")
FLAG_TARGETS = ("held", "in_bowl")


@dataclass(frozen=True)
class ProbeLabels:
    """Hold privileged per-timestep labels and their episode groups."""

    cube_xyz: NDArray[np.float64]
    cube_yaw: NDArray[np.float64]
    bowl_xy: NDArray[np.float64]
    tcp_xyz: NDArray[np.float64]
    gripper: NDArray[np.float64]
    held: NDArray[np.bool_]
    in_bowl: NDArray[np.bool_]
    episode: NDArray[np.int64]

    def __len__(self) -> int:
        """Return the number of labelled timesteps."""
        return len(self.episode)

    def regression_target(self, name: str) -> NDArray[np.float64]:
        """Return a 2-D regression target; yaw becomes (sin 4θ, cos 4θ)."""
        if name == "cube_yaw":
            return np.stack((np.sin(4 * self.cube_yaw), np.cos(4 * self.cube_yaw)), 1)
        value = getattr(self, name)
        return value.reshape(len(self), -1)

    def subset(self, indices: NDArray[np.int64]) -> ProbeLabels:
        """Return the labels at ``indices``."""
        return ProbeLabels(
            **{name: getattr(self, name)[indices] for name in self.__dataclass_fields__}
        )


def _pose_slice(name: str) -> slice:
    """Return the environment-state columns of one object's pose."""
    start = OBJECT_POSE_NAMES.index(name) * _POSE_WIDTH
    return slice(start, start + _POSE_WIDTH)


def _yaw_from_quaternions(quaternions_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return yaw (rotation about world z) for w, x, y, z quaternions."""
    w, x, y, z = quaternions_wxyz.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def episode_labels(cache: FrameCache, episode_position: int, group: int) -> ProbeLabels:
    """Derive the labels of one cached episode from logged state, never images."""
    episode = cache.episodes[episode_position]
    rows = slice(episode.start, episode.start + episode.length)
    environment = cache.environment_state[rows].astype(np.float64)
    state = cache.state[rows].astype(np.float64)
    cube_pose = environment[:, _pose_slice(episode.target_object)]
    bowl_pose = environment[:, _pose_slice("bowl")]
    cube_xyz = cube_pose[:, :3]
    tcp_xyz = state[:, _TCP_SLICE]
    held = (cube_xyz[:, 2] > HELD_MIN_HEIGHT) & (
        np.linalg.norm(cube_xyz - tcp_xyz, axis=1) < HELD_MAX_TCP_DISTANCE
    )
    in_bowl = np.array(
        [
            cube_within_bowl(cube, bowl)
            for cube, bowl in zip(cube_xyz, bowl_pose[:, :3], strict=True)
        ],
        dtype=bool,
    )
    return ProbeLabels(
        cube_xyz=cube_xyz,
        cube_yaw=_yaw_from_quaternions(cube_pose[:, 3:7]),
        bowl_xy=bowl_pose[:, :2],
        tcp_xyz=tcp_xyz,
        gripper=state[:, _GRIPPER_INDEX],
        held=held,
        in_bowl=in_bowl,
        episode=np.full(episode.length, group, dtype=np.int64),
    )


def labels_for_episodes(
    caches: Sequence[FrameCache], episodes: Sequence[EpisodeRef]
) -> ProbeLabels:
    """Concatenate labels in the order of ``episodes``; groups are list positions."""
    positions = {
        (cache_index, episode.start): position
        for cache_index, cache in enumerate(caches)
        for position, episode in enumerate(cache.episodes)
    }
    parts = [
        episode_labels(
            caches[reference.cache_index],
            positions[(reference.cache_index, reference.episode.start)],
            group,
        )
        for group, reference in enumerate(episodes)
    ]
    return ProbeLabels(
        **{
            name: np.concatenate([getattr(part, name) for part in parts])
            for name in ProbeLabels.__dataclass_fields__
        }
    )


def grouped_folds(
    groups: NDArray[np.int64], folds: int = CV_FOLDS, seed: int = 0
) -> NDArray[np.int64]:
    """Assign each sample a fold so that no group (episode) spans two folds."""
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f"Need at least {folds} groups for {folds}-fold CV")
    order = np.random.default_rng(seed).permutation(unique)
    fold_of_group = {int(group): index % folds for index, group in enumerate(order)}
    return np.array([fold_of_group[int(group)] for group in groups], dtype=np.int64)


class Standardizer:
    """Standardize features with statistics from the probe-training set."""

    def __init__(self, features: NDArray[np.floating]) -> None:
        """Store per-feature mean and standard deviation (floored)."""
        values = np.asarray(features, dtype=np.float32)
        self.mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = np.maximum(values.std(axis=0, dtype=np.float64), 1e-6).astype(
            np.float32
        )

    def __call__(self, features: NDArray[np.floating]) -> NDArray[np.float32]:
        """Return standardized features in float32 (wide readouts are large)."""
        values = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        return values.astype(np.float32, copy=False)


@dataclass(frozen=True)
class RidgeProbe:
    """A fitted multi-output ridge regression on standardized features."""

    standardizer: Standardizer
    coefficients: NDArray[np.float64]
    intercept: NDArray[np.float64]
    alpha: float

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Predict targets for raw features."""
        return (
            self.standardizer(features).astype(np.float64) @ self.coefficients
            + self.intercept
        )


def _eigh(gram: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigendecompose a Gram matrix; wide (grid) readouts use float32 for speed.

    Float32 halves the cost of the 6,144- and 12,288-wide grid readouts; the
    ridge penalty keeps the solution insensitive to its rounding.
    """
    if len(gram) <= WIDE_FEATURES:
        return np.linalg.eigh(gram)
    eigenvalues, eigenvectors = torch.linalg.eigh(
        torch.from_numpy(gram.astype(np.float32))
    )
    return eigenvalues.double().numpy(), eigenvectors.double().numpy()


def _ridge_solutions(
    gram: NDArray[np.float64],
    cross: NDArray[np.float64],
    alphas: Sequence[float],
) -> list[NDArray[np.float64]]:
    """Solve (G + aI) b = c for every alpha with one eigendecomposition."""
    eigenvalues, eigenvectors = _eigh(gram)
    rotated = eigenvectors.T @ cross
    return [
        eigenvectors @ (rotated / (eigenvalues + alpha)[:, None]) for alpha in alphas
    ]


def fit_ridge(
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    groups: NDArray[np.int64],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    folds: int = CV_FOLDS,
) -> RidgeProbe:
    """Fit one ridge probe, choosing the penalty by episode-grouped CV."""
    return fit_ridge_targets(
        features, {"target": targets}, groups, alphas=alphas, folds=folds
    )["target"]


def fit_ridge_targets(
    features: NDArray[np.floating],
    targets: Mapping[str, NDArray[np.floating]],
    groups: NDArray[np.int64],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    folds: int = CV_FOLDS,
) -> dict[str, RidgeProbe]:
    """Fit ridge probes for several targets, choosing each penalty by grouped CV.

    Each target's penalty minimizes its validation MSE summed over standardized
    columns. All targets share one eigendecomposition per fold, and per-fold
    Gram matrices are formed by subtraction from the total, so wide (grid)
    features cost one pass over the data.
    """
    standardizer = Standardizer(features)
    x = standardizer(features)
    blocks = {
        name: np.asarray(value, dtype=np.float64).reshape(len(x), -1)
        for name, value in targets.items()
    }
    columns: dict[str, slice] = {}
    offset = 0
    for name, block in blocks.items():
        columns[name] = slice(offset, offset + block.shape[1])
        offset += block.shape[1]
    y = np.concatenate(list(blocks.values()), axis=1)
    y_scale = np.maximum(y.std(axis=0), 1e-9)
    fold_ids = grouped_folds(groups, folds)
    fold_stats = []
    for fold in range(folds):
        mask = fold_ids == fold
        part = x[mask]
        fold_stats.append(
            (
                (part.T @ part).astype(np.float64),
                part.T.astype(np.float64) @ y[mask],
                part.sum(0, dtype=np.float64),
                y[mask].sum(0),
                int(mask.sum()),
            )
        )
    total_gram = sum(stat[0] for stat in fold_stats)
    total_cross = sum(stat[1] for stat in fold_stats)
    total_x = sum(stat[2] for stat in fold_stats)
    total_y = sum(stat[3] for stat in fold_stats)
    errors = np.zeros((len(alphas), y.shape[1]))
    for fold, (gram_f, cross_f, x_f, y_f, count_f) in enumerate(fold_stats):
        count = len(x) - count_f
        mean_x = (total_x - x_f) / count
        mean_y = (total_y - y_f) / count
        gram = total_gram - gram_f - count * np.outer(mean_x, mean_x)
        cross = total_cross - cross_f - count * np.outer(mean_x, mean_y)
        mask = fold_ids == fold
        centered = x[mask] - mean_x
        for index, coefficients in enumerate(_ridge_solutions(gram, cross, alphas)):
            prediction = centered @ coefficients + mean_y
            errors[index] += np.sum(((prediction - y[mask]) / y_scale) ** 2, axis=0)
    mean_x = total_x / len(x)
    mean_y = total_y / len(x)
    gram = total_gram - len(x) * np.outer(mean_x, mean_x)
    cross = total_cross - len(x) * np.outer(mean_x, mean_y)
    chosen = {
        name: float(alphas[int(np.argmin(errors[:, columns[name]].sum(axis=1)))])
        for name in blocks
    }
    eigenvalues, eigenvectors = _eigh(gram)
    rotated = eigenvectors.T @ cross
    probes = {}
    for name, alpha in chosen.items():
        cols = columns[name]
        coefficients = eigenvectors @ (
            rotated[:, cols] / (eigenvalues + alpha)[:, None]
        )
        probes[name] = RidgeProbe(
            standardizer, coefficients, mean_y[cols] - mean_x @ coefficients, alpha
        )
    return probes


@dataclass(frozen=True)
class LogisticProbe:
    """A fitted, class-balanced logistic regression on standardized features."""

    standardizer: Standardizer
    weight: NDArray[np.float64]
    bias: float
    penalty: float

    def scores(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return decision scores (logits) for raw features."""
        return self.standardizer(features).astype(np.float64) @ self.weight + self.bias


def _fit_logistic_weights(
    x: NDArray[np.floating],
    y: NDArray[np.bool_],
    penalty: float,
    device: torch.device,
) -> tuple[NDArray[np.float64], float]:
    """Fit balanced, L2-penalized logistic regression with full-batch L-BFGS."""
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    sample_weight = torch.where(labels > 0.5, 0.5 / positives, 0.5 / negatives)
    weight = torch.zeros(features.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = features @ weight + bias
        loss = (
            nn.functional.binary_cross_entropy_with_logits(
                logits, labels, weight=sample_weight, reduction="sum"
            )
            + 0.5 * penalty * weight.square().sum()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach().cpu().double().numpy(), float(bias.detach().cpu())


def fit_logistic(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_],
    groups: NDArray[np.int64],
    *,
    penalties: Sequence[float] = LOGISTIC_PENALTIES,
    folds: int = CV_FOLDS,
    device: torch.device | None = None,
    max_samples: int = LOGISTIC_MAX_SAMPLES,
    seed: int = 0,
) -> LogisticProbe:
    """Fit a balanced logistic probe, choosing the penalty by grouped CV.

    Full-batch L-BFGS costs a pass over the features per iteration, so at most
    ``max_samples`` training timesteps, chosen at random, are used.
    """
    device = device or torch.device("cpu")
    if len(features) > max_samples:
        keep = np.sort(
            np.random.default_rng(seed).choice(
                len(features), max_samples, replace=False
            )
        )
        features, labels, groups = (
            features[keep],
            np.asarray(labels)[keep],
            groups[keep],
        )
    standardizer = Standardizer(features)
    x = standardizer(features)
    y = np.asarray(labels, dtype=bool)
    if y.all() or not y.any():
        raise ValueError("Logistic probes need both classes in the training labels")
    fold_ids = grouped_folds(groups, folds)
    accuracies = np.zeros(len(penalties))
    for fold in range(folds):
        train, test = fold_ids != fold, fold_ids == fold
        if y[train].all() or not y[train].any():
            continue
        for index, penalty in enumerate(penalties):
            weight, bias = _fit_logistic_weights(x[train], y[train], penalty, device)
            accuracies[index] += balanced_accuracy(y[test], x[test] @ weight + bias > 0)
    penalty = float(penalties[int(np.argmax(accuracies))])
    weight, bias = _fit_logistic_weights(x, y, penalty, device)
    return LogisticProbe(standardizer, weight, bias, penalty)


class MLPProbe:
    """A one-hidden-layer MLP probe with early stopping on a grouped fold."""

    def __init__(
        self,
        features: NDArray[np.floating],
        targets: NDArray[np.floating],
        groups: NDArray[np.int64],
        *,
        classification: bool,
        hidden: int = 256,
        seed: int = 0,
        max_epochs: int = 200,
        patience: int = 10,
        batch_size: int = 512,
    ) -> None:
        """Train the MLP, keeping the weights with the best held-out-fold loss."""
        self.standardizer = Standardizer(features)
        x = torch.as_tensor(self.standardizer(features), dtype=torch.float32)
        y = torch.as_tensor(
            np.asarray(targets, dtype=np.float64).reshape(len(x), -1),
            dtype=torch.float32,
        )
        self.classification = classification
        if classification:
            self.target_mean = np.zeros(1)
            self.target_std = np.ones(1)
            positive = float(y.mean())
            self._positive_weight = torch.tensor((1.0 - positive) / max(positive, 1e-6))
        else:
            self.target_mean = y.mean(0).double().numpy()
            self.target_std = np.maximum(y.std(0).double().numpy(), 1e-9)
            y = (
                y - torch.as_tensor(self.target_mean, dtype=torch.float32)
            ) / torch.as_tensor(self.target_std, dtype=torch.float32)
        validation = torch.as_tensor(grouped_folds(groups, CV_FOLDS, seed) == 0)
        generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, y.shape[1])
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        train_x, train_y = x[~validation], y[~validation]
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(max_epochs):
            self.model.train()
            order = torch.randperm(len(train_x), generator=generator)
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                optimizer.zero_grad()
                self._loss(self.model(train_x[batch]), train_y[batch]).backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                loss = float(self._loss(self.model(x[validation]), y[validation]))
            if loss < best_loss - 1e-6:
                best_loss, stale = loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def _loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return balanced BCE for flags or MSE for standardized targets."""
        if self.classification:
            return nn.functional.binary_cross_entropy_with_logits(
                output, target, pos_weight=self._positive_weight
            )
        return nn.functional.mse_loss(output, target)

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return predictions in target units, or logits for flags."""
        with torch.no_grad():
            output = (
                self.model(
                    torch.as_tensor(self.standardizer(features), dtype=torch.float32)
                )
                .double()
                .numpy()
            )
        return output * self.target_std + self.target_mean


def per_sample_error(
    target: str, prediction: NDArray[np.floating], labels: ProbeLabels
) -> NDArray[np.float64]:
    """Return per-timestep errors: cm for positions, degrees for yaw, mm for gripper."""
    truth = labels.regression_target(target)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(truth.shape)
    if target == "cube_yaw":
        predicted_yaw = np.arctan2(prediction[:, 0], prediction[:, 1]) / 4.0
        difference = (predicted_yaw - labels.cube_yaw + np.pi / 4) % (
            np.pi / 2
        ) - np.pi / 4
        return np.degrees(np.abs(difference))
    if target == "gripper":
        return 1000.0 * np.abs(prediction[:, 0] - truth[:, 0])
    return 100.0 * np.linalg.norm(prediction - truth, axis=1)


def balanced_accuracy(truth: NDArray[np.bool_], predicted: NDArray[np.bool_]) -> float:
    """Return the mean of the true-positive and true-negative rates."""
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    rates = [
        np.mean(predicted[truth == value] == value)
        for value in (True, False)
        if np.any(truth == value)
    ]
    return float(np.mean(rates))


def auroc(truth: NDArray[np.bool_], scores: NDArray[np.floating]) -> float:
    """Return the area under the ROC curve via the rank-sum statistic (ties averaged)."""
    truth = np.asarray(truth, dtype=bool)
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = np.asarray(scores)[order]
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return float(
        (ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def episode_bootstrap(
    numerators: NDArray[np.floating],
    denominators: NDArray[np.floating],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile CI for sum(numerators) / sum(denominators), resampling episodes.

    Inputs are per-episode totals, shape (episodes,) or (seeds, episodes). With
    a seed axis, seeds are resampled first, then episodes within each, as in
    ``eval.stats.percentile_bootstrap``.
    """
    top = np.asarray(numerators, dtype=np.float64)
    bottom = np.asarray(denominators, dtype=np.float64)
    if top.shape != bottom.shape or top.ndim not in {1, 2} or top.size == 0:
        raise ValueError("numerators and denominators must share a 1-D or 2-D shape")
    if top.ndim == 1:
        top, bottom = top[None], bottom[None]
    random = np.random.default_rng(seed)
    seeds, episodes = top.shape
    estimates = np.empty(resamples)
    batch = max(1, min(resamples, 2_000_000 // top.size))
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        if seeds == 1:
            sampled_seeds = np.zeros((count, 1), dtype=np.int64)
        else:
            sampled_seeds = random.integers(0, seeds, size=(count, seeds))
        sampled_episodes = random.integers(0, episodes, size=(count, seeds, episodes))
        rows = sampled_seeds[:, :, None]
        estimates[start : start + count] = top[rows, sampled_episodes].sum(
            axis=(1, 2)
        ) / bottom[rows, sampled_episodes].sum(axis=(1, 2))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def per_episode_totals(
    values: NDArray[np.floating], episode: NDArray[np.int64], episodes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-episode sums of ``values`` and per-episode sample counts."""
    sums = np.bincount(episode, weights=values, minlength=episodes)
    counts = np.bincount(episode, minlength=episodes).astype(np.float64)
    return sums, counts

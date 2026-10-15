"""Confidence intervals and resampling statistics for evaluation results."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

type FloatInterval = tuple[float, float]


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> FloatInterval:
    """Return the two-sided Wilson score interval for a binomial success rate."""
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    z = (
        1.959963984540054
        if confidence == 0.95
        else _normal_quantile(0.5 + confidence / 2.0)
    )
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def percentile_bootstrap(
    values: ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> FloatInterval:
    """Return a seeded percentile interval for flat samples or seed-by-episode results."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim not in {1, 2} or samples.size == 0:
        raise ValueError("values must be a non-empty one- or two-dimensional array")
    if samples.ndim == 2 and any(dimension == 0 for dimension in samples.shape):
        raise ValueError("seed-by-episode values cannot contain an empty dimension")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must all be finite")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if isinstance(resamples, bool) or not isinstance(resamples, int):
        raise TypeError("resamples must be an integer")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    random = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    cells_per_resample = samples.size
    batch_size = max(1, min(resamples, 1_000_000 // cells_per_resample))
    written = 0
    while written < resamples:
        batch_count = min(batch_size, resamples - written)
        if samples.ndim == 1:
            indices = random.integers(
                0,
                samples.size,
                size=(batch_count, samples.size),
            )
            batch_means = np.mean(samples[indices], axis=1)
        else:
            training_seed_count, episode_count = samples.shape
            sampled_seeds = random.integers(
                0,
                training_seed_count,
                size=(batch_count, training_seed_count),
            )
            sampled_episodes = random.integers(
                0,
                episode_count,
                size=(batch_count, training_seed_count, episode_count),
            )
            batch_means = np.mean(
                samples[sampled_seeds[:, :, np.newaxis], sampled_episodes],
                axis=(1, 2),
            )
        bootstrap_means[written : written + batch_count] = batch_means
        written += batch_count

    tail_probability = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )
    return float(lower), float(upper)


def _normal_quantile(probability: float) -> float:
    """Return the standard-normal quantile without an additional dependency."""
    return float(ndtri(probability))

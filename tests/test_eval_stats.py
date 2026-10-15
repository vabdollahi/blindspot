import numpy as np
import pytest

from eval.stats import percentile_bootstrap, wilson_interval


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (45, 50, (0.7864, 0.9565)),
        (25, 50, (0.3664, 0.6336)),
    ],
)
def test_wilson_interval_matches_hand_computed_values(
    successes: int,
    trials: int,
    expected: tuple[float, float],
) -> None:
    """Match the roadmap's two hand-computed Wilson examples."""
    interval = wilson_interval(successes, trials)

    assert interval == pytest.approx(expected, abs=0.0002)


def test_wilson_interval_handles_all_successes_and_failures() -> None:
    """Keep endpoint intervals bounded by valid probabilities."""
    all_success = wilson_interval(10, 10)
    no_success = wilson_interval(0, 10)

    assert 0.0 < all_success[0] < 1.0
    assert all_success[1] == pytest.approx(1.0)
    assert no_success[0] == 0.0
    assert 0.0 < no_success[1] < 1.0


def test_percentile_bootstrap_is_seeded_and_covers_the_sample_mean() -> None:
    """Reproduce a percentile interval and include its observed sample mean."""
    values = np.array([0.0, 0.0, 1.0, 1.0])

    first = percentile_bootstrap(values, resamples=2_000, seed=13)
    second = percentile_bootstrap(values, resamples=2_000, seed=13)

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_percentile_bootstrap_resamples_training_seeds_and_episodes() -> None:
    """Preserve between-seed variation when bootstrapping a seed-by-episode matrix."""
    values = np.vstack((np.zeros(20), np.ones(20)))

    first = percentile_bootstrap(values, resamples=2_000, seed=21)
    second = percentile_bootstrap(values, resamples=2_000, seed=21)
    flat_interval = percentile_bootstrap(values.reshape(-1), resamples=2_000, seed=21)

    assert first == second
    assert first[0] == 0.0
    assert first[1] == 1.0
    assert first[1] - first[0] > flat_interval[1] - flat_interval[0]


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 0), (-1, 3), (4, 3)],
)
def test_wilson_interval_rejects_invalid_counts(successes: int, trials: int) -> None:
    """Reject impossible binomial count combinations."""
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)

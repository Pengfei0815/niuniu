"""Tests for dining-time distributions."""

import numpy as np

from queue_model.distributions import TruncatedLognormalDiningTime, TruncatedWeibullDiningTime


def test_sample_never_exceeds_max_time() -> None:
    """Samples from truncated models should never exceed max_time."""
    weibull = TruncatedWeibullDiningTime(shape=3.0, scale=95.0, max_time=120.0)
    lognormal = TruncatedLognormalDiningTime(mu=4.45, sigma=0.35, max_time=120.0)

    assert np.max(weibull.sample(5000, random_state=1)) <= 120.0
    assert np.max(lognormal.sample(5000, random_state=2)) <= 120.0


def test_survival_is_monotone_decreasing() -> None:
    """Survival values should be non-increasing in time."""
    distribution = TruncatedWeibullDiningTime(shape=2.5, scale=90.0, max_time=120.0)
    grid = np.linspace(0.0, 130.0, 200)
    survival = distribution.survival(grid)
    assert np.all(np.diff(survival) <= 1e-12)


"""Tests for release-time simulation."""

import numpy as np

from queue_model.distributions import TruncatedWeibullDiningTime
from queue_model.simulation import simulate_all_release_times


def test_simulated_release_times_are_sorted() -> None:
    """The merged release process should be returned in chronological order."""
    distribution = TruncatedWeibullDiningTime(shape=3.0, scale=95.0, max_time=120.0)
    releases = simulate_all_release_times(
        dining_time_distribution=distribution,
        table_ages=np.full(5, 30.0),
        n_releases=25,
        random_state=123,
    )
    assert len(releases) == 25
    assert np.all(np.diff(releases) >= 0)


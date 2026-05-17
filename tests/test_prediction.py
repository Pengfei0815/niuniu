"""Tests for high-level prediction behavior."""

import numpy as np

from queue_model.distributions import TruncatedWeibullDiningTime
from queue_model.prediction import predict_entry_time


def test_larger_queue_ahead_has_larger_mean_wait() -> None:
    """A larger number of tables ahead should generally increase mean waiting time."""
    distribution = TruncatedWeibullDiningTime(shape=3.0, scale=95.0, max_time=120.0)
    table_ages = np.linspace(5.0, 95.0, 20)

    short_queue = predict_entry_time(
        current_time=120.0,
        queue_ahead=2,
        n_tables=20,
        dining_time_distribution=distribution,
        table_ages=table_ages,
        n_simulations=1500,
        random_state=11,
    )
    long_queue = predict_entry_time(
        current_time=120.0,
        queue_ahead=15,
        n_tables=20,
        dining_time_distribution=distribution,
        table_ages=table_ages,
        n_simulations=1500,
        random_state=11,
    )

    assert float(long_queue["mean_wait"]) > float(short_queue["mean_wait"])


"""Monte Carlo simulation of table release processes."""

from __future__ import annotations

import heapq
from typing import Sequence

import numpy as np

from queue_model.distributions import DiningTimeDistribution
from queue_model.utils import RandomStateLike, get_rng


def simulate_table_release_times(
    dining_time_distribution: DiningTimeDistribution,
    age: float,
    n_releases: int,
    random_state: RandomStateLike = None,
) -> np.ndarray:
    """Simulate future release times for one table from the current moment.

    The first release uses the conditional residual lifetime given current
    table age. Later releases add fresh iid dining times.
    """
    if n_releases <= 0:
        return np.array([], dtype=float)
    rng = get_rng(random_state)
    residual = dining_time_distribution.sample_residual(age=age, size=1, random_state=rng)[0]
    if n_releases == 1:
        return np.array([residual], dtype=float)

    future_durations = dining_time_distribution.sample(n_releases - 1, random_state=rng)
    increments = np.concatenate([[residual], future_durations])
    return np.cumsum(increments)


def simulate_all_release_times(
    dining_time_distribution: DiningTimeDistribution,
    table_ages: Sequence[float],
    n_releases: int,
    random_state: RandomStateLike = None,
) -> np.ndarray:
    """Simulate and sort the next ``n_releases`` release events across all tables."""
    if n_releases <= 0:
        return np.array([], dtype=float)
    if len(table_ages) == 0:
        raise ValueError("table_ages must contain at least one table.")

    rng = get_rng(random_state)
    heap: list[tuple[float, int]] = []
    for table_id, age in enumerate(table_ages):
        residual = dining_time_distribution.sample_residual(age=float(age), size=1, random_state=rng)[0]
        heapq.heappush(heap, (float(residual), table_id))

    releases = np.empty(n_releases, dtype=float)
    for idx in range(n_releases):
        release_time, table_id = heapq.heappop(heap)
        releases[idx] = release_time
        next_duration = dining_time_distribution.sample(1, random_state=rng)[0]
        heapq.heappush(heap, (float(release_time + next_duration), table_id))
    return releases


def monte_carlo_waiting_time(
    queue_ahead: int,
    table_ages: Sequence[float],
    dining_time_distribution: DiningTimeDistribution,
    n_simulations: int = 10000,
    random_state: RandomStateLike = None,
) -> np.ndarray:
    """Sample waiting time W_k(c)=D_(k+1)(c) by Monte Carlo simulation."""
    if queue_ahead < 0:
        raise ValueError("queue_ahead must be non-negative.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    rng = get_rng(random_state)
    target_release = queue_ahead + 1
    samples = np.empty(n_simulations, dtype=float)
    for sim in range(n_simulations):
        releases = simulate_all_release_times(
            dining_time_distribution=dining_time_distribution,
            table_ages=table_ages,
            n_releases=target_release,
            random_state=rng,
        )
        samples[sim] = releases[target_release - 1]
    return samples


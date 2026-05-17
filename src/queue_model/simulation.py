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


def simulate_transient_release_times_from_opening(
    current_time: float,
    dining_time_distribution: DiningTimeDistribution,
    n_tables: int,
    n_releases: int,
    random_state: RandomStateLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate future releases after first simulating service from 16:00.

    At t=0 all tables are assumed occupied by the first wave of customers. The
    restaurant is saturated, so every release before ``current_time`` is
    immediately followed by a new party entering that table. Returned release
    times are measured relative to ``current_time``.
    """
    if current_time < 0:
        raise ValueError("current_time must be non-negative.")
    if n_tables <= 0:
        raise ValueError("n_tables must be positive.")
    if n_releases <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    rng = get_rng(random_state)
    heap: list[tuple[float, int, float]] = []
    initial_durations = dining_time_distribution.sample(n_tables, random_state=rng)
    for table_id, duration in enumerate(initial_durations):
        heapq.heappush(heap, (float(duration), table_id, 0.0))

    # Move the renewal process forward to the observation time. A release at
    # exactly current_time belongs to the future event process with wait 0.
    while heap and heap[0][0] < current_time:
        release_time, table_id, _service_start = heapq.heappop(heap)
        next_duration = dining_time_distribution.sample(1, random_state=rng)[0]
        heapq.heappush(heap, (float(release_time + next_duration), table_id, float(release_time)))

    current_table_ages = np.array([current_time - service_start for _, _, service_start in heap], dtype=float)
    releases = np.empty(n_releases, dtype=float)
    for idx in range(n_releases):
        release_time, table_id, _service_start = heapq.heappop(heap)
        releases[idx] = release_time - current_time
        next_duration = dining_time_distribution.sample(1, random_state=rng)[0]
        heapq.heappush(heap, (float(release_time + next_duration), table_id, float(release_time)))

    return releases, current_table_ages


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


def monte_carlo_waiting_time_from_opening(
    current_time: float,
    queue_ahead: int,
    n_tables: int,
    dining_time_distribution: DiningTimeDistribution,
    n_simulations: int = 10000,
    random_state: RandomStateLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample waiting time using the transient process that starts at 16:00."""
    if queue_ahead < 0:
        raise ValueError("queue_ahead must be non-negative.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    rng = get_rng(random_state)
    target_release = queue_ahead + 1
    samples = np.empty(n_simulations, dtype=float)
    first_table_ages = np.empty(n_tables, dtype=float)
    for sim in range(n_simulations):
        releases, table_ages = simulate_transient_release_times_from_opening(
            current_time=current_time,
            dining_time_distribution=dining_time_distribution,
            n_tables=n_tables,
            n_releases=target_release,
            random_state=rng,
        )
        samples[sim] = releases[target_release - 1]
        if sim == 0:
            first_table_ages = table_ages
    return samples, first_table_ages

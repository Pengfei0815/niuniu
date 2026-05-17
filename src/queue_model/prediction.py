"""High-level entry-time prediction API."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from queue_model.distributions import DiningTimeDistribution
from queue_model.simulation import monte_carlo_waiting_time
from queue_model.utils import RandomStateLike, get_rng


def generate_stationary_table_ages(
    n_tables: int,
    dining_time_distribution: DiningTimeDistribution,
    random_state: RandomStateLike = None,
) -> np.ndarray:
    """Approximate current table ages under a saturated stationary renewal regime.

    For an ongoing renewal interval of length T, the equilibrium age density is
    proportional to S_T(a). A practical sampler draws T from the length-biased
    interval distribution and then draws age uniformly on [0, T]. This is used
    when exact current table ages are unavailable.
    """
    if n_tables <= 0:
        raise ValueError("n_tables must be positive.")

    rng = get_rng(random_state)
    pool_size = max(5000, n_tables * 200)
    intervals = dining_time_distribution.sample(pool_size, random_state=rng)
    weights = intervals / np.sum(intervals)
    chosen = rng.choice(intervals, size=n_tables, replace=True, p=weights)
    ages = rng.uniform(0.0, chosen)
    return np.clip(ages, 0.0, np.nextafter(dining_time_distribution.max_time, 0.0))


def predict_entry_time(
    current_time: float,
    queue_ahead: int,
    n_tables: int,
    dining_time_distribution: DiningTimeDistribution,
    table_ages: Sequence[float] | None = None,
    n_simulations: int = 10000,
    random_state: RandomStateLike = None,
) -> dict[str, float | np.ndarray]:
    """Predict waiting and entry-time distribution summaries for one customer.

    Returns summary statistics plus raw ``waiting_time_samples`` and
    ``entry_time_samples`` so callers can plot or compute custom probabilities.
    If ``table_ages`` is missing, ages are sampled from the approximate
    stationary renewal age distribution for a continuously full restaurant.
    """
    if current_time < 0:
        raise ValueError("current_time must be non-negative.")
    if n_tables <= 0:
        raise ValueError("n_tables must be positive.")

    rng = get_rng(random_state)
    if table_ages is None:
        ages = generate_stationary_table_ages(n_tables, dining_time_distribution, random_state=rng)
    else:
        ages = np.asarray(table_ages, dtype=float)
        if ages.shape != (n_tables,):
            raise ValueError("table_ages must have length n_tables.")
        if np.any(ages < 0) or np.any(ages >= dining_time_distribution.max_time):
            raise ValueError("table_ages must be in [0, max_time).")

    wait_samples = monte_carlo_waiting_time(
        queue_ahead=queue_ahead,
        table_ages=ages,
        dining_time_distribution=dining_time_distribution,
        n_simulations=n_simulations,
        random_state=rng,
    )
    entry_samples = current_time + wait_samples

    result: dict[str, float | np.ndarray] = {
        "mean_wait": float(np.mean(wait_samples)),
        "median_wait": float(np.median(wait_samples)),
        "q10_wait": float(np.quantile(wait_samples, 0.10)),
        "q90_wait": float(np.quantile(wait_samples, 0.90)),
        "p_wait_le_30": float(np.mean(wait_samples <= 30.0)),
        "p_wait_le_60": float(np.mean(wait_samples <= 60.0)),
        "p_wait_le_90": float(np.mean(wait_samples <= 90.0)),
        "mean_entry_time": float(np.mean(entry_samples)),
        "median_entry_time": float(np.median(entry_samples)),
        "q10_entry_time": float(np.quantile(entry_samples, 0.10)),
        "q90_entry_time": float(np.quantile(entry_samples, 0.90)),
        "waiting_time_samples": wait_samples,
        "entry_time_samples": entry_samples,
        "table_ages_used": ages,
    }
    return result


"""Survival-analysis utilities for dining-time models."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from queue_model.distributions import TruncatedWeibullDiningTime


def conditional_residual_survival(distribution, age: float, r: float | np.ndarray) -> float | np.ndarray:
    """Return P(R > r | T > age) for an occupied table of current age ``age``.

    Parameters
    ----------
    distribution:
        Dining-time distribution with a ``survival`` method.
    age:
        Minutes already spent at the table.
    r:
        Future residual time in minutes.
    """
    if age < 0:
        raise ValueError("age must be non-negative.")

    denominator = distribution.survival(age)
    if denominator <= 0:
        raise ValueError("distribution.survival(age) must be positive.")

    values = np.asarray(r, dtype=float)
    max_residual = max(distribution.max_time - age, 0.0)
    numerator = distribution.survival(age + values)
    survival = np.where(values < 0, 1.0, np.where(values >= max_residual, 0.0, numerator / denominator))
    survival = np.clip(survival, 0.0, 1.0)
    return float(survival) if np.ndim(r) == 0 else survival


def estimate_weibull_mle_from_censored_data(
    times: np.ndarray,
    event_observed: np.ndarray,
    max_time: float = 120.0,
) -> TruncatedWeibullDiningTime:
    """Estimate Weibull parameters from right-censored dining data.

    ``event_observed=1`` means the customer left before the maximum duration and
    contributes log f_X(t). ``event_observed=0`` means the table reached
    ``max_time`` and contributes log S_X(max_time) as right censoring.
    """
    observed_times = np.asarray(times, dtype=float)
    events = np.asarray(event_observed, dtype=int)
    if observed_times.shape != events.shape:
        raise ValueError("times and event_observed must have the same shape.")
    if np.any(observed_times <= 0):
        raise ValueError("all times must be positive.")
    if not np.all(np.isin(events, [0, 1])):
        raise ValueError("event_observed must contain only 0 and 1.")

    censored_times = np.where(events == 1, observed_times, max_time)

    def neg_log_likelihood(log_params: np.ndarray) -> float:
        shape = np.exp(log_params[0])
        scale = np.exp(log_params[1])
        t = censored_times

        log_pdf = np.log(shape) - shape * np.log(scale) + (shape - 1.0) * np.log(t) - (t / scale) ** shape
        log_survival = -((t / scale) ** shape)
        log_likelihood = np.where(events == 1, log_pdf, log_survival)
        return float(-np.sum(log_likelihood))

    initial_scale = max(float(np.median(censored_times)), 1.0)
    result = optimize.minimize(
        neg_log_likelihood,
        x0=np.log(np.array([2.0, initial_scale])),
        method="Nelder-Mead",
        options={"maxiter": 10000},
    )
    if not result.success:
        raise RuntimeError(f"Weibull MLE optimization failed: {result.message}")

    shape, scale = np.exp(result.x)
    return TruncatedWeibullDiningTime(shape=float(shape), scale=float(scale), max_time=max_time)


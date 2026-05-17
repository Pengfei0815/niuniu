"""Dining-time distributions truncated by a hard maximum dining duration.

The natural dining time X follows a parametric continuous distribution. The
observed table occupation time is T = min(X, max_time), which creates a point
mass at max_time for customers who use the full allowed duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import integrate, stats

from queue_model.utils import RandomStateLike, get_rng


class DiningTimeDistribution(Protocol):
    """Protocol implemented by dining-time distributions used by the simulator."""

    max_time: float

    def pdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return the continuous density for T before the point mass."""

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return P(T <= t), including the point mass at max_time."""

    def survival(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return P(T > t). At max_time and beyond this is zero."""

    def sample(self, size: int | tuple[int, ...], random_state: RandomStateLike = None) -> np.ndarray:
        """Sample actual dining durations T."""

    def mean(self) -> float:
        """Return E[T]."""

    def prob_full_duration(self) -> float:
        """Return P(T = max_time)."""

    def sample_residual(
        self, age: float, size: int | tuple[int, ...], random_state: RandomStateLike = None
    ) -> np.ndarray:
        """Sample T - age conditional on T > age."""


@dataclass(frozen=True)
class TruncatedWeibullDiningTime:
    """Weibull natural dining time with hard truncation at ``max_time`` minutes."""

    shape: float
    scale: float
    max_time: float = 120.0

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0 or self.max_time <= 0:
            raise ValueError("shape, scale, and max_time must be positive.")

    def _dist(self) -> stats.rv_continuous:
        return stats.weibull_min(c=self.shape, scale=self.scale)

    def pdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return density on [0, max_time); the atom at max_time is excluded."""
        values = np.asarray(t, dtype=float)
        valid = (values >= 0) & (values < self.max_time)
        z = np.maximum(values / self.scale, 0.0)
        density = (self.shape / self.scale) * np.power(z, self.shape - 1.0) * np.exp(-(z**self.shape))
        density = np.where(valid, density, 0.0)
        return float(density) if np.ndim(t) == 0 else density

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return CDF of T = min(X, max_time), including F_T(max_time)=1."""
        values = np.asarray(t, dtype=float)
        z = np.maximum(values / self.scale, 0.0)
        natural_cdf = 1.0 - np.exp(-(z**self.shape))
        probs = np.where(values < 0, 0.0, np.where(values < self.max_time, natural_cdf, 1.0))
        return float(probs) if np.ndim(t) == 0 else probs

    def survival(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return P(T > t), so survival(max_time)=0 for actual observed T."""
        values = np.asarray(t, dtype=float)
        z = np.maximum(values / self.scale, 0.0)
        natural_survival = np.exp(-(z**self.shape))
        surv = np.where(values < 0, 1.0, np.where(values < self.max_time, natural_survival, 0.0))
        return float(surv) if np.ndim(t) == 0 else surv

    def sample(self, size: int | tuple[int, ...], random_state: RandomStateLike = None) -> np.ndarray:
        """Sample actual dining times capped at ``max_time``."""
        rng = get_rng(random_state)
        natural = self.scale * rng.weibull(self.shape, size=size)
        return np.minimum(natural, self.max_time)

    def mean(self) -> float:
        """Return E[min(X, max_time)] using the survival integral identity."""
        value, _ = integrate.quad(lambda x: self._dist().sf(x), 0.0, self.max_time, limit=200)
        return float(value)

    def prob_full_duration(self) -> float:
        """Return the point mass P(X >= max_time)."""
        return float(np.exp(-((self.max_time / self.scale) ** self.shape)))

    def sample_residual(
        self, age: float, size: int | tuple[int, ...], random_state: RandomStateLike = None
    ) -> np.ndarray:
        """Sample residual time T-age conditional on the table still being occupied.

        The inverse-CDF draw uses F_X(age) + U * (1 - F_X(age)) and then caps at
        max_time. If the latent X exceeds max_time, the residual is exactly
        max_time-age, preserving the atom at the full duration.
        """
        if age < 0:
            raise ValueError("age must be non-negative.")
        if age >= self.max_time:
            raise ValueError("age must be smaller than max_time for an occupied table.")

        rng = get_rng(random_state)
        exponential = rng.exponential(scale=1.0, size=size)
        natural = self.scale * (((age / self.scale) ** self.shape + exponential) ** (1.0 / self.shape))
        actual = np.minimum(natural, self.max_time)
        return np.clip(actual - age, 0.0, self.max_time - age)


@dataclass(frozen=True)
class TruncatedLognormalDiningTime:
    """Lognormal natural dining time with hard truncation at ``max_time`` minutes."""

    mu: float
    sigma: float
    max_time: float = 120.0

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.max_time <= 0:
            raise ValueError("sigma and max_time must be positive.")

    def _dist(self) -> stats.rv_continuous:
        return stats.lognorm(s=self.sigma, scale=np.exp(self.mu))

    def pdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return density on [0, max_time); the atom at max_time is excluded."""
        values = np.asarray(t, dtype=float)
        density = np.where((values >= 0) & (values < self.max_time), self._dist().pdf(values), 0.0)
        return float(density) if np.ndim(t) == 0 else density

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return CDF of T = min(X, max_time), including F_T(max_time)=1."""
        values = np.asarray(t, dtype=float)
        probs = np.where(values < 0, 0.0, np.where(values < self.max_time, self._dist().cdf(values), 1.0))
        return float(probs) if np.ndim(t) == 0 else probs

    def survival(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return P(T > t), so survival(max_time)=0 for actual observed T."""
        values = np.asarray(t, dtype=float)
        surv = np.where(values < 0, 1.0, np.where(values < self.max_time, self._dist().sf(values), 0.0))
        return float(surv) if np.ndim(t) == 0 else surv

    def sample(self, size: int | tuple[int, ...], random_state: RandomStateLike = None) -> np.ndarray:
        """Sample actual dining times capped at ``max_time``."""
        rng = get_rng(random_state)
        natural = rng.lognormal(mean=self.mu, sigma=self.sigma, size=size)
        return np.minimum(natural, self.max_time)

    def mean(self) -> float:
        """Return E[min(X, max_time)] using the survival integral identity."""
        value, _ = integrate.quad(lambda x: self._dist().sf(x), 0.0, self.max_time, limit=200)
        return float(value)

    def prob_full_duration(self) -> float:
        """Return the point mass P(X >= max_time)."""
        return float(self._dist().sf(self.max_time))

    def sample_residual(
        self, age: float, size: int | tuple[int, ...], random_state: RandomStateLike = None
    ) -> np.ndarray:
        """Sample residual time T-age conditional on T > age."""
        return _sample_residual_from_dist(self._dist(), self.max_time, age, size, random_state)


def _sample_residual_from_dist(
    natural_dist: stats.rv_continuous,
    max_time: float,
    age: float,
    size: int | tuple[int, ...],
    random_state: RandomStateLike = None,
) -> np.ndarray:
    """Sample residual time from the latent distribution conditional on survival."""
    if age < 0:
        raise ValueError("age must be non-negative.")
    if age >= max_time:
        raise ValueError("age must be smaller than max_time for an occupied table.")

    rng = get_rng(random_state)
    lower = natural_dist.cdf(age)
    uniforms = rng.uniform(lower, 1.0, size=size)
    natural = natural_dist.ppf(uniforms)
    actual = np.minimum(natural, max_time)
    residual = actual - age
    return np.clip(residual, 0.0, max_time - age)

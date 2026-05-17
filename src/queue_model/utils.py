"""Utility helpers for time formatting and random number generation."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

RandomStateLike = Optional[Union[int, np.random.Generator]]


def get_rng(random_state: RandomStateLike = None) -> np.random.Generator:
    """Return a NumPy random generator from an int seed, generator, or None."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def minutes_to_clock(minutes_since_1600: float) -> str:
    """Convert minutes since 16:00 to a HH:MM clock string."""
    total_minutes = int(round(16 * 60 + minutes_since_1600))
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def ensure_1d_array(values: float | np.ndarray) -> np.ndarray:
    """Convert scalar or array-like input to a one-dimensional float array."""
    return np.atleast_1d(np.asarray(values, dtype=float))


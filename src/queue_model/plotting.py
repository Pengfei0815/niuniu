"""Plotting helpers for waiting-time prediction outputs."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_waiting_time_distribution(samples: np.ndarray):
    """Plot a histogram of Monte Carlo waiting-time samples."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(samples, bins=40, density=True, alpha=0.75, color="#4C78A8", edgecolor="white")
    ax.axvline(np.median(samples), color="#F58518", linewidth=2, label="median")
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted waiting-time distribution")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_arrival_time_comparison(results_by_time: Mapping[str, Mapping[str, float]]):
    """Plot mean, median, and 90% quantile waiting times across arrival times."""
    labels = list(results_by_time.keys())
    mean_values = [float(results_by_time[label]["mean_wait"]) for label in labels]
    median_values = [float(results_by_time[label]["median_wait"]) for label in labels]
    q90_values = [float(results_by_time[label]["q90_wait"]) for label in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mean_values, marker="o", linewidth=2, label="mean")
    ax.plot(x, median_values, marker="s", linewidth=2, label="median")
    ax.plot(x, q90_values, marker="^", linewidth=2, label="q90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Arrival / ticket time")
    ax.set_ylabel("Waiting time (minutes)")
    ax.set_title("Waiting-time comparison by arrival time")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


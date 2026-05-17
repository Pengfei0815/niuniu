"""Compare waiting-time predictions for several ticket-taking times."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from queue_model import TruncatedWeibullDiningTime, predict_entry_time
from queue_model.plotting import plot_arrival_time_comparison


def main() -> None:
    """Run a synthetic comparison and save a matplotlib figure."""
    distribution = TruncatedWeibullDiningTime(shape=8.0, scale=95.0, max_time=120.0)
    scenarios = [
        ("16:30", 30.0, 8),
        ("17:00", 60.0, 14),
        ("17:30", 90.0, 20),
        ("18:00", 120.0, 26),
        ("18:30", 150.0, 24),
        ("19:00", 180.0, 18),
    ]

    results_by_time: dict[str, dict[str, float]] = {}
    print("不同取号时间的等待时间比较")
    print("time   queue_ahead   mean   median   q90")
    for index, (label, current_time, queue_ahead) in enumerate(scenarios):
        result = predict_entry_time(
            current_time=current_time,
            queue_ahead=queue_ahead,
            n_tables=40,
            dining_time_distribution=distribution,
            n_simulations=3000,
            random_state=100 + index,
        )
        results_by_time[label] = result
        print(
            f"{label}      {queue_ahead:>3d}       "
            f"{result['mean_wait']:>5.1f}   {result['median_wait']:>6.1f}   {result['q90_wait']:>5.1f}"
        )

    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, _ = plot_arrival_time_comparison(results_by_time)
    output_path = figure_dir / "arrival_time_comparison.png"
    fig.savefig(output_path, dpi=160)
    print(f"图像已保存: {output_path}")


if __name__ == "__main__":
    main()

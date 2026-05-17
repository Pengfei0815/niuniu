"""Example for predicting one customer's entry time."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from queue_model import TruncatedWeibullDiningTime, predict_entry_time
from queue_model.utils import minutes_to_clock


def main() -> None:
    """Predict a single customer's waiting and entry-time summary."""
    current_time = 135.0
    queue_ahead = 18
    n_tables = 40
    distribution = TruncatedWeibullDiningTime(shape=3.0, scale=95.0, max_time=120.0)

    result = predict_entry_time(
        current_time=current_time,
        queue_ahead=queue_ahead,
        n_tables=n_tables,
        dining_time_distribution=distribution,
        n_simulations=8000,
        random_state=7,
    )

    print(
        f"当前时间 {minutes_to_clock(current_time)}，前方还有 {queue_ahead} 桌。"
    )
    print(
        f"预计 {result['mean_wait']:.1f} 分钟后进场；"
        f"中位数 {result['median_wait']:.1f} 分钟；"
        f"保守估计 {result['q90_wait']:.1f} 分钟内进场。"
    )
    print(
        f"预计进场时间约为 {minutes_to_clock(float(result['median_entry_time']))}，"
        f"90% 分位进场时间为 {minutes_to_clock(float(result['q90_entry_time']))}。"
    )


if __name__ == "__main__":
    main()


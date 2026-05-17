"""Synthetic demonstration of queue waiting-time prediction."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from queue_model import TruncatedWeibullDiningTime, predict_entry_time
from queue_model.utils import minutes_to_clock


def main() -> None:
    """Run the requested synthetic 18:00 demo."""
    distribution = TruncatedWeibullDiningTime(shape=8.0, scale=95.0, max_time=120.0)
    result = predict_entry_time(
        current_time=120.0,
        queue_ahead=20,
        n_tables=40,
        dining_time_distribution=distribution,
        n_simulations=10000,
        random_state=2026,
    )

    print("牛New寿喜烧自助餐排队预测 - synthetic demo")
    print("当前时间: 18:00, 前方桌数: 20, 餐桌数: 40")
    print(f"平均等待: {result['mean_wait']:.1f} 分钟")
    print(f"中位等待: {result['median_wait']:.1f} 分钟")
    print(f"10% 分位等待: {result['q10_wait']:.1f} 分钟")
    print(f"90% 分位等待: {result['q90_wait']:.1f} 分钟")
    print(f"P(W <= 30): {result['p_wait_le_30']:.3f}")
    print(f"P(W <= 60): {result['p_wait_le_60']:.3f}")
    print(f"P(W <= 90): {result['p_wait_le_90']:.3f}")
    print(f"预计平均进场时间: {minutes_to_clock(float(result['mean_entry_time']))}")
    print(f"预计中位进场时间: {minutes_to_clock(float(result['median_entry_time']))}")


if __name__ == "__main__":
    main()

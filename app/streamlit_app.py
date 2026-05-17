"""Streamlit frontend for the restaurant queue survival model."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import importlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import queue_model.prediction as prediction_module
import queue_model.simulation as simulation_module
from queue_model import TruncatedLognormalDiningTime, TruncatedWeibullDiningTime
from queue_model.utils import minutes_to_clock

simulation_module = importlib.reload(simulation_module)
prediction_module = importlib.reload(prediction_module)
predict_entry_time = prediction_module.predict_entry_time


TIME_OPTIONS = {
    "16:00": 0.0,
    "16:15": 15.0,
    "16:30": 30.0,
    "16:45": 45.0,
    "17:00": 60.0,
    "17:15": 75.0,
    "17:30": 90.0,
    "17:45": 105.0,
    "18:00": 120.0,
    "18:15": 135.0,
    "18:30": 150.0,
    "18:45": 165.0,
    "19:00": 180.0,
    "19:15": 195.0,
    "19:30": 210.0,
    "19:45": 225.0,
    "20:00": 240.0,
    "20:15": 255.0,
    "20:30": 270.0,
    "20:45": 285.0,
    "21:00": 300.0,
}


def configure_page() -> None:
    """Configure Streamlit page metadata and styling."""
    st.set_page_config(
        page_title="何时吃上牛",
        page_icon="N",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.35rem;
            padding-bottom: 2.4rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }
        .hero {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 1.55rem 1.25rem 1.25rem 1.25rem;
            margin: 0.65rem 0 1rem 0;
            background:
                linear-gradient(135deg, rgba(10, 95, 115, 0.12), rgba(214, 125, 48, 0.12)),
                #ffffff;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            overflow: hidden;
        }
        .app-title {
            display: block;
            font-size: 2.18rem;
            font-weight: 820;
            color: #0F2433;
            line-height: 1.35;
            min-height: 3.2rem;
            padding: 0;
            margin-bottom: 0.1rem;
            overflow: hidden;
            letter-spacing: 0;
        }
        .app-subtitle {
            color: #4B5563;
            font-size: 1.02rem;
            margin-bottom: 0;
        }
        .footer-note {
            margin-top: 1.6rem;
            padding-top: 1rem;
            border-top: 1px solid #E5E7EB;
            color: #52606D;
            font-size: 0.94rem;
            text-align: center;
        }
        .footer-note a {
            color: #0B7285;
            font-weight: 650;
            text-decoration: none;
        }
        .footer-note a:hover {
            text-decoration: underline;
        }
        .summary-band {
            border: 1px solid rgba(11, 114, 133, 0.16);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            margin: 0 0 1rem 0;
        }
        .clock-text {
            font-size: 1.9rem;
            font-weight: 820;
            color: #0B7285;
            line-height: 1.18;
            margin-bottom: 0.55rem;
        }
        .risk-text {
            font-size: 1.9rem;
            font-weight: 820;
            color: #C2410C;
            line-height: 1.18;
        }
        .caption {
            color: #66788A;
            font-size: 0.88rem;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748B;
        }
        .stButton > button {
            border-radius: 12px;
            min-height: 2.8rem;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
        }
        @media (max-width: 760px) {
            .block-container {
                padding: 2rem 0.75rem 1.8rem 0.75rem;
            }
            .hero {
                border-radius: 16px;
                padding: 1.4rem 1rem 1rem 1rem;
                margin-top: 0.55rem;
            }
            .app-title {
                font-size: 1.72rem;
                line-height: 1.42;
                min-height: 2.9rem;
            }
            .app-subtitle {
                font-size: 0.94rem;
            }
            .clock-text,
            .risk-text {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_distribution(
    distribution_name: str,
    weibull_shape: float,
    weibull_scale: float,
    lognormal_median: float,
    lognormal_sigma: float,
):
    """Construct the selected dining-time distribution from UI values."""
    if distribution_name == "Weibull":
        return TruncatedWeibullDiningTime(shape=weibull_shape, scale=weibull_scale, max_time=120.0)
    return TruncatedLognormalDiningTime(mu=float(np.log(lognormal_median)), sigma=lognormal_sigma, max_time=120.0)


@st.cache_data(show_spinner=False)
def cached_prediction(
    current_time: float,
    queue_ahead: int,
    n_tables: int,
    distribution_name: str,
    weibull_shape: float,
    weibull_scale: float,
    lognormal_median: float,
    lognormal_sigma: float,
    table_age_mode: str,
    uniform_age: float,
    n_simulations: int,
    random_seed: int,
) -> dict[str, float | np.ndarray]:
    """Run prediction with only hashable primitive cache arguments."""
    distribution = build_distribution(
        distribution_name=distribution_name,
        weibull_shape=weibull_shape,
        weibull_scale=weibull_scale,
        lognormal_median=lognormal_median,
        lognormal_sigma=lognormal_sigma,
    )
    table_ages = None
    current_state_model = "transient"
    if table_age_mode == "统一桌龄":
        table_ages = np.full(n_tables, uniform_age, dtype=float)
    elif table_age_mode == "平稳近似":
        current_state_model = "stationary"

    return predict_entry_time(
        current_time=current_time,
        queue_ahead=queue_ahead,
        n_tables=n_tables,
        dining_time_distribution=distribution,
        table_ages=table_ages,
        current_state_model=current_state_model,
        n_simulations=n_simulations,
        random_state=random_seed,
    )


def waiting_time_figure(samples: np.ndarray) -> plt.Figure:
    """Create a waiting-time histogram with key quantile lines."""
    q10, median, q90 = np.quantile(samples, [0.1, 0.5, 0.9])
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.hist(samples, bins=36, density=True, color="#4C78A8", edgecolor="white", alpha=0.82)
    ax.axvline(q10, color="#38A169", linewidth=2, label="q10")
    ax.axvline(median, color="#D97706", linewidth=2, label="median")
    ax.axvline(q90, color="#C2410C", linewidth=2, label="q90")
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Waiting-time predictive distribution")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def entry_time_figure(entry_samples: np.ndarray) -> plt.Figure:
    """Create an entry-time histogram in clock-time units."""
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.hist(entry_samples, bins=30, color="#72B7B2", edgecolor="white", alpha=0.86)
    ticks = np.linspace(np.min(entry_samples), np.max(entry_samples), 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([minutes_to_clock(float(tick)) for tick in ticks])
    ax.set_xlabel("Predicted entry time")
    ax.set_ylabel("Monte Carlo count")
    ax.set_title("Entry-time distribution")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    return fig


def comparison_figure(results_by_time: dict[str, dict[str, float | np.ndarray]]) -> plt.Figure:
    """Create a comparison line chart for mean, median, and q90 wait."""
    labels = list(results_by_time.keys())
    x = np.arange(len(labels))
    mean_wait = [float(results_by_time[label]["mean_wait"]) for label in labels]
    median_wait = [float(results_by_time[label]["median_wait"]) for label in labels]
    q90_wait = [float(results_by_time[label]["q90_wait"]) for label in labels]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(x, mean_wait, marker="o", linewidth=2.4, color="#4C78A8", label="mean")
    ax.plot(x, median_wait, marker="s", linewidth=2.4, color="#D97706", label="median")
    ax.plot(x, q90_wait, marker="^", linewidth=2.4, color="#C2410C", label="q90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Waiting time (minutes)")
    ax.set_title("Waiting-time comparison by arrival time")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_probability(label: str, probability: float) -> None:
    """Render a compact probability bar."""
    st.caption(label)
    st.progress(float(probability), text=f"{probability:.1%}")


def render_single_prediction(
    distribution_name: str,
    weibull_shape: float,
    weibull_scale: float,
    lognormal_median: float,
    lognormal_sigma: float,
    n_tables: int,
    n_simulations: int,
    random_seed: int,
) -> None:
    """Render the single-customer prediction tab."""
    input_col, output_col = st.columns([0.95, 1.35], gap="medium")
    with input_col:
        st.subheader("现场状态")
        current_label = st.select_slider("当前时间", options=list(TIME_OPTIONS.keys()), value="18:00")
        queue_ahead = st.number_input("前方桌数", min_value=0, max_value=260, value=20, step=1)
        table_age_mode = st.radio("当前桌龄", ["从16:00开门模拟", "平稳近似", "统一桌龄"], horizontal=True)
        uniform_age = 45.0
        if table_age_mode == "统一桌龄":
            uniform_age = st.slider("每桌已用餐分钟数", min_value=0.0, max_value=119.0, value=45.0, step=1.0)

        run_button = st.button("更新预测", type="primary", width="stretch")
        st.caption("默认从 16:00 第一波入座开始模拟；平稳近似只适合较晚且接近稳定的满座时段。")

    with output_col:
        current_time = TIME_OPTIONS[current_label]
        with st.spinner("正在模拟未来释放过程..."):
            result = cached_prediction(
                current_time=current_time,
                queue_ahead=int(queue_ahead),
                n_tables=n_tables,
                distribution_name=distribution_name,
                weibull_shape=weibull_shape,
                weibull_scale=weibull_scale,
                lognormal_median=lognormal_median,
                lognormal_sigma=lognormal_sigma,
                table_age_mode=table_age_mode,
                uniform_age=uniform_age,
                n_simulations=n_simulations,
                random_seed=random_seed + (1 if run_button else 0),
            )

        st.markdown(
            f"""
            <div class="summary-band">
              <div class="caption">中位进场时间</div>
              <div class="clock-text">{minutes_to_clock(float(result["median_entry_time"]))}</div>
              <div class="caption">90% 分位进场时间</div>
              <div class="risk-text">{minutes_to_clock(float(result["q90_entry_time"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("平均等待", f"{float(result['mean_wait']):.1f} 分钟")
        metric_cols[1].metric("中位等待", f"{float(result['median_wait']):.1f} 分钟")
        metric_cols[2].metric("10% 分位", f"{float(result['q10_wait']):.1f} 分钟")
        metric_cols[3].metric("90% 分位", f"{float(result['q90_wait']):.1f} 分钟")

        probability_cols = st.columns(3, gap="small")
        with probability_cols[0]:
            render_probability("30 分钟内进场", float(result["p_wait_le_30"]))
        with probability_cols[1]:
            render_probability("60 分钟内进场", float(result["p_wait_le_60"]))
        with probability_cols[2]:
            render_probability("90 分钟内进场", float(result["p_wait_le_90"]))

    chart_col, entry_col = st.columns([1.2, 1.0], gap="medium")
    with chart_col:
        st.pyplot(waiting_time_figure(np.asarray(result["waiting_time_samples"])), clear_figure=True)
    with entry_col:
        st.pyplot(entry_time_figure(np.asarray(result["entry_time_samples"])), clear_figure=True)


def render_comparison(
    distribution_name: str,
    weibull_shape: float,
    weibull_scale: float,
    lognormal_median: float,
    lognormal_sigma: float,
    n_tables: int,
    n_simulations: int,
    random_seed: int,
) -> None:
    """Render the arrival-time comparison tab."""
    st.subheader("不同取号时间对比")
    default_queues = {
        "16:30": 8,
        "17:00": 14,
        "17:30": 20,
        "18:00": 26,
        "18:30": 24,
        "19:00": 18,
    }
    queue_cols = st.columns(len(default_queues))
    queue_by_label: dict[str, int] = {}
    for col, (label, default_queue) in zip(queue_cols, default_queues.items()):
        with col:
            queue_by_label[label] = int(st.number_input(label, min_value=0, max_value=260, value=default_queue, step=1))

    results_by_time: dict[str, dict[str, float | np.ndarray]] = {}
    with st.spinner("正在比较不同时段..."):
        for index, (label, queue_ahead) in enumerate(queue_by_label.items()):
            results_by_time[label] = cached_prediction(
                current_time=TIME_OPTIONS[label],
                queue_ahead=queue_ahead,
                n_tables=n_tables,
                distribution_name=distribution_name,
                weibull_shape=weibull_shape,
                weibull_scale=weibull_scale,
                lognormal_median=lognormal_median,
                lognormal_sigma=lognormal_sigma,
                table_age_mode="从16:00开门模拟",
                uniform_age=45.0,
                n_simulations=max(800, n_simulations // 3),
                random_seed=random_seed + 100 + index,
            )

    st.pyplot(comparison_figure(results_by_time), clear_figure=True)

    rows = []
    for label, result in results_by_time.items():
        rows.append(
            {
                "取号时间": label,
                "前方桌数": queue_by_label[label],
                "平均等待": f"{float(result['mean_wait']):.1f}",
                "中位等待": f"{float(result['median_wait']):.1f}",
                "90% 分位": f"{float(result['q90_wait']):.1f}",
                "中位进场": minutes_to_clock(float(result["median_entry_time"])),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def render_sidebar():
    """Render shared model controls and return selected values."""
    with st.sidebar:
        st.header("模型参数")
        n_tables = st.slider("餐桌数", min_value=10, max_value=120, value=40, step=1)
        n_simulations = st.slider("Monte Carlo 次数", min_value=800, max_value=12000, value=4000, step=400)
        random_seed = st.number_input("随机种子", min_value=0, max_value=999999, value=2026, step=1)

        st.divider()
        distribution_name = st.radio("用餐时间分布", ["Weibull", "Lognormal"], horizontal=True)
        weibull_shape = 8.0
        weibull_scale = 95.0
        lognormal_median = 90.0
        lognormal_sigma = 0.35
        if distribution_name == "Weibull":
            weibull_shape = st.slider("Weibull shape", min_value=0.8, max_value=12.0, value=8.0, step=0.1)
            weibull_scale = st.slider("Weibull scale", min_value=40.0, max_value=140.0, value=95.0, step=1.0)
        else:
            lognormal_median = st.slider("Lognormal median", min_value=40.0, max_value=120.0, value=90.0, step=1.0)
            lognormal_sigma = st.slider("Lognormal sigma", min_value=0.1, max_value=1.0, value=0.35, step=0.05)

        distribution = build_distribution(
            distribution_name=distribution_name,
            weibull_shape=weibull_shape,
            weibull_scale=weibull_scale,
            lognormal_median=lognormal_median,
            lognormal_sigma=lognormal_sigma,
        )
        st.metric("平均用餐时长", f"{distribution.mean():.1f} 分钟")
        st.metric("用满 120 分钟概率", f"{distribution.prob_full_duration():.1%}")
        with st.expander("为什么用 Monte Carlo"):
            st.write(
                "单桌用餐时间的均值可以直接算，但目标等待时间是所有桌子未来释放事件排序后的第 k+1 个事件。"
                "这个顺序统计量叠加了条件剩余寿命、120 分钟点质量和多轮翻台卷积，闭式分位数通常不可得。"
                "Monte Carlo 直接模拟释放过程，更适合输出中位数、90% 分位和 30/60/90 分钟内进场概率。"
            )

    return (
        distribution_name,
        weibull_shape,
        weibull_scale,
        lognormal_median,
        lognormal_sigma,
        n_tables,
        n_simulations,
        int(random_seed),
    )


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    st.markdown(
        """<div class="hero">
          <div class="app-title">何时吃上牛</div>
          <div class="app-subtitle">排队不是玄学。用生存模型估计你大概什么时候能吃上牛牛。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_image = PROJECT_ROOT / "figures" / "summary.png"
    if summary_image.exists():
        st.image(str(summary_image), width="stretch")

    params = render_sidebar()
    tab_single, tab_compare = st.tabs(["单人预测", "时段对比"])
    with tab_single:
        render_single_prediction(*params)
    with tab_compare:
        render_comparison(*params)

    st.markdown(
        """
        <div class="footer-note">
          不要让你的人生在等待牛牛的时间虚度！<br>
          GitHub:
          <a href="https://github.com/Pengfei0815/niuniu" target="_blank">
            https://github.com/Pengfei0815/niuniu
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

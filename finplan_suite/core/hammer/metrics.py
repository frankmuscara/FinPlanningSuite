"""Performance metrics calculation for HAMMER."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Return metrics
    total_return: float
    cagr: float

    # Risk metrics
    volatility: float             # Annualized
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int    # Days

    # Relative metrics (vs benchmark)
    beta: Optional[float]
    alpha: Optional[float]
    tracking_error: Optional[float]
    information_ratio: Optional[float]

    # Turnover metrics
    rebalances_per_year: float
    blocked_rebalances: int
    partial_rebalances: int = 0
    total_turnover: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "Total Return": self.total_return,
            "CAGR": self.cagr,
            "Volatility (Ann.)": self.volatility,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown": self.max_drawdown,
            "Max DD Duration (days)": self.max_drawdown_duration,
            "Beta": self.beta,
            "Alpha": self.alpha,
            "Tracking Error": self.tracking_error,
            "Information Ratio": self.information_ratio,
            "Rebalances/Year": self.rebalances_per_year,
            "Blocked Rebalances": self.blocked_rebalances,
            "Partial Rebalances": self.partial_rebalances,
            "Total Turnover": self.total_turnover,
        }

    def format_dict(self) -> dict:
        """Convert to formatted string dictionary."""
        return {
            "Total Return": f"{self.total_return:.2%}",
            "CAGR": f"{self.cagr:.2%}",
            "Volatility (Ann.)": f"{self.volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max DD Duration": f"{self.max_drawdown_duration} days",
            "Beta": f"{self.beta:.2f}" if self.beta else "N/A",
            "Alpha": f"{self.alpha:.2%}" if self.alpha else "N/A",
            "Tracking Error": f"{self.tracking_error:.2%}" if self.tracking_error else "N/A",
            "Info Ratio": f"{self.information_ratio:.2f}" if self.information_ratio else "N/A",
            "Rebalances/Year": f"{self.rebalances_per_year:.1f}",
            "Blocked Rebalances": str(self.blocked_rebalances),
            "Partial Rebalances": str(self.partial_rebalances),
            "Total Turnover": f"{self.total_turnover:.2%}",
        }


def compute_metrics(
    nav: pd.Series,
    benchmark_nav: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    rebalances_per_year: float = 0,
    blocked_rebalances: int = 0,
    partial_rebalances: int = 0,
    total_turnover: float = 0,
) -> PerformanceMetrics:
    """Compute performance metrics from NAV series.

    Args:
        nav: Portfolio NAV time series
        benchmark_nav: Benchmark NAV time series (optional)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year
        rebalances_per_year: Average rebalances per year
        blocked_rebalances: Count of blocked rebalances
        partial_rebalances: Count of partial rebalances (equity frozen)
        total_turnover: Sum of turnover from all rebalances

    Returns:
        PerformanceMetrics object
    """
    nav = nav.dropna()
    if len(nav) < 2:
        raise ValueError("Need at least 2 data points")

    returns = nav.pct_change().dropna()

    # Basic return metrics
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(nav) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Risk metrics
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    sharpe = (
        np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        if excess_returns.std() > 0 else 0
    )

    # Sortino ratio (downside deviation)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt((downside_returns ** 2).mean()) if len(downside_returns) > 0 else 0
    sortino = (
        np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
        if downside_std > 0 else 0
    )

    # Max drawdown
    cumulative = nav / nav.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Max drawdown duration
    is_in_drawdown = drawdown < 0
    dd_groups = (~is_in_drawdown).cumsum()
    dd_durations = is_in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # Relative metrics (if benchmark provided)
    beta = None
    alpha = None
    tracking_error = None
    information_ratio = None

    if benchmark_nav is not None:
        benchmark_nav = benchmark_nav.dropna()
        common_idx = nav.index.intersection(benchmark_nav.index)
        nav_aligned = nav.loc[common_idx]
        benchmark_aligned = benchmark_nav.loc[common_idx]

        if len(common_idx) > 1:
            port_returns = nav_aligned.pct_change().dropna()
            bench_returns = benchmark_aligned.pct_change().dropna()

            common_ret_idx = port_returns.index.intersection(bench_returns.index)
            port_returns = port_returns.loc[common_ret_idx]
            bench_returns = bench_returns.loc[common_ret_idx]

            covariance = port_returns.cov(bench_returns)
            bench_variance = bench_returns.var()
            beta = covariance / bench_variance if bench_variance > 0 else 0

            bench_cagr = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
            alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

            active_returns = port_returns - bench_returns
            tracking_error = active_returns.std() * np.sqrt(periods_per_year)

            information_ratio = (
                active_returns.mean() * periods_per_year / tracking_error
                if tracking_error > 0 else 0
            )

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        beta=beta,
        alpha=alpha,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        rebalances_per_year=rebalances_per_year,
        blocked_rebalances=blocked_rebalances,
        partial_rebalances=partial_rebalances,
        total_turnover=total_turnover,
    )


def compute_metrics_for_window(
    nav: pd.Series,
    benchmark_nav: Optional[pd.Series],
    window_end: date,
    years: float,
    **kwargs,
) -> Tuple[PerformanceMetrics, date, date]:
    """Compute metrics for a specific time window."""
    window_start = window_end - timedelta(days=int(years * 365.25))
    nav_slice = nav.loc[str(window_start):str(window_end)]
    benchmark_slice = None
    if benchmark_nav is not None:
        benchmark_slice = benchmark_nav.loc[str(window_start):str(window_end)]

    if len(nav_slice) < 2:
        raise ValueError(f"Insufficient data for {years}-year window")

    actual_start = nav_slice.index[0]
    actual_end = nav_slice.index[-1]

    metrics = compute_metrics(
        nav_slice,
        benchmark_slice,
        rebalances_per_year=kwargs.get("rebalances_per_year", 0),
        blocked_rebalances=kwargs.get("blocked_rebalances", 0),
        total_turnover=kwargs.get("total_turnover", 0),
    )

    return metrics, actual_start, actual_end


def compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """Compute drawdown time series from NAV."""
    cumulative = nav / nav.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown

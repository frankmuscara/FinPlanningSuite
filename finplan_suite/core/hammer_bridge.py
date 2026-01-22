"""Bridge between FinPlanningSuite and HAMMER.

Provides utilities to convert between FinPlanningSuite data structures
(CMA, Client, Portfolio) and HAMMER data structures (PortfolioConfig,
StrategyConfig, BacktestResult).
"""

from datetime import date
from typing import Dict, List, Optional, Tuple
import json
import os
import pandas as pd

from finplan_suite.core.hammer import (
    PortfolioConfig,
    StrategyConfig,
    StrategyMode,
    BacktestEngine,
    BacktestResult,
    PerformanceMetrics,
    compute_metrics,
    AssetClass,
    get_asset_class,
)
from finplan_suite.core.hammer.strategies import RebalanceFrequency


# Path for storing backtest results
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
HAMMER_MODELS_DIR = os.path.join(DATA_DIR, "hammer_models")
BACKTEST_CACHE_DIR = os.path.join(DATA_DIR, "backtest_results")


def ensure_dirs():
    """Ensure data directories exist."""
    os.makedirs(HAMMER_MODELS_DIR, exist_ok=True)
    os.makedirs(BACKTEST_CACHE_DIR, exist_ok=True)


def cma_to_hammer_inputs(
    cma_path: str = None
) -> Tuple[List[str], Dict[str, float], float]:
    """Extract HAMMER-compatible inputs from CMA file.

    Args:
        cma_path: Path to cma.json. Uses default if None.

    Returns:
        Tuple of (tickers, expected_returns_dict, risk_free_rate)
    """
    if cma_path is None:
        cma_path = os.path.join(DATA_DIR, "cma.json")

    with open(cma_path) as f:
        cma = json.load(f)

    tickers = cma["tickers"]
    exp_returns = cma["exp_returns"]
    rf = cma.get("rf", 0.02)

    # Create dict mapping ticker to expected return
    returns_dict = dict(zip(tickers, exp_returns))

    return tickers, returns_dict, rf


def portfolio_weights_to_hammer_config(
    tickers: List[str],
    weights: Dict[str, float],
    benchmark: str = "SPY",
    initial_capital: float = 100000,
    start_date: date = None,
    end_date: date = None,
) -> PortfolioConfig:
    """Convert Portfolio Builder weights to HAMMER PortfolioConfig.

    Args:
        tickers: List of ticker symbols
        weights: Dictionary mapping ticker to weight (0-1)
        benchmark: Benchmark ticker symbol
        initial_capital: Starting capital for backtest
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        PortfolioConfig ready for HAMMER backtest
    """
    # Default to 10 years of history
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(end_date.year - 10, end_date.month, end_date.day)

    # Filter to only tickers with non-zero weights
    active_weights = {t: w for t, w in weights.items() if w > 0}
    active_tickers = list(active_weights.keys())

    return PortfolioConfig(
        tickers=active_tickers,
        target_weights=active_weights,
        benchmark=benchmark,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
    )


def create_strategy_config(
    mode: str = "hammer",
    drift_threshold: float = 0.05,
    frequency: str = None,
) -> StrategyConfig:
    """Create HAMMER StrategyConfig from UI inputs.

    Args:
        mode: Strategy mode ("buy_hold", "periodic", "drift", "hammer")
        drift_threshold: Drift threshold for drift/hammer modes (0-1)
        frequency: Rebalance frequency for periodic mode

    Returns:
        StrategyConfig
    """
    mode_map = {
        "buy_hold": StrategyMode.BUY_HOLD,
        "periodic": StrategyMode.PERIODIC,
        "drift": StrategyMode.DRIFT,
        "hammer": StrategyMode.HAMMER,
    }

    freq_map = {
        "monthly": RebalanceFrequency.MONTHLY,
        "quarterly": RebalanceFrequency.QUARTERLY,
        "annual": RebalanceFrequency.ANNUAL,
    }

    strategy_mode = mode_map.get(mode.lower(), StrategyMode.HAMMER)
    rebal_freq = freq_map.get(frequency.lower()) if frequency else None

    return StrategyConfig(
        mode=strategy_mode,
        rebalance_frequency=rebal_freq,
        drift_threshold=drift_threshold,
    )


def run_comparison_backtest(
    portfolio_config: PortfolioConfig,
    drift_threshold: float = 0.05,
) -> Tuple[BacktestResult, PerformanceMetrics]:
    """Run HAMMER backtest and compare against benchmark.

    Args:
        portfolio_config: Portfolio configuration
        drift_threshold: Drift threshold for HAMMER strategy

    Returns:
        Tuple of (hammer_result, benchmark_metrics)
    """
    # HAMMER strategy (drift + VIX gate)
    hammer_strategy = StrategyConfig(
        mode=StrategyMode.HAMMER,
        drift_threshold=drift_threshold,
    )

    # Run HAMMER backtest
    hammer_engine = BacktestEngine(portfolio_config, hammer_strategy)
    hammer_result = hammer_engine.run()

    # Compute benchmark metrics from the benchmark_nav in the result
    benchmark_metrics = compute_benchmark_metrics(hammer_result.benchmark_nav)

    return hammer_result, benchmark_metrics


def compute_benchmark_metrics(
    benchmark_nav: pd.Series,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """Compute performance metrics for a benchmark (buy-and-hold).

    Args:
        benchmark_nav: Benchmark NAV time series
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object for benchmark
    """
    # Benchmark is buy-and-hold, so no rebalancing metrics
    return compute_metrics(
        benchmark_nav,
        benchmark_nav,  # Benchmark vs itself (alpha/beta will be 0/1)
        risk_free_rate=risk_free_rate,
        rebalances_per_year=0,
        blocked_rebalances=0,
        partial_rebalances=0,
        total_turnover=0,
    )


def compute_result_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """Compute performance metrics from backtest result.

    Args:
        result: BacktestResult from HAMMER engine
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object
    """
    return compute_metrics(
        result.nav,
        result.benchmark_nav,
        risk_free_rate=risk_free_rate,
        rebalances_per_year=result.rebalances_per_year,
        blocked_rebalances=len(result.blocked_events),
        partial_rebalances=len(result.partial_events),
        total_turnover=result.total_turnover,
    )


def metrics_to_comparison_dict(
    hammer_metrics: PerformanceMetrics,
    benchmark_metrics: PerformanceMetrics,
) -> Dict[str, Dict[str, str]]:
    """Format metrics for side-by-side comparison table.

    Args:
        hammer_metrics: Metrics from HAMMER strategy
        benchmark_metrics: Metrics from benchmark (buy-and-hold)

    Returns:
        Dictionary with metric names as keys, containing HAMMER/Benchmark/Diff values
    """
    comparison = {}

    h = hammer_metrics
    b = benchmark_metrics

    rows = [
        ("Annualized Return", h.cagr, b.cagr, True),
        ("Volatility", h.volatility, b.volatility, False),  # Lower is better
        ("Sharpe Ratio", h.sharpe_ratio, b.sharpe_ratio, True),
        ("Sortino Ratio", h.sortino_ratio, b.sortino_ratio, True),
        ("Max Drawdown", h.max_drawdown, b.max_drawdown, False),  # Less negative is better
        ("Alpha vs Benchmark", h.alpha, None, True),  # Benchmark alpha is always 0
        ("Beta vs Benchmark", h.beta, None, None),  # Benchmark beta is always 1
        ("Total Turnover", h.total_turnover, b.total_turnover, False),
        ("Equity-Frozen Events", h.partial_rebalances, None, None),  # Only HAMMER has this
    ]

    for name, h_val, b_val, higher_better in rows:
        # Format HAMMER value
        if h_val is None:
            h_str = "N/A"
        elif name in ("Sharpe Ratio", "Sortino Ratio", "Beta vs Benchmark"):
            h_str = f"{h_val:.2f}"
        elif name == "Equity-Frozen Events":
            h_str = str(int(h_val))
        else:
            h_str = f"{h_val:.2%}"

        # Format Benchmark value
        if b_val is None:
            if name == "Alpha vs Benchmark":
                b_str = "0.00%"  # By definition
            elif name == "Beta vs Benchmark":
                b_str = "1.00"  # By definition
            elif name == "Equity-Frozen Events":
                b_str = "—"  # N/A for benchmark
            else:
                b_str = "N/A"
        elif name in ("Sharpe Ratio", "Sortino Ratio"):
            b_str = f"{b_val:.2f}"
        elif name == "Equity-Frozen Events":
            b_str = str(int(b_val))
        else:
            b_str = f"{b_val:.2%}"

        # Calculate difference
        if h_val is None or b_val is None:
            if name == "Alpha vs Benchmark" and h_val is not None:
                # Alpha IS the outperformance vs benchmark
                diff_str = f"{h_val:+.2%}" if h_val > 0 else f"{h_val:.2%}"
                if h_val > 0:
                    diff_str += " ✓"
            elif name == "Equity-Frozen Events" and h_val is not None:
                diff_str = f"{int(h_val)} events"
            else:
                diff_str = "—"
        else:
            diff = h_val - b_val

            # Format based on metric type
            if name in ("Sharpe Ratio", "Sortino Ratio"):
                diff_str = f"{diff:+.2f}"
            else:
                diff_str = f"{diff:+.2%}"

            # Add indicator if HAMMER is better
            if higher_better is True and diff > 0:
                diff_str += " ✓"
            elif higher_better is False and diff < 0:
                diff_str += " ✓"

        comparison[name] = {
            "HAMMER": h_str,
            "Benchmark": b_str,
            "Difference": diff_str,
        }

    return comparison


def classify_portfolio_assets(
    tickers: List[str]
) -> Dict[str, List[str]]:
    """Classify portfolio tickers by asset class.

    Args:
        tickers: List of ticker symbols

    Returns:
        Dictionary mapping asset class name to list of tickers
    """
    result = {
        "Equity": [],
        "Fixed Income": [],
        "Alternatives": [],
        "Cash": [],
    }

    class_name_map = {
        AssetClass.EQUITY: "Equity",
        AssetClass.FIXED_INCOME: "Fixed Income",
        AssetClass.ALTERNATIVES: "Alternatives",
        AssetClass.CASH: "Cash",
    }

    for ticker in tickers:
        asset_class = get_asset_class(ticker)
        class_name = class_name_map[asset_class]
        result[class_name].append(ticker)

    return result


def save_backtest_result(
    result: BacktestResult,
    name: str,
) -> str:
    """Save backtest result to cache.

    Args:
        result: BacktestResult to save
        name: Name for the result file

    Returns:
        Path to saved file
    """
    ensure_dirs()

    # Export to DataFrame and save as CSV
    df = result.to_dataframe()
    filename = f"{name}_{date.today().isoformat()}.csv"
    filepath = os.path.join(BACKTEST_CACHE_DIR, filename)
    df.to_csv(filepath)

    return filepath


def generate_client_summary(
    hammer_result: BacktestResult,
    hammer_metrics: PerformanceMetrics,
    benchmark_metrics: PerformanceMetrics,
    benchmark_name: str = "Benchmark",
    client_name: str = "Client",
) -> str:
    """Generate client-facing summary text.

    Args:
        hammer_result: HAMMER backtest result
        hammer_metrics: HAMMER performance metrics
        benchmark_metrics: Benchmark performance metrics
        benchmark_name: Name of the benchmark (e.g., "SPY")
        client_name: Client name for personalization

    Returns:
        Formatted summary text
    """
    # Calculate key differences vs benchmark
    return_diff = hammer_metrics.cagr - benchmark_metrics.cagr
    sharpe_diff = hammer_metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio
    dd_diff = hammer_metrics.max_drawdown - benchmark_metrics.max_drawdown  # Less negative = better
    vol_diff = hammer_metrics.volatility - benchmark_metrics.volatility

    partial_count = len(hammer_result.partial_events)
    blocked_count = len(hammer_result.blocked_events)
    years = (hammer_result.effective_end - hammer_result.effective_start).days / 365.25

    summary = f"""
PORTFOLIO ANALYSIS SUMMARY
==========================
Client: {client_name}
Analysis Date: {date.today().strftime('%B %d, %Y')}
Backtest Period: {hammer_result.effective_start} to {hammer_result.effective_end} ({years:.1f} years)
Benchmark: {benchmark_name}

STRATEGY: HAMMER Rebalancing

The HAMMER strategy uses VIX term structure analysis to avoid rebalancing
during market stress periods. When the VIX curve inverts (indicating market
panic), HAMMER freezes intra-equity trades while still allowing asset
allocation adjustments.

KEY FINDINGS vs {benchmark_name}:
"""

    findings = []

    if return_diff > 0:
        findings.append(f"• Outperformed {benchmark_name} by {return_diff:.2%} annually")
    elif return_diff < 0:
        findings.append(f"• Underperformed {benchmark_name} by {abs(return_diff):.2%} (may have lower risk)")

    if sharpe_diff > 0:
        findings.append(f"• Better risk-adjusted returns (Sharpe +{sharpe_diff:.2f} vs benchmark)")

    if dd_diff > 0:  # Less negative = improvement
        findings.append(f"• Smaller maximum drawdown by {abs(dd_diff):.2%}")

    if vol_diff < 0:
        findings.append(f"• Lower volatility by {abs(vol_diff):.2%}")

    if partial_count > 0 or blocked_count > 0:
        findings.append(f"• Protected portfolio during {partial_count + blocked_count} market stress events")

    if hammer_metrics.alpha is not None and hammer_metrics.alpha > 0:
        findings.append(f"• Generated {hammer_metrics.alpha:.2%} alpha vs benchmark")

    summary += "\n".join(findings)

    summary += f"""

PERFORMANCE COMPARISON:
                    HAMMER      {benchmark_name:^12}   Difference
─────────────────────────────────────────────────────────────
Annualized Return   {hammer_metrics.cagr:>7.2%}       {benchmark_metrics.cagr:>7.2%}        {return_diff:>+7.2%}
Volatility          {hammer_metrics.volatility:>7.2%}       {benchmark_metrics.volatility:>7.2%}        {vol_diff:>+7.2%}
Sharpe Ratio        {hammer_metrics.sharpe_ratio:>7.2f}         {benchmark_metrics.sharpe_ratio:>7.2f}          {sharpe_diff:>+7.2f}
Max Drawdown        {hammer_metrics.max_drawdown:>7.2%}       {benchmark_metrics.max_drawdown:>7.2%}        {dd_diff:>+7.2%}
"""

    if hammer_metrics.alpha is not None:
        summary += f"Alpha vs Benchmark  {hammer_metrics.alpha:>7.2%}         0.00%          {hammer_metrics.alpha:>+7.2%}\n"

    if hammer_metrics.beta is not None:
        summary += f"Beta vs Benchmark   {hammer_metrics.beta:>7.2f}          1.00           {hammer_metrics.beta - 1:>+7.2f}\n"

    summary += f"""
HAMMER ACTIVITY:
• Rebalances blocked during VIX inversion: {blocked_count}
• Partial rebalances (equity frozen): {partial_count}
• Total portfolio turnover: {hammer_metrics.total_turnover:.2%}
"""

    return summary

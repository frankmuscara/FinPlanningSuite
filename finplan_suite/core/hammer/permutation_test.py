"""Monte Carlo permutation test for validating HAMMER strategy alpha.

This module runs a permutation test to determine whether HAMMER's VIX-based
blocking strategy produces statistically significant alpha compared to
randomly blocking the same number of rebalancing dates.

Usage:
    python -m finplan_suite.core.hammer.permutation_test
"""

import os
import random
from datetime import date
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

from .portfolio import PortfolioConfig
from .strategies import StrategyConfig, StrategyMode
from .backtest import BacktestEngine, BacktestResult
from .metrics import compute_metrics, PerformanceMetrics


# Default configuration
DEFAULT_PORTFOLIO = {
    "COWZ": 0.35,
    "QQQ": 0.35,
    "VYMI": 0.15,
    "QLEIX": 0.10,
    "BOND": 0.05,
}
DEFAULT_START = date(2017, 1, 2)
DEFAULT_END = date(2026, 1, 21)
DEFAULT_DRIFT_THRESHOLD = 0.04  # ~26 blocked events over the period
DEFAULT_BENCHMARK = "SPY"
DEFAULT_ITERATIONS = 10000


@dataclass
class PermutationResult:
    """Results from a single permutation iteration."""
    cagr: float
    sharpe: float
    turnover: float
    blocked_dates: Set[date]


@dataclass
class PermutationTestResults:
    """Full results from the permutation test."""
    # HAMMER actual results
    hammer_cagr: float
    hammer_sharpe: float
    hammer_turnover: float
    hammer_blocked_dates: List[date]

    # Distribution statistics
    cagr_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    turnover_distribution: np.ndarray

    # Statistical measures
    cagr_percentile: float
    sharpe_percentile: float
    turnover_percentile: float

    cagr_pvalue: float
    sharpe_pvalue: float
    turnover_pvalue: float

    # All rebalance trigger dates
    all_trigger_dates: List[date]
    n_iterations: int


def run_hammer_baseline(
    portfolio_weights: Dict[str, float],
    start_date: date,
    end_date: date,
    drift_threshold: float,
    benchmark: str,
    initial_capital: float = 100000,
) -> Tuple[PerformanceMetrics, List[date], List[date], pd.DataFrame, pd.Series]:
    """Run HAMMER baseline to get actual results and all trigger dates.

    This runs a custom simulation that tracks ALL days where rebalancing
    was triggered (drift > threshold), not just unique events.

    Returns:
        Tuple of (metrics, all_trigger_dates, hammer_blocked_dates, prices, vix_slope)
    """
    from .data import fetch_prices, validate_prices
    from .vix import fetch_vix_data, is_vix_blocked
    from .portfolio import Position, calculate_turnover

    tickers = list(portfolio_weights.keys())

    # Fetch price data
    all_tickers = tickers + [benchmark]
    prices, coverage = fetch_prices(all_tickers, start_date, end_date)

    issues = validate_prices(prices)
    if issues["errors"]:
        raise ValueError(f"Data errors: {issues['errors']}")

    # Fetch VIX data
    _, _, vix_slope = fetch_vix_data(start_date, end_date)
    vix_slope = vix_slope.reindex(prices.index, method="ffill")

    # Run simulation tracking ALL trigger dates
    position = Position()
    all_trigger_dates = []
    hammer_blocked_dates = []
    nav_history = []
    total_turnover = 0.0
    rebalance_count = 0

    benchmark_start_price = prices[benchmark].iloc[0]
    benchmark_shares = initial_capital / benchmark_start_price

    for i, current_date in enumerate(prices.index):
        current_prices = prices.loc[current_date].to_dict()
        current_date_val = current_date.date() if hasattr(current_date, "date") else current_date
        is_first_day = (i == 0)

        if is_first_day:
            portfolio_value = initial_capital
            position = position.rebalance_to(portfolio_weights, current_prices, portfolio_value)
            nav_history.append(position.value(current_prices))
            continue

        portfolio_value = position.value(current_prices)
        current_drift = position.max_drift(current_prices, portfolio_weights)

        should_rebalance = current_drift > drift_threshold

        if should_rebalance:
            all_trigger_dates.append(current_date_val)
            current_vix_slope = vix_slope.loc[current_date]
            vix_blocked = is_vix_blocked(current_vix_slope)

            if vix_blocked:
                hammer_blocked_dates.append(current_date_val)
            else:
                old_weights = position.weights(current_prices)
                position = position.rebalance_to(portfolio_weights, current_prices, portfolio_value)
                total_turnover += calculate_turnover(old_weights, portfolio_weights)
                rebalance_count += 1
                portfolio_value = position.value(current_prices)

        nav_history.append(portfolio_value)

    # Build NAV series
    nav = pd.Series(nav_history, index=prices.index, name="nav")
    benchmark_nav = benchmark_shares * prices[benchmark]

    # Compute metrics
    n_years = len(nav) / 252
    metrics = compute_metrics(
        nav,
        benchmark_nav,
        rebalances_per_year=rebalance_count / n_years if n_years > 0 else 0,
        blocked_rebalances=len(hammer_blocked_dates),
        total_turnover=total_turnover,
    )

    return metrics, all_trigger_dates, hammer_blocked_dates, prices, vix_slope


def run_permutation_iteration(
    portfolio_weights: Dict[str, float],
    drift_threshold: float,
    benchmark: str,
    blocked_dates: Set[date],
    prices: pd.DataFrame,
    initial_capital: float = 100000,
) -> PermutationResult:
    """Run a single permutation with specified blocked dates.

    Args:
        portfolio_weights: Target weights
        drift_threshold: Drift threshold for rebalancing
        benchmark: Benchmark ticker
        blocked_dates: Set of dates to block
        prices: Pre-fetched price data
        initial_capital: Starting capital

    Returns:
        PermutationResult with metrics
    """
    from .portfolio import Position, calculate_turnover

    position = Position()
    nav_history = []
    total_turnover = 0.0
    rebalance_count = 0
    actual_blocked = 0

    benchmark_start_price = prices[benchmark].iloc[0]
    benchmark_shares = initial_capital / benchmark_start_price

    for i, current_date in enumerate(prices.index):
        current_prices = prices.loc[current_date].to_dict()
        current_date_val = current_date.date() if hasattr(current_date, "date") else current_date
        is_first_day = (i == 0)

        if is_first_day:
            portfolio_value = initial_capital
            position = position.rebalance_to(portfolio_weights, current_prices, portfolio_value)
            nav_history.append(position.value(current_prices))
            continue

        portfolio_value = position.value(current_prices)
        current_drift = position.max_drift(current_prices, portfolio_weights)

        should_rebalance = current_drift > drift_threshold

        if should_rebalance:
            # Check if this date is in blocked_dates
            if current_date_val in blocked_dates:
                actual_blocked += 1
            else:
                old_weights = position.weights(current_prices)
                position = position.rebalance_to(portfolio_weights, current_prices, portfolio_value)
                total_turnover += calculate_turnover(old_weights, portfolio_weights)
                rebalance_count += 1
                portfolio_value = position.value(current_prices)

        nav_history.append(portfolio_value)

    # Build NAV series
    nav = pd.Series(nav_history, index=prices.index, name="nav")
    benchmark_nav = benchmark_shares * prices[benchmark]

    # Compute metrics
    n_years = len(nav) / 252
    metrics = compute_metrics(
        nav,
        benchmark_nav,
        rebalances_per_year=rebalance_count / n_years if n_years > 0 else 0,
        blocked_rebalances=actual_blocked,
        total_turnover=total_turnover,
    )

    return PermutationResult(
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        turnover=metrics.total_turnover,
        blocked_dates=blocked_dates,
    )


def run_permutation_test(
    portfolio_weights: Dict[str, float] = None,
    start_date: date = None,
    end_date: date = None,
    drift_threshold: float = None,
    benchmark: str = None,
    n_iterations: int = None,
    random_seed: int = 42,
    show_progress: bool = True,
) -> PermutationTestResults:
    """Run the full Monte Carlo permutation test.

    Args:
        portfolio_weights: Target portfolio weights
        start_date: Backtest start date
        end_date: Backtest end date
        drift_threshold: Drift threshold for rebalancing
        benchmark: Benchmark ticker
        n_iterations: Number of permutation iterations
        random_seed: Random seed for reproducibility
        show_progress: Whether to show progress bar

    Returns:
        PermutationTestResults with all statistics
    """
    # Apply defaults
    portfolio_weights = portfolio_weights or DEFAULT_PORTFOLIO
    start_date = start_date or DEFAULT_START
    end_date = end_date or DEFAULT_END
    drift_threshold = drift_threshold or DEFAULT_DRIFT_THRESHOLD
    benchmark = benchmark or DEFAULT_BENCHMARK
    n_iterations = n_iterations or DEFAULT_ITERATIONS

    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 60)
    print("HAMMER PERMUTATION TEST")
    print("=" * 60)
    print(f"Portfolio: GC HAMMER Max Growth")
    print(f"Period: {start_date} to {end_date}")
    print(f"Drift Threshold: {drift_threshold:.0%}")
    print(f"Benchmark: {benchmark}")
    print(f"Iterations: {n_iterations:,}")
    print()

    # Step 1: Run HAMMER baseline
    print("Running HAMMER baseline...")
    hammer_metrics, all_trigger_dates, hammer_blocked_dates, prices, vix_slope = run_hammer_baseline(
        portfolio_weights,
        start_date,
        end_date,
        drift_threshold,
        benchmark,
    )

    n_blocked = len(hammer_blocked_dates)

    print(f"HAMMER Results:")
    print(f"  CAGR: {hammer_metrics.cagr:.2%}")
    print(f"  Sharpe: {hammer_metrics.sharpe_ratio:.2f}")
    print(f"  Turnover: {hammer_metrics.total_turnover:.2%}")
    print(f"  Blocked Days: {n_blocked}")
    print(f"  Total Trigger Days: {len(all_trigger_dates)}")
    print()

    if n_blocked == 0:
        raise ValueError("HAMMER had 0 blocked days - cannot run permutation test")

    if len(all_trigger_dates) < n_blocked:
        raise ValueError(
            f"Not enough trigger dates ({len(all_trigger_dates)}) "
            f"to block {n_blocked} days"
        )

    # Step 2: Run permutation iterations
    print(f"Running {n_iterations:,} permutation iterations...")

    cagr_results = []
    sharpe_results = []
    turnover_results = []

    iterator = range(n_iterations)
    if show_progress:
        iterator = tqdm(iterator, desc="Permutations")

    for _ in iterator:
        # Randomly select n_blocked dates from all trigger dates
        random_blocked = set(random.sample(all_trigger_dates, n_blocked))

        perm_result = run_permutation_iteration(
            portfolio_weights,
            drift_threshold,
            benchmark,
            random_blocked,
            prices,
        )

        cagr_results.append(perm_result.cagr)
        sharpe_results.append(perm_result.sharpe)
        turnover_results.append(perm_result.turnover)

    # Convert to numpy arrays
    cagr_distribution = np.array(cagr_results)
    sharpe_distribution = np.array(sharpe_results)
    turnover_distribution = np.array(turnover_results)

    # Step 3: Calculate statistics
    # Percentile rank (what percentage of random results are BELOW HAMMER's result)
    cagr_percentile = (cagr_distribution < hammer_metrics.cagr).mean() * 100
    sharpe_percentile = (sharpe_distribution < hammer_metrics.sharpe_ratio).mean() * 100
    # For turnover, lower is better
    turnover_percentile = (turnover_distribution > hammer_metrics.total_turnover).mean() * 100

    # P-values (one-tailed: proportion that matched or exceeded HAMMER)
    # For CAGR/Sharpe: higher is better, so p-value = proportion >= HAMMER
    cagr_pvalue = (cagr_distribution >= hammer_metrics.cagr).mean()
    sharpe_pvalue = (sharpe_distribution >= hammer_metrics.sharpe_ratio).mean()
    # For turnover: lower is better, so p-value = proportion <= HAMMER
    turnover_pvalue = (turnover_distribution <= hammer_metrics.total_turnover).mean()

    return PermutationTestResults(
        hammer_cagr=hammer_metrics.cagr,
        hammer_sharpe=hammer_metrics.sharpe_ratio,
        hammer_turnover=hammer_metrics.total_turnover,
        hammer_blocked_dates=hammer_blocked_dates,
        cagr_distribution=cagr_distribution,
        sharpe_distribution=sharpe_distribution,
        turnover_distribution=turnover_distribution,
        cagr_percentile=cagr_percentile,
        sharpe_percentile=sharpe_percentile,
        turnover_percentile=turnover_percentile,
        cagr_pvalue=cagr_pvalue,
        sharpe_pvalue=sharpe_pvalue,
        turnover_pvalue=turnover_pvalue,
        all_trigger_dates=all_trigger_dates,
        n_iterations=n_iterations,
    )


def plot_results(
    results: PermutationTestResults,
    output_dir: str,
) -> List[str]:
    """Generate histogram plots for each metric.

    Args:
        results: Permutation test results
        output_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    metrics = [
        ("CAGR", results.cagr_distribution, results.hammer_cagr,
         results.cagr_percentile, results.cagr_pvalue, True),
        ("Sharpe Ratio", results.sharpe_distribution, results.hammer_sharpe,
         results.sharpe_percentile, results.sharpe_pvalue, True),
        ("Total Turnover", results.turnover_distribution, results.hammer_turnover,
         results.turnover_percentile, results.turnover_pvalue, False),
    ]

    for name, distribution, hammer_val, percentile, pvalue, higher_better in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        n, bins, patches = ax.hist(
            distribution,
            bins=50,
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5,
        )

        # Mark HAMMER's actual value
        ax.axvline(
            hammer_val,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'HAMMER Actual: {hammer_val:.4f}',
        )

        # Add shaded region showing where HAMMER outperforms
        if higher_better:
            ax.axvspan(hammer_val, distribution.max() * 1.1, alpha=0.1, color='red')
        else:
            ax.axvspan(distribution.min() * 0.9, hammer_val, alpha=0.1, color='red')

        # Labels and title
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(
            f'{name} Distribution: Random Blocking vs HAMMER\n'
            f'(n={results.n_iterations:,} permutations)',
            fontsize=14,
        )

        # Add statistics text box
        direction = "higher" if higher_better else "lower"
        stats_text = (
            f'HAMMER: {hammer_val:.4f}\n'
            f'Random Mean: {distribution.mean():.4f}\n'
            f'Random Std: {distribution.std():.4f}\n'
            f'Percentile: {percentile:.1f}%\n'
            f'p-value ({direction} is better): {pvalue:.4f}'
        )

        # Position text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=props,
        )

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis for percentages if applicable
        if name in ("CAGR", "Total Turnover"):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

        plt.tight_layout()

        # Save
        filename = f"permutation_{name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        saved_files.append(filepath)
        plt.close(fig)

        print(f"Saved: {filepath}")

    # Create combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, distribution, hammer_val, percentile, pvalue, higher_better) in zip(axes, metrics):
        ax.hist(distribution, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(hammer_val, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}\np={pvalue:.4f}')
        ax.grid(True, alpha=0.3)

        if name in ("CAGR", "Total Turnover"):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    fig.suptitle(
        f'HAMMER Permutation Test Results (n={results.n_iterations:,})',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()

    combined_path = os.path.join(output_dir, "permutation_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    saved_files.append(combined_path)
    plt.close(fig)

    print(f"Saved: {combined_path}")

    return saved_files


def print_summary_table(results: PermutationTestResults):
    """Print summary table to console."""
    print()
    print("=" * 80)
    print("PERMUTATION TEST SUMMARY")
    print("=" * 80)
    print()

    # Header
    print(f"{'Metric':<20} {'HAMMER':>12} {'Dist Mean':>12} {'Dist Std':>12} {'Percentile':>12} {'p-value':>12}")
    print("-" * 80)

    # CAGR
    print(f"{'CAGR':<20} {results.hammer_cagr:>11.2%} {results.cagr_distribution.mean():>11.2%} "
          f"{results.cagr_distribution.std():>11.2%} {results.cagr_percentile:>11.1f}% {results.cagr_pvalue:>12.4f}")

    # Sharpe
    print(f"{'Sharpe Ratio':<20} {results.hammer_sharpe:>12.2f} {results.sharpe_distribution.mean():>12.2f} "
          f"{results.sharpe_distribution.std():>12.2f} {results.sharpe_percentile:>11.1f}% {results.sharpe_pvalue:>12.4f}")

    # Turnover
    print(f"{'Total Turnover':<20} {results.hammer_turnover:>11.2%} {results.turnover_distribution.mean():>11.2%} "
          f"{results.turnover_distribution.std():>11.2%} {results.turnover_percentile:>11.1f}% {results.turnover_pvalue:>12.4f}")

    print("-" * 80)
    print()

    # Interpretation
    print("INTERPRETATION:")
    print(f"  - HAMMER blocked {len(results.hammer_blocked_dates)} rebalancing events based on VIX")
    print(f"  - {results.n_iterations:,} random blockings of the same count were simulated")
    print()

    sig_level = 0.05

    if results.cagr_pvalue < sig_level:
        print(f"  * CAGR: HAMMER's result is STATISTICALLY SIGNIFICANT (p={results.cagr_pvalue:.4f} < {sig_level})")
        print(f"    Only {results.cagr_pvalue*100:.2f}% of random blockings achieved equal or higher CAGR")
    else:
        print(f"  * CAGR: Not statistically significant (p={results.cagr_pvalue:.4f} >= {sig_level})")

    if results.sharpe_pvalue < sig_level:
        print(f"  * Sharpe: HAMMER's result is STATISTICALLY SIGNIFICANT (p={results.sharpe_pvalue:.4f} < {sig_level})")
        print(f"    Only {results.sharpe_pvalue*100:.2f}% of random blockings achieved equal or higher Sharpe")
    else:
        print(f"  * Sharpe: Not statistically significant (p={results.sharpe_pvalue:.4f} >= {sig_level})")

    if results.turnover_pvalue < sig_level:
        print(f"  * Turnover: HAMMER's result is STATISTICALLY SIGNIFICANT (p={results.turnover_pvalue:.4f} < {sig_level})")
        print(f"    Only {results.turnover_pvalue*100:.2f}% of random blockings achieved equal or lower turnover")
    else:
        print(f"  * Turnover: Not statistically significant (p={results.turnover_pvalue:.4f} >= {sig_level})")

    print()
    print("=" * 80)


def save_results_csv(results: PermutationTestResults, output_dir: str) -> str:
    """Save results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    # Summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['CAGR', 'Sharpe Ratio', 'Total Turnover'],
        'HAMMER_Actual': [results.hammer_cagr, results.hammer_sharpe, results.hammer_turnover],
        'Distribution_Mean': [
            results.cagr_distribution.mean(),
            results.sharpe_distribution.mean(),
            results.turnover_distribution.mean(),
        ],
        'Distribution_Std': [
            results.cagr_distribution.std(),
            results.sharpe_distribution.std(),
            results.turnover_distribution.std(),
        ],
        'Percentile': [results.cagr_percentile, results.sharpe_percentile, results.turnover_percentile],
        'P_Value': [results.cagr_pvalue, results.sharpe_pvalue, results.turnover_pvalue],
    })

    summary_path = os.path.join(output_dir, "permutation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Full distribution data
    dist_df = pd.DataFrame({
        'CAGR': results.cagr_distribution,
        'Sharpe': results.sharpe_distribution,
        'Turnover': results.turnover_distribution,
    })

    dist_path = os.path.join(output_dir, "permutation_distributions.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"Saved: {dist_path}")

    # HAMMER blocked dates
    blocked_df = pd.DataFrame({
        'Blocked_Date': results.hammer_blocked_dates,
    })
    blocked_path = os.path.join(output_dir, "hammer_blocked_dates.csv")
    blocked_df.to_csv(blocked_path, index=False)
    print(f"Saved: {blocked_path}")

    return summary_path


def main():
    """Run the full permutation test."""
    # Output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "results",
        "permutation_test"
    )

    # Run test
    results = run_permutation_test(
        portfolio_weights=DEFAULT_PORTFOLIO,
        start_date=DEFAULT_START,
        end_date=DEFAULT_END,
        drift_threshold=DEFAULT_DRIFT_THRESHOLD,
        benchmark=DEFAULT_BENCHMARK,
        n_iterations=DEFAULT_ITERATIONS,
        random_seed=42,
    )

    # Print summary
    print_summary_table(results)

    # Save plots
    print("\nGenerating plots...")
    plot_results(results, output_dir)

    # Save CSV
    print("\nSaving data...")
    save_results_csv(results, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""Hansen-Hodrick predictive regression tests for the HAMMER VIX-gating signal.

Tests whether VIX term structure inversion predicts subsequent cross-sectional
spread variance collapse -- the theoretical mechanism justifying HAMMER's
rebalance blocking.

Key idea: During VIX inversions, asset correlations spike and return spreads
compress. Rebalancing during this period is counterproductive because you'd
be selling into a market where relative value signals have collapsed.

Uses Hansen-Hodrick (1980) standard errors to correct for overlapping
observation periods in multi-day-horizon predictive regressions.

Usage:
    python -m finplan_suite.core.hammer.signal_test
"""

import os
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from .data import fetch_prices
from .vix import fetch_vix_data


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def build_regression_data(
    tickers: List[str],
    start_date: date,
    end_date: date,
    horizon: int = 21,
) -> pd.DataFrame:
    """Build aligned dataset for predictive regressions.

    Args:
        tickers: Portfolio constituent tickers
        start_date: Start date
        end_date: End date
        horizon: Forecast horizon in trading days (default 21 = ~1 month)

    Returns:
        DataFrame with columns:
            vix_slope       - VIX3M - VIX (continuous signal)
            inversion       - 1 if vix_slope < 0 (binary signal)
            vix_level       - VIX level (control)
            future_rv       - Realized variance of equal-weighted portfolio, t+1 to t+h
            future_spread   - Cross-sectional spread variance, t+1 to t+h
            lagged_rv       - Realized variance, t-h+1 to t (control)
            lagged_spread   - Lagged spread variance (control)
    """
    # Fetch price data
    prices, _ = fetch_prices(tickers, start_date, end_date)

    # Fetch VIX data
    vix, vix3m, vix_slope = fetch_vix_data(start_date, end_date)

    # Compute daily returns for each ticker
    returns = prices.pct_change().dropna()

    # Equal-weighted portfolio return
    portfolio_returns = returns.mean(axis=1)

    # Cross-sectional return spread: std of returns across tickers each day
    # This measures how differently the assets are moving
    spread = returns.std(axis=1)

    # Align VIX slope to returns index
    vix_slope_aligned = vix_slope.reindex(returns.index, method="ffill")
    vix_aligned = vix.reindex(returns.index, method="ffill")

    # Build regression DataFrame
    reg = pd.DataFrame(index=returns.index)

    # Predictors (measured at time t)
    reg["vix_slope"] = vix_slope_aligned
    reg["inversion"] = (vix_slope_aligned < 0).astype(float)
    reg["vix_level"] = vix_aligned

    # Future realized variance: sum of squared portfolio returns, t+1 to t+h
    reg["future_rv"] = (
        portfolio_returns.pow(2)
        .rolling(window=horizon)
        .sum()
        .shift(-horizon)
    )

    # Future cross-sectional spread variance: sum of squared spreads, t+1 to t+h
    reg["future_spread"] = (
        spread.pow(2)
        .rolling(window=horizon)
        .sum()
        .shift(-horizon)
    )

    # Lagged realized variance (control)
    reg["lagged_rv"] = (
        portfolio_returns.pow(2)
        .rolling(window=horizon)
        .sum()
    )

    # Lagged spread variance (control)
    reg["lagged_spread"] = (
        spread.pow(2)
        .rolling(window=horizon)
        .sum()
    )

    reg = reg.dropna()
    return reg


# ---------------------------------------------------------------------------
# Hansen-Hodrick regression
# ---------------------------------------------------------------------------

@dataclass
class RegressionSpec:
    """One regression specification and its results."""
    name: str
    description: str
    dep_var: str
    indep_vars: List[str]
    horizon: int

    # Filled after estimation
    beta: Optional[np.ndarray] = None
    se_ols: Optional[np.ndarray] = None
    se_hh: Optional[np.ndarray] = None
    se_nw: Optional[np.ndarray] = None
    t_ols: Optional[np.ndarray] = None
    t_hh: Optional[np.ndarray] = None
    t_nw: Optional[np.ndarray] = None
    p_ols: Optional[np.ndarray] = None
    p_hh: Optional[np.ndarray] = None
    p_nw: Optional[np.ndarray] = None
    r_squared: Optional[float] = None
    n_obs: Optional[int] = None
    var_names: List[str] = field(default_factory=list)


def run_regression(
    reg_data: pd.DataFrame,
    spec: RegressionSpec,
) -> RegressionSpec:
    """Run OLS regression with OLS, Hansen-Hodrick, and Newey-West SEs.

    Args:
        reg_data: Regression DataFrame from build_regression_data
        spec: Regression specification

    Returns:
        Updated spec with estimation results
    """
    y = reg_data[spec.dep_var].values
    X_raw = reg_data[spec.indep_vars].values
    X = sm.add_constant(X_raw)

    n_lags = spec.horizon - 1
    spec.var_names = ["const"] + spec.indep_vars
    spec.n_obs = len(y)

    # OLS (naive, ignoring overlap)
    ols_result = sm.OLS(y, X).fit()
    spec.beta = ols_result.params
    spec.se_ols = ols_result.bse
    spec.t_ols = ols_result.tvalues
    spec.p_ols = ols_result.pvalues
    spec.r_squared = ols_result.rsquared

    # Hansen-Hodrick (uniform/truncated kernel)
    hh_result = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"kernel": "uniform", "maxlags": n_lags},
        use_t=True,
    )
    spec.se_hh = hh_result.bse
    spec.t_hh = hh_result.tvalues
    spec.p_hh = hh_result.pvalues

    # Newey-West (Bartlett kernel, for robustness comparison)
    nw_result = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"kernel": "bartlett", "maxlags": n_lags},
        use_t=True,
    )
    spec.se_nw = nw_result.bse
    spec.t_nw = nw_result.tvalues
    spec.p_nw = nw_result.pvalues

    return spec


# ---------------------------------------------------------------------------
# Full test suite
# ---------------------------------------------------------------------------

@dataclass
class SignalTestResults:
    """Complete results from the signal test suite."""
    specs: List[RegressionSpec]
    reg_data: pd.DataFrame
    horizon: int
    tickers: List[str]
    start_date: date
    end_date: date


def run_signal_tests(
    tickers: List[str],
    start_date: date,
    end_date: date,
    horizon: int = 21,
) -> SignalTestResults:
    """Run the full battery of Hansen-Hodrick predictive regressions.

    Args:
        tickers: Portfolio constituent tickers
        start_date: Start date
        end_date: End date
        horizon: Forecast horizon in trading days

    Returns:
        SignalTestResults with all regression results
    """
    print("=" * 70)
    print("HAMMER SIGNAL TEST: Hansen-Hodrick Predictive Regressions")
    print("=" * 70)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Forecast Horizon: {horizon} trading days (~{horizon/21:.0f} month)")
    print()

    # Build data
    print("Fetching data and constructing variables...")
    reg_data = build_regression_data(tickers, start_date, end_date, horizon)
    print(f"Observations: {len(reg_data)}")
    print(f"Date range: {reg_data.index[0].date()} to {reg_data.index[-1].date()}")
    print()

    # Define specifications
    specs = [
        RegressionSpec(
            name="slope_rv",
            description="VIX Slope -> Future Realized Variance",
            dep_var="future_rv",
            indep_vars=["vix_slope"],
            horizon=horizon,
        ),
        RegressionSpec(
            name="inversion_rv",
            description="VIX Inversion (binary) -> Future Realized Variance",
            dep_var="future_rv",
            indep_vars=["inversion"],
            horizon=horizon,
        ),
        RegressionSpec(
            name="slope_spread",
            description="VIX Slope -> Future Spread Variance",
            dep_var="future_spread",
            indep_vars=["vix_slope"],
            horizon=horizon,
        ),
        RegressionSpec(
            name="inversion_spread",
            description="VIX Inversion (binary) -> Future Spread Variance",
            dep_var="future_spread",
            indep_vars=["inversion"],
            horizon=horizon,
        ),
        RegressionSpec(
            name="full_rv",
            description="Full Model -> Future Realized Variance (with controls)",
            dep_var="future_rv",
            indep_vars=["vix_slope", "vix_level", "lagged_rv"],
            horizon=horizon,
        ),
        RegressionSpec(
            name="full_spread",
            description="Full Model -> Future Spread Variance (with controls)",
            dep_var="future_spread",
            indep_vars=["vix_slope", "vix_level", "lagged_spread"],
            horizon=horizon,
        ),
    ]

    # Run regressions
    for spec in specs:
        print(f"Running: {spec.description}...")
        run_regression(reg_data, spec)

    return SignalTestResults(
        specs=specs,
        reg_data=reg_data,
        horizon=horizon,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: SignalTestResults):
    """Print comprehensive results table."""
    print()
    print("=" * 90)
    print("RESULTS: Hansen-Hodrick Predictive Regressions")
    print(f"Forecast horizon h = {results.horizon} days | "
          f"HH lags = {results.horizon - 1} | "
          f"N = {results.specs[0].n_obs}")
    print("=" * 90)

    for spec in results.specs:
        print(f"\n--- {spec.description} ---")
        print(f"{'Variable':<16} {'Beta':>10} {'SE(OLS)':>10} {'SE(HH)':>10} {'SE(NW)':>10} "
              f"{'t(HH)':>8} {'p(HH)':>8} {'SE ratio':>9}")
        print("-" * 90)

        for i, var in enumerate(spec.var_names):
            ratio = spec.se_hh[i] / spec.se_ols[i] if spec.se_ols[i] > 0 else float("inf")
            sig = ""
            if spec.p_hh[i] < 0.01:
                sig = "***"
            elif spec.p_hh[i] < 0.05:
                sig = "**"
            elif spec.p_hh[i] < 0.10:
                sig = "*"

            print(f"{var:<16} {spec.beta[i]:>10.6f} {spec.se_ols[i]:>10.6f} "
                  f"{spec.se_hh[i]:>10.6f} {spec.se_nw[i]:>10.6f} "
                  f"{spec.t_hh[i]:>7.2f}{sig:<1} {spec.p_hh[i]:>8.4f} {ratio:>8.1f}x")

        print(f"  R-squared: {spec.r_squared:.4f}")

    print()
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10 (Hansen-Hodrick)")
    print(f"SE ratio = SE(HH) / SE(OLS) — shows how much OLS understates uncertainty")
    print()

    # Interpretation
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    # Find the key specs
    slope_spread = next((s for s in results.specs if s.name == "slope_spread"), None)
    inversion_spread = next((s for s in results.specs if s.name == "inversion_spread"), None)
    full_spread = next((s for s in results.specs if s.name == "full_spread"), None)

    if slope_spread:
        # vix_slope coefficient is index 1 (after const)
        beta = slope_spread.beta[1]
        p = slope_spread.p_hh[1]
        print(f"\nVIX Slope -> Spread Variance:")
        if beta < 0:
            print(f"  Beta = {beta:.6f} (negative): Higher VIX slope (contango) predicts")
            print(f"  LOWER future spread variance. During inversion (slope<0), spread")
            print(f"  variance is HIGHER. p={p:.4f} (HH)")
        else:
            print(f"  Beta = {beta:.6f} (positive): Higher VIX slope predicts HIGHER")
            print(f"  future spread variance. During inversion, spread variance")
            print(f"  COMPRESSES (supporting HAMMER). p={p:.4f} (HH)")

    if inversion_spread:
        beta = inversion_spread.beta[1]
        p = inversion_spread.p_hh[1]
        print(f"\nVIX Inversion (binary) -> Spread Variance:")
        if beta > 0:
            print(f"  Beta = {beta:.6f} (positive): Inversion periods are followed by")
            print(f"  HIGHER spread variance. p={p:.4f} (HH)")
        else:
            print(f"  Beta = {beta:.6f} (negative): Inversion periods are followed by")
            print(f"  LOWER spread variance (spreads compress). p={p:.4f} (HH)")

        if p < 0.05:
            print(f"  ** STATISTICALLY SIGNIFICANT at 5% with Hansen-Hodrick SEs **")
        elif p < 0.10:
            print(f"  * Marginally significant at 10% with Hansen-Hodrick SEs *")
        else:
            print(f"  Not significant at conventional levels with Hansen-Hodrick SEs")

    if full_spread:
        beta = full_spread.beta[1]  # vix_slope coefficient
        p = full_spread.p_hh[1]
        ratio = full_spread.se_hh[1] / full_spread.se_ols[1]
        print(f"\nFull Model (with controls) -> Spread Variance:")
        print(f"  VIX slope beta = {beta:.6f}, p(HH) = {p:.4f}")
        print(f"  SE inflation: {ratio:.1f}x (HH vs OLS)")
        print(f"  This shows how much naive OLS overstates significance")

    print()


def plot_results(results: SignalTestResults, output_dir: str) -> List[str]:
    """Generate diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    reg = results.reg_data

    # Plot 1: VIX slope vs future spread variance (scatter + regression line)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(reg["vix_slope"], reg["future_spread"], alpha=0.15, s=5, color="steelblue")
    # Add regression line
    slope_coef = np.polyfit(reg["vix_slope"], reg["future_spread"], 1)
    x_line = np.linspace(reg["vix_slope"].min(), reg["vix_slope"].max(), 100)
    ax.plot(x_line, np.polyval(slope_coef, x_line), color="red", linewidth=2)
    ax.set_xlabel("VIX Slope (VIX3M - VIX)")
    ax.set_ylabel(f"Future {results.horizon}d Spread Variance")
    ax.set_title("VIX Slope vs Future Spread Variance")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(reg["vix_slope"], reg["future_rv"], alpha=0.15, s=5, color="darkorange")
    slope_coef_rv = np.polyfit(reg["vix_slope"], reg["future_rv"], 1)
    ax.plot(x_line, np.polyval(slope_coef_rv, x_line), color="red", linewidth=2)
    ax.set_xlabel("VIX Slope (VIX3M - VIX)")
    ax.set_ylabel(f"Future {results.horizon}d Realized Variance")
    ax.set_title("VIX Slope vs Future Realized Variance")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle("HAMMER Signal: Predictive Relationships", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "signal_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    # Plot 2: SE comparison across specifications
    fig, ax = plt.subplots(figsize=(12, 6))

    spec_names = []
    ratios_hh = []
    ratios_nw = []

    for spec in results.specs:
        # Use the first non-constant coefficient
        idx = 1
        ratio_hh = spec.se_hh[idx] / spec.se_ols[idx]
        ratio_nw = spec.se_nw[idx] / spec.se_ols[idx]
        spec_names.append(spec.name)
        ratios_hh.append(ratio_hh)
        ratios_nw.append(ratio_nw)

    x = np.arange(len(spec_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ratios_hh, width, label="Hansen-Hodrick / OLS", color="steelblue")
    bars2 = ax.bar(x + width / 2, ratios_nw, width, label="Newey-West / OLS", color="darkorange")

    ax.set_xlabel("Regression Specification")
    ax.set_ylabel("SE Ratio (HAC / OLS)")
    ax.set_title(f"Standard Error Inflation: HAC vs OLS\n(h={results.horizon} days, "
                 f"overlap causes OLS to understate uncertainty)")
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names, rotation=30, ha="right")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="OLS baseline (1.0x)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}x", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}x", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "se_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    # Plot 3: Time series of VIX slope with spread variance
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(reg.index, reg["vix_slope"], color="steelblue", alpha=0.7, linewidth=0.8)
    ax1.fill_between(reg.index, 0, reg["vix_slope"],
                      where=reg["vix_slope"] < 0, color="red", alpha=0.3, label="VIX Inverted")
    ax1.set_ylabel("VIX Slope (VIX3M - VIX)", color="steelblue")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(reg.index, reg["future_spread"], color="darkorange", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel(f"Future {results.horizon}d Spread Variance", color="darkorange")

    ax1.set_xlabel("Date")
    ax1.set_title("VIX Term Structure vs Future Cross-Sectional Spread Variance")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "signal_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    for p in saved:
        print(f"Saved: {p}")

    return saved


def save_results_csv(results: SignalTestResults, output_dir: str):
    """Save regression results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for spec in results.specs:
        for i, var in enumerate(spec.var_names):
            rows.append({
                "Specification": spec.name,
                "Description": spec.description,
                "Variable": var,
                "Beta": spec.beta[i],
                "SE_OLS": spec.se_ols[i],
                "SE_HH": spec.se_hh[i],
                "SE_NW": spec.se_nw[i],
                "t_OLS": spec.t_ols[i],
                "t_HH": spec.t_hh[i],
                "t_NW": spec.t_nw[i],
                "p_OLS": spec.p_ols[i],
                "p_HH": spec.p_hh[i],
                "p_NW": spec.p_nw[i],
                "SE_ratio_HH_OLS": spec.se_hh[i] / spec.se_ols[i] if spec.se_ols[i] > 0 else None,
                "R_squared": spec.r_squared,
                "N_obs": spec.n_obs,
                "Horizon": spec.horizon,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "hansen_hodrick_results.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run signal tests for both portfolios."""
    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "results",
        "signal_test",
    )

    start = date(2017, 1, 2)
    end = date(2026, 1, 21)

    # Test with GC HAMMER Max Growth constituents
    tickers_maxgrowth = ["COWZ", "QQQ", "VYMI", "QLEIX", "BOND"]
    print("\n" + "#" * 70)
    print("# PORTFOLIO 1: GC HAMMER Max Growth")
    print("#" * 70)
    results1 = run_signal_tests(tickers_maxgrowth, start, end, horizon=21)
    print_results(results1)
    output_dir1 = os.path.join(output_base, "max_growth")
    plot_results(results1, output_dir1)
    save_results_csv(results1, output_dir1)

    # Test with QQQ/COWZ/XLF portfolio
    tickers_equity = ["QQQ", "COWZ", "XLF"]
    print("\n" + "#" * 70)
    print("# PORTFOLIO 2: QQQ/COWZ/XLF")
    print("#" * 70)
    results2 = run_signal_tests(tickers_equity, start, end, horizon=21)
    print_results(results2)
    output_dir2 = os.path.join(output_base, "qqq_cowz_xlf")
    plot_results(results2, output_dir2)
    save_results_csv(results2, output_dir2)

    print(f"\nAll results saved to: {output_base}")


if __name__ == "__main__":
    main()

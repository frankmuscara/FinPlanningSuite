"""Data fetching and validation for HAMMER.

All price data uses TOTAL RETURN (adjusted close prices) which includes:
- Dividend reinvestment (prices adjusted backwards for dividend payments)
- Stock splits (prices adjusted for split ratios)

This means backtests reflect what an investor would actually earn, not just
price appreciation.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_prices(
    tickers: List[str],
    start_date: date,
    end_date: date,
    max_nan_fill: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[date, date]]]:
    """Fetch TOTAL RETURN prices (adjusted close) for multiple tickers.

    Uses yfinance with auto_adjust=True which:
    - Adjusts historical prices backwards to reflect dividend payments
    - Adjusts for stock splits automatically
    - Returns prices that capture total return, not just price return

    Example: A $100 stock that paid a $2 dividend will show previous prices
    adjusted down by ~2%, so holding shares through time captures total return.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        max_nan_fill: Maximum consecutive NaNs to forward-fill

    Returns:
        Tuple of (prices DataFrame, coverage dict mapping ticker to (first, last) dates)
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance required. Install with: pip install yfinance")

    # Add buffer for alignment
    buffer_start = start_date - timedelta(days=10)

    # Download data with auto_adjust=True for TOTAL RETURN
    # This adjusts prices for dividends and splits, giving us total return
    data = yf.download(
        tickers,
        start=buffer_start.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,  # IMPORTANT: Includes dividends in price history
    )

    if data.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    # Extract Close prices
    if len(tickers) == 1:
        prices = data[["Close"]].copy()
        prices.columns = tickers
    else:
        prices = data["Close"].copy()

    # Track coverage before filling
    coverage = {}
    for ticker in tickers:
        if ticker in prices.columns:
            valid_data = prices[ticker].dropna()
            if len(valid_data) > 0:
                coverage[ticker] = (
                    valid_data.index[0].date(),
                    valid_data.index[-1].date()
                )
            else:
                coverage[ticker] = (None, None)
        else:
            coverage[ticker] = (None, None)

    # Forward-fill NaNs (up to limit)
    prices = prices.ffill(limit=max_nan_fill)

    # Trim to requested date range
    prices = prices.loc[start_date.isoformat():end_date.isoformat()]

    # Validate no remaining NaNs in critical period
    remaining_nans = prices.isna().sum()
    if remaining_nans.any():
        problem_tickers = remaining_nans[remaining_nans > 0].index.tolist()
        # Fill remaining with last valid or drop
        prices = prices.ffill().bfill()

    return prices, coverage


def validate_prices(prices: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate price data for common issues.

    Returns dict with 'errors' and 'warnings' lists.
    """
    issues = {"errors": [], "warnings": []}

    # Check for negative prices
    if (prices < 0).any().any():
        neg_tickers = prices.columns[(prices < 0).any()].tolist()
        issues["errors"].append(f"Negative prices found: {neg_tickers}")

    # Check for zero prices
    if (prices == 0).any().any():
        zero_tickers = prices.columns[(prices == 0).any()].tolist()
        issues["warnings"].append(f"Zero prices found: {zero_tickers}")

    # Check for extreme moves (>50% in one day)
    returns = prices.pct_change()
    extreme_moves = (returns.abs() > 0.5).any()
    if extreme_moves.any():
        extreme_tickers = prices.columns[extreme_moves].tolist()
        issues["warnings"].append(f"Extreme daily moves (>50%): {extreme_tickers}")

    # Check data length
    if len(prices) < 20:
        issues["warnings"].append(f"Very short history: only {len(prices)} days")

    return issues


def get_common_date_range(
    coverage: Dict[str, Tuple[date, date]]
) -> Tuple[Optional[date], Optional[date]]:
    """Get the intersection of all tickers' date ranges.

    Args:
        coverage: Dict mapping ticker to (start, end) dates

    Returns:
        Tuple of (common_start, common_end) or (None, None) if no overlap
    """
    valid_ranges = [(s, e) for s, e in coverage.values() if s is not None]

    if not valid_ranges:
        return None, None

    common_start = max(r[0] for r in valid_ranges)
    common_end = min(r[1] for r in valid_ranges)

    if common_start > common_end:
        return None, None

    return common_start, common_end


def get_business_days(start_date: date, end_date: date) -> pd.DatetimeIndex:
    """Get business days between two dates."""
    return pd.bdate_range(start=start_date, end=end_date)


def align_to_business_days(
    series: pd.Series,
    start_date: date,
    end_date: date,
) -> pd.Series:
    """Align series to business days, forward-filling gaps."""
    bdays = get_business_days(start_date, end_date)
    aligned = series.reindex(bdays, method="ffill")
    return aligned

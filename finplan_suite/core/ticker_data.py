"""Ticker data fetching and caching for Portfolio Builder.

Provides a wrapper around yfinance with session-level caching to avoid
repeated API calls for the same ticker.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


@dataclass
class TickerInfo:
    """Container for ticker information."""
    ticker: str
    name: str
    expense_ratio: Optional[float]  # As decimal (0.0003 = 0.03%)
    dividend_yield: Optional[float]  # As decimal
    last_price: Optional[float]
    returns_1yr: Optional[float]  # Annualized
    returns_5yr: Optional[float]  # Annualized
    returns_10yr: Optional[float]  # Annualized
    is_valid: bool
    error_message: Optional[str] = None

    def expense_ratio_pct(self) -> str:
        """Format expense ratio as percentage string."""
        if self.expense_ratio is None:
            return "N/A"
        return f"{self.expense_ratio * 100:.2f}%"

    def dividend_yield_pct(self) -> str:
        """Format dividend yield as percentage string."""
        if self.dividend_yield is None:
            return "N/A"
        return f"{self.dividend_yield * 100:.2f}%"

    def returns_1yr_pct(self) -> str:
        """Format 1-year return as percentage string."""
        if self.returns_1yr is None:
            return "N/A"
        return f"{self.returns_1yr * 100:.2f}%"

    def returns_5yr_pct(self) -> str:
        """Format 5-year return as percentage string."""
        if self.returns_5yr is None:
            return "N/A"
        return f"{self.returns_5yr * 100:.2f}%"

    def returns_10yr_pct(self) -> str:
        """Format 10-year return as percentage string."""
        if self.returns_10yr is None:
            return "N/A"
        return f"{self.returns_10yr * 100:.2f}%"


class TickerDataCache:
    """Session-level cache for ticker data."""

    def __init__(self):
        self._cache: Dict[str, TickerInfo] = {}
        self._price_cache: Dict[str, Tuple[datetime, float]] = {}

    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._price_cache.clear()

    def get(self, ticker: str) -> Optional[TickerInfo]:
        """Get cached ticker info if available."""
        return self._cache.get(ticker.upper())

    def set(self, ticker: str, info: TickerInfo):
        """Cache ticker info."""
        self._cache[ticker.upper()] = info

    def has(self, ticker: str) -> bool:
        """Check if ticker is cached."""
        return ticker.upper() in self._cache


# Global cache instance
_cache = TickerDataCache()


def get_cache() -> TickerDataCache:
    """Get the global ticker cache."""
    return _cache


def clear_cache():
    """Clear the global ticker cache."""
    _cache.clear()


def _calculate_annualized_return(prices, years: int) -> Optional[float]:
    """Calculate annualized return from price series.

    Args:
        prices: Price series (pandas Series with datetime index)
        years: Number of years to look back

    Returns:
        Annualized return as decimal, or None if insufficient data
    """
    if prices is None or len(prices) < 2:
        return None

    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=years * 365)

    # Find closest available date to target start
    available_dates = prices.index[prices.index >= start_date]
    if len(available_dates) < 2:
        return None

    actual_start = available_dates[0]
    actual_end = prices.index[-1]

    start_price = prices.loc[actual_start]
    end_price = prices.loc[actual_end]

    if start_price <= 0:
        return None

    # Calculate actual years between dates
    actual_years = (actual_end - actual_start).days / 365.25
    if actual_years < years * 0.9:  # Require at least 90% of requested period
        return None

    # Annualized return
    total_return = end_price / start_price
    annualized = total_return ** (1 / actual_years) - 1

    return annualized


def fetch_ticker_info(ticker: str, use_cache: bool = True) -> TickerInfo:
    """Fetch comprehensive ticker information.

    Args:
        ticker: Ticker symbol
        use_cache: Whether to use cached data if available

    Returns:
        TickerInfo object with all available data
    """
    ticker = ticker.upper().strip()

    # Check cache first
    if use_cache and _cache.has(ticker):
        return _cache.get(ticker)

    if not HAS_YFINANCE:
        return TickerInfo(
            ticker=ticker,
            name=ticker,
            expense_ratio=None,
            dividend_yield=None,
            last_price=None,
            returns_1yr=None,
            returns_5yr=None,
            returns_10yr=None,
            is_valid=False,
            error_message="yfinance not installed"
        )

    try:
        yf_ticker = yf.Ticker(ticker)

        # Get basic info
        info = yf_ticker.info

        if not info or info.get("regularMarketPrice") is None:
            # Try to check if it's a valid ticker by fetching history
            hist = yf_ticker.history(period="5d")
            if hist.empty:
                result = TickerInfo(
                    ticker=ticker,
                    name=ticker,
                    expense_ratio=None,
                    dividend_yield=None,
                    last_price=None,
                    returns_1yr=None,
                    returns_5yr=None,
                    returns_10yr=None,
                    is_valid=False,
                    error_message=f"Ticker '{ticker}' not found"
                )
                _cache.set(ticker, result)
                return result

        # Extract name
        name = info.get("shortName") or info.get("longName") or ticker

        # Extract expense ratio (for ETFs/funds)
        expense_ratio = None
        if "annualReportExpenseRatio" in info:
            expense_ratio = info["annualReportExpenseRatio"]
        elif "expenseRatio" in info:
            expense_ratio = info["expenseRatio"]

        # Extract dividend yield
        dividend_yield = info.get("trailingAnnualDividendYield") or info.get("dividendYield")

        # Get last price
        last_price = info.get("regularMarketPrice") or info.get("previousClose")

        # Fetch historical prices for return calculations
        hist = yf_ticker.history(period="11y", auto_adjust=True)

        returns_1yr = None
        returns_5yr = None
        returns_10yr = None

        if not hist.empty and "Close" in hist.columns:
            prices = hist["Close"]
            returns_1yr = _calculate_annualized_return(prices, 1)
            returns_5yr = _calculate_annualized_return(prices, 5)
            returns_10yr = _calculate_annualized_return(prices, 10)

        result = TickerInfo(
            ticker=ticker,
            name=name,
            expense_ratio=expense_ratio,
            dividend_yield=dividend_yield,
            last_price=last_price,
            returns_1yr=returns_1yr,
            returns_5yr=returns_5yr,
            returns_10yr=returns_10yr,
            is_valid=True,
            error_message=None
        )

        _cache.set(ticker, result)
        return result

    except Exception as e:
        result = TickerInfo(
            ticker=ticker,
            name=ticker,
            expense_ratio=None,
            dividend_yield=None,
            last_price=None,
            returns_1yr=None,
            returns_5yr=None,
            returns_10yr=None,
            is_valid=False,
            error_message=str(e)
        )
        _cache.set(ticker, result)
        return result


def fetch_multiple_tickers(tickers: List[str], use_cache: bool = True) -> Dict[str, TickerInfo]:
    """Fetch info for multiple tickers.

    Args:
        tickers: List of ticker symbols
        use_cache: Whether to use cached data

    Returns:
        Dict mapping ticker to TickerInfo
    """
    results = {}
    for ticker in tickers:
        results[ticker.upper()] = fetch_ticker_info(ticker, use_cache)
    return results


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Quick validation of a ticker symbol.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    info = fetch_ticker_info(ticker)
    return info.is_valid, info.error_message or ""


def calculate_portfolio_stats(
    holdings: Dict[str, float],
    ticker_data: Dict[str, TickerInfo]
) -> Dict[str, Optional[float]]:
    """Calculate weighted portfolio statistics.

    Args:
        holdings: Dict mapping ticker to weight (0-1)
        ticker_data: Dict mapping ticker to TickerInfo

    Returns:
        Dict with portfolio statistics
    """
    total_weight = sum(holdings.values())
    if total_weight == 0:
        return {
            "weighted_expense_ratio": None,
            "weighted_yield": None,
            "weighted_return_1yr": None,
            "weighted_return_5yr": None,
            "weighted_return_10yr": None,
        }

    # Normalize weights
    weights = {t: w / total_weight for t, w in holdings.items()}

    # Calculate weighted averages
    expense_sum = 0.0
    expense_weight = 0.0
    yield_sum = 0.0
    yield_weight = 0.0
    return_1yr_sum = 0.0
    return_1yr_weight = 0.0
    return_5yr_sum = 0.0
    return_5yr_weight = 0.0
    return_10yr_sum = 0.0
    return_10yr_weight = 0.0

    for ticker, weight in weights.items():
        info = ticker_data.get(ticker.upper())
        if info is None or not info.is_valid:
            continue

        if info.expense_ratio is not None:
            expense_sum += info.expense_ratio * weight
            expense_weight += weight

        if info.dividend_yield is not None:
            yield_sum += info.dividend_yield * weight
            yield_weight += weight

        if info.returns_1yr is not None:
            return_1yr_sum += info.returns_1yr * weight
            return_1yr_weight += weight

        if info.returns_5yr is not None:
            return_5yr_sum += info.returns_5yr * weight
            return_5yr_weight += weight

        if info.returns_10yr is not None:
            return_10yr_sum += info.returns_10yr * weight
            return_10yr_weight += weight

    return {
        "weighted_expense_ratio": expense_sum if expense_weight > 0.5 else None,
        "weighted_yield": yield_sum if yield_weight > 0.5 else None,
        "weighted_return_1yr": return_1yr_sum if return_1yr_weight > 0.5 else None,
        "weighted_return_5yr": return_5yr_sum if return_5yr_weight > 0.5 else None,
        "weighted_return_10yr": return_10yr_sum if return_10yr_weight > 0.5 else None,
    }

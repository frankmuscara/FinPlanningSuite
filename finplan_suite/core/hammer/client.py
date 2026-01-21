"""
Client Portfolio Management

Import, compare, and generate rebalance trades for client portfolios.
"""

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date

import pandas as pd
import yfinance as yf

from .models import PortfolioModel


@dataclass
class ClientHolding:
    """A single holding in a client portfolio."""

    ticker: str
    shares: float
    current_price: Optional[float] = None
    current_value: Optional[float] = None

    def __post_init__(self):
        self.ticker = self.ticker.upper().strip()
        if self.current_price and not self.current_value:
            self.current_value = self.shares * self.current_price


@dataclass
class ClientPortfolio:
    """A client's current portfolio holdings."""

    name: str
    holdings: List[ClientHolding]
    cash: float = 0.0
    as_of_date: Optional[date] = None

    @property
    def total_value(self) -> float:
        """Total portfolio value including cash."""
        holdings_value = sum(h.current_value or 0 for h in self.holdings)
        return holdings_value + self.cash

    @property
    def holdings_value(self) -> float:
        """Value of holdings only (excluding cash)."""
        return sum(h.current_value or 0 for h in self.holdings)

    def get_weights(self) -> Dict[str, float]:
        """Get current weights as percentages (0-100)."""
        total = self.total_value
        if total == 0:
            return {}

        weights = {}
        for h in self.holdings:
            if h.current_value:
                weights[h.ticker] = (h.current_value / total) * 100

        if self.cash > 0:
            weights["CASH"] = (self.cash / total) * 100

        return weights

    def get_holding(self, ticker: str) -> Optional[ClientHolding]:
        """Get a specific holding by ticker."""
        ticker = ticker.upper()
        for h in self.holdings:
            if h.ticker == ticker:
                return h
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        data = []
        for h in self.holdings:
            data.append({
                "Ticker": h.ticker,
                "Shares": h.shares,
                "Price": h.current_price,
                "Value": h.current_value,
                "Weight": (h.current_value / self.total_value * 100) if self.total_value > 0 else 0,
            })
        if self.cash > 0:
            data.append({
                "Ticker": "CASH",
                "Shares": None,
                "Price": 1.0,
                "Value": self.cash,
                "Weight": (self.cash / self.total_value * 100) if self.total_value > 0 else 0,
            })
        return pd.DataFrame(data)


@dataclass
class Trade:
    """A recommended trade."""

    ticker: str
    action: str  # "BUY" or "SELL"
    shares: float
    estimated_value: float
    current_weight: float
    target_weight: float
    reason: str = ""

    @property
    def is_buy(self) -> bool:
        return self.action == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.action == "SELL"


@dataclass
class RebalanceResult:
    """Results of a portfolio rebalance comparison."""

    client_portfolio: ClientPortfolio
    target_model: PortfolioModel
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    weight_differences: Dict[str, float]
    trades: List[Trade]
    total_buy_value: float
    total_sell_value: float
    net_cash_needed: float
    prices: Dict[str, float]

    @property
    def is_balanced(self) -> bool:
        """Check if portfolio is already balanced (within 1% threshold)."""
        return all(abs(d) < 1.0 for d in self.weight_differences.values())

    def to_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=["Action", "Ticker", "Shares", "Est. Value", "Current %", "Target %"])

        data = []
        for t in self.trades:
            data.append({
                "Action": t.action,
                "Ticker": t.ticker,
                "Shares": round(t.shares, 4),
                "Est. Value": round(t.estimated_value, 2),
                "Current %": round(t.current_weight, 2),
                "Target %": round(t.target_weight, 2),
            })
        return pd.DataFrame(data)

    def to_comparison_dataframe(self) -> pd.DataFrame:
        """Create comparison DataFrame."""
        all_tickers = set(self.current_weights.keys()) | set(self.target_weights.keys())
        all_tickers.discard("CASH")  # Handle cash separately

        data = []
        for ticker in sorted(all_tickers):
            current = self.current_weights.get(ticker, 0)
            target = self.target_weights.get(ticker, 0)
            diff = current - target

            data.append({
                "Ticker": ticker,
                "Current %": round(current, 2),
                "Target %": round(target, 2),
                "Difference": round(diff, 2),
                "Status": "✓" if abs(diff) < 1.0 else ("↑ Over" if diff > 0 else "↓ Under"),
            })

        return pd.DataFrame(data)


def parse_csv_portfolio(
    csv_content: str,
    portfolio_name: str = "Client Portfolio",
    fetch_prices: bool = True,
) -> ClientPortfolio:
    """
    Parse a CSV file into a ClientPortfolio.

    Supported CSV formats:
    1. ticker,shares - Will fetch current prices
    2. ticker,shares,price - Uses provided prices
    3. ticker,value - Assumes this is the dollar value (will calc shares from price)

    The parser auto-detects the format based on headers or data patterns.
    """
    # Clean up the CSV content
    csv_content = csv_content.strip()

    # Try to detect format
    reader = csv.reader(io.StringIO(csv_content))
    rows = list(reader)

    if len(rows) < 1:
        raise ValueError("CSV is empty")

    # Check for header row
    first_row = rows[0]
    has_header = any(
        h.lower() in ["ticker", "symbol", "stock", "shares", "quantity", "value", "price"]
        for h in first_row
    )

    if has_header:
        headers = [h.lower().strip() for h in first_row]
        data_rows = rows[1:]
    else:
        # Assume ticker,shares format
        headers = ["ticker", "shares"]
        data_rows = rows

    # Map common header variations
    header_map = {
        "ticker": ["ticker", "symbol", "stock", "holding", "security"],
        "shares": ["shares", "quantity", "qty", "units", "amount"],
        "price": ["price", "cost", "value_per_share"],
        "value": ["value", "total", "market_value", "market value", "total_value"],
    }

    def find_column(target: str) -> Optional[int]:
        for i, h in enumerate(headers):
            if h in header_map.get(target, [target]):
                return i
        return None

    ticker_col = find_column("ticker")
    shares_col = find_column("shares")
    price_col = find_column("price")
    value_col = find_column("value")

    if ticker_col is None:
        raise ValueError("Could not find ticker/symbol column in CSV")

    # Parse holdings
    holdings_data = []
    cash = 0.0

    for row in data_rows:
        if len(row) <= ticker_col:
            continue

        ticker = row[ticker_col].upper().strip()
        if not ticker:
            continue

        # Handle cash
        if ticker in ["CASH", "$CASH", "USD", "MONEY MARKET", "MM"]:
            if value_col is not None and len(row) > value_col:
                try:
                    cash = float(row[value_col].replace(",", "").replace("$", ""))
                except ValueError:
                    pass
            elif shares_col is not None and len(row) > shares_col:
                try:
                    cash = float(row[shares_col].replace(",", "").replace("$", ""))
                except ValueError:
                    pass
            continue

        # Parse shares
        shares = None
        if shares_col is not None and len(row) > shares_col:
            try:
                shares = float(row[shares_col].replace(",", ""))
            except ValueError:
                pass

        # Parse price
        price = None
        if price_col is not None and len(row) > price_col:
            try:
                price = float(row[price_col].replace(",", "").replace("$", ""))
            except ValueError:
                pass

        # Parse value
        value = None
        if value_col is not None and len(row) > value_col:
            try:
                value = float(row[value_col].replace(",", "").replace("$", ""))
            except ValueError:
                pass

        holdings_data.append({
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": value,
        })

    # Fetch prices if needed
    tickers_to_fetch = []
    for h in holdings_data:
        if h["price"] is None:
            tickers_to_fetch.append(h["ticker"])

    fetched_prices = {}
    if fetch_prices and tickers_to_fetch:
        fetched_prices = fetch_current_prices(tickers_to_fetch)

    # Build holdings
    holdings = []
    for h in holdings_data:
        price = h["price"] or fetched_prices.get(h["ticker"])
        shares = h["shares"]
        value = h["value"]

        # Calculate missing values
        if shares is None and price and value:
            shares = value / price
        if value is None and price and shares:
            value = price * shares

        if shares is not None:
            holdings.append(ClientHolding(
                ticker=h["ticker"],
                shares=shares,
                current_price=price,
                current_value=value,
            ))

    return ClientPortfolio(
        name=portfolio_name,
        holdings=holdings,
        cash=cash,
        as_of_date=date.today(),
    )


def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for a list of tickers."""
    if not tickers:
        return {}

    prices = {}
    try:
        # Fetch all at once for efficiency
        data = yf.download(tickers, period="1d", progress=False)

        if len(tickers) == 1:
            # Single ticker returns different format
            if "Close" in data.columns:
                price = data["Close"].iloc[-1]
                if pd.notna(price):
                    prices[tickers[0]] = float(price)
        else:
            # Multiple tickers
            if "Close" in data.columns:
                for ticker in tickers:
                    if ticker in data["Close"].columns:
                        price = data["Close"][ticker].iloc[-1]
                        if pd.notna(price):
                            prices[ticker] = float(price)
    except Exception as e:
        print(f"Warning: Could not fetch prices: {e}")

    return prices


def compare_to_model(
    client_portfolio: ClientPortfolio,
    target_model: PortfolioModel,
    include_missing_from_client: bool = True,
) -> RebalanceResult:
    """
    Compare a client portfolio to a target model.

    Returns comparison data and recommended trades.
    """
    # Get current weights
    current_weights = client_portfolio.get_weights()

    # Get target weights (as percentages)
    target_weights = dict(target_model.allocations)

    # Handle tickers in client but not in model (set target to 0)
    for ticker in current_weights:
        if ticker not in target_weights and ticker != "CASH":
            target_weights[ticker] = 0

    # Handle tickers in model but not in client
    if include_missing_from_client:
        for ticker in target_weights:
            if ticker not in current_weights:
                current_weights[ticker] = 0

    # Calculate differences
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    all_tickers.discard("CASH")

    weight_differences = {}
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        target = target_weights.get(ticker, 0)
        weight_differences[ticker] = current - target

    # Fetch prices for all tickers we need
    tickers_needing_prices = []
    prices = {}

    for h in client_portfolio.holdings:
        if h.current_price:
            prices[h.ticker] = h.current_price
        else:
            tickers_needing_prices.append(h.ticker)

    for ticker in target_weights:
        if ticker not in prices and ticker != "CASH":
            tickers_needing_prices.append(ticker)

    if tickers_needing_prices:
        fetched = fetch_current_prices(list(set(tickers_needing_prices)))
        prices.update(fetched)

    # Generate trades
    trades = generate_trades(
        client_portfolio=client_portfolio,
        target_weights=target_weights,
        prices=prices,
    )

    total_buy = sum(t.estimated_value for t in trades if t.is_buy)
    total_sell = sum(t.estimated_value for t in trades if t.is_sell)

    return RebalanceResult(
        client_portfolio=client_portfolio,
        target_model=target_model,
        current_weights=current_weights,
        target_weights=target_weights,
        weight_differences=weight_differences,
        trades=trades,
        total_buy_value=total_buy,
        total_sell_value=total_sell,
        net_cash_needed=total_buy - total_sell - client_portfolio.cash,
        prices=prices,
    )


def generate_trades(
    client_portfolio: ClientPortfolio,
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    threshold_pct: float = 1.0,
) -> List[Trade]:
    """
    Generate trades to rebalance portfolio to target weights.

    Args:
        client_portfolio: Current portfolio
        target_weights: Target weights as percentages (0-100)
        prices: Current prices for all tickers
        threshold_pct: Minimum weight difference to trigger trade

    Returns:
        List of Trade objects
    """
    trades = []
    total_value = client_portfolio.total_value

    if total_value <= 0:
        return trades

    current_weights = client_portfolio.get_weights()

    # Get all tickers
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    all_tickers.discard("CASH")

    for ticker in sorted(all_tickers):
        current_weight = current_weights.get(ticker, 0)
        target_weight = target_weights.get(ticker, 0)
        diff = target_weight - current_weight

        # Skip if within threshold
        if abs(diff) < threshold_pct:
            continue

        # Get price
        price = prices.get(ticker)
        if not price:
            continue

        # Calculate target value and current value
        target_value = (target_weight / 100) * total_value
        current_value = (current_weight / 100) * total_value

        # Calculate trade
        value_diff = target_value - current_value
        shares_diff = abs(value_diff / price)

        if value_diff > 0:
            action = "BUY"
            reason = f"Underweight by {abs(diff):.1f}%"
        else:
            action = "SELL"
            reason = f"Overweight by {abs(diff):.1f}%"

        trades.append(Trade(
            ticker=ticker,
            action=action,
            shares=shares_diff,
            estimated_value=abs(value_diff),
            current_weight=current_weight,
            target_weight=target_weight,
            reason=reason,
        ))

    # Sort: sells first, then buys
    trades.sort(key=lambda t: (0 if t.is_sell else 1, t.ticker))

    return trades


def generate_trade_csv(trades: List[Trade]) -> str:
    """Generate CSV content for trades."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Action", "Ticker", "Shares", "Estimated Value", "Notes"])

    for t in trades:
        writer.writerow([
            t.action,
            t.ticker,
            round(t.shares, 4),
            round(t.estimated_value, 2),
            t.reason,
        ])

    return output.getvalue()

"""Portfolio configuration and position tracking for HAMMER."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class PortfolioConfig:
    """Configuration for a portfolio backtest."""

    tickers: List[str]
    target_weights: Dict[str, float]
    benchmark: str
    initial_capital: float
    start_date: date
    end_date: date

    def __post_init__(self):
        """Validate configuration."""
        # Normalize weights
        total_weight = sum(self.target_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            # Auto-normalize
            self.target_weights = {
                k: v / total_weight for k, v in self.target_weights.items()
            }

        # Validate tickers match weights
        if set(self.tickers) != set(self.target_weights.keys()):
            raise ValueError("Tickers must match target_weights keys")

        # Validate dates
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        # Validate capital
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tickers": self.tickers,
            "target_weights": self.target_weights,
            "benchmark": self.benchmark,
            "initial_capital": self.initial_capital,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioConfig":
        """Create from dictionary."""
        return cls(
            tickers=data["tickers"],
            target_weights=data["target_weights"],
            benchmark=data["benchmark"],
            initial_capital=data["initial_capital"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
        )


@dataclass
class Position:
    """Tracks shares held in each asset."""

    shares: Dict[str, float] = field(default_factory=dict)

    def value(self, prices: Dict[str, float]) -> float:
        """Calculate total position value at given prices."""
        return sum(
            self.shares.get(ticker, 0) * price
            for ticker, price in prices.items()
        )

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate current weights at given prices."""
        total = self.value(prices)
        if total == 0:
            return {ticker: 0.0 for ticker in self.shares}
        return {
            ticker: (self.shares.get(ticker, 0) * prices.get(ticker, 0)) / total
            for ticker in self.shares
        }

    def max_drift(
        self,
        prices: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> float:
        """Calculate maximum absolute drift from target weights."""
        current = self.weights(prices)
        drifts = [
            abs(current.get(ticker, 0) - target_weights.get(ticker, 0))
            for ticker in target_weights
        ]
        return max(drifts) if drifts else 0.0

    def rebalance_to(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        total_value: Optional[float] = None,
    ) -> "Position":
        """Create new position with target weights.

        Args:
            target_weights: Target weight per ticker
            prices: Current prices per ticker
            total_value: Total value to allocate (uses current value if None)

        Returns:
            New Position with updated shares
        """
        if total_value is None:
            total_value = self.value(prices)

        new_shares = {}
        for ticker, weight in target_weights.items():
            price = prices.get(ticker, 0)
            if price > 0:
                new_shares[ticker] = (total_value * weight) / price
            else:
                new_shares[ticker] = 0

        return Position(shares=new_shares)

    def copy(self) -> "Position":
        """Create a copy of this position."""
        return Position(shares=self.shares.copy())


def calculate_turnover(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
) -> float:
    """Calculate turnover as sum of absolute weight changes.

    Turnover of 1.0 means 100% of portfolio was traded.
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    return sum(
        abs(new_weights.get(t, 0) - old_weights.get(t, 0))
        for t in all_tickers
    ) / 2  # Divide by 2 because buys = sells

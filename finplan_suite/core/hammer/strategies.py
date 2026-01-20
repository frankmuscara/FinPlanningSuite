"""Rebalancing strategy definitions for HAMMER."""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional
import pandas as pd


class StrategyMode(Enum):
    """Available rebalancing strategies."""

    BUY_HOLD = "buy_hold"          # Never rebalance
    PERIODIC = "periodic"          # Rebalance on schedule
    DRIFT = "drift"                # Rebalance when drift exceeds threshold
    HAMMER = "hammer"              # Drift + VIX gate


class RebalanceFrequency(Enum):
    """Rebalance frequencies for periodic strategy."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class StrategyConfig:
    """Configuration for a rebalancing strategy."""

    mode: StrategyMode
    rebalance_frequency: Optional[RebalanceFrequency] = None  # For periodic
    drift_threshold: float = 0.05  # 5% default for drift/hammer

    def __post_init__(self):
        """Validate configuration."""
        if self.mode == StrategyMode.PERIODIC and self.rebalance_frequency is None:
            raise ValueError("rebalance_frequency required for PERIODIC mode")

        if self.drift_threshold <= 0 or self.drift_threshold > 1:
            raise ValueError("drift_threshold must be between 0 and 1")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "rebalance_frequency": (
                self.rebalance_frequency.value
                if self.rebalance_frequency else None
            ),
            "drift_threshold": self.drift_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyConfig":
        """Create from dictionary."""
        return cls(
            mode=StrategyMode(data["mode"]),
            rebalance_frequency=(
                RebalanceFrequency(data["rebalance_frequency"])
                if data.get("rebalance_frequency") else None
            ),
            drift_threshold=data.get("drift_threshold", 0.05),
        )


def is_period_end(current_date: date, frequency: RebalanceFrequency) -> bool:
    """Check if current_date is end of period for given frequency.

    Uses last business day of period logic.
    """
    if frequency == RebalanceFrequency.MONTHLY:
        # Check if next business day is in different month
        next_day = current_date + pd.Timedelta(days=1)
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        return next_day.month != current_date.month

    elif frequency == RebalanceFrequency.QUARTERLY:
        # End of quarter = end of March, June, September, December
        if current_date.month not in [3, 6, 9, 12]:
            return False
        next_day = current_date + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        return next_day.month != current_date.month

    elif frequency == RebalanceFrequency.ANNUAL:
        # End of year = end of December
        if current_date.month != 12:
            return False
        next_day = current_date + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        return next_day.year != current_date.year

    return False


def check_rebalance_trigger(
    current_date: date,
    current_drift: float,
    strategy: StrategyConfig,
    is_first_day: bool = False,
) -> bool:
    """Check if rebalance should be triggered.

    Args:
        current_date: Current simulation date
        current_drift: Maximum weight drift from target
        strategy: Strategy configuration
        is_first_day: Whether this is the first day (always invest)

    Returns:
        True if rebalance should be triggered
    """
    # Always invest on first day
    if is_first_day:
        return True

    if strategy.mode == StrategyMode.BUY_HOLD:
        return False

    elif strategy.mode == StrategyMode.PERIODIC:
        return is_period_end(current_date, strategy.rebalance_frequency)

    elif strategy.mode in (StrategyMode.DRIFT, StrategyMode.HAMMER):
        return current_drift > strategy.drift_threshold

    return False

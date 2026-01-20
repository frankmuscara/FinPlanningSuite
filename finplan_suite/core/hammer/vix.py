"""VIX term structure handling for HAMMER."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


@dataclass
class VixSignal:
    """Current VIX signal status."""

    vix: float
    vix3m: float
    slope: float                  # VIX3M - VIX
    slope_pct: float              # Slope as % of VIX
    is_inverted: bool             # True = backwardation = HALT
    level: str                    # 'low', 'normal', 'elevated', 'high'

    @property
    def is_safe(self) -> bool:
        """Returns True if conditions are safe for rebalancing."""
        return not self.is_inverted

    @property
    def status_text(self) -> str:
        """Human-readable status."""
        if self.is_inverted:
            return "HALT TRADING - VIX Curve Inverted"
        elif self.level == "high":
            return "CAUTION - VIX Elevated"
        else:
            return "SAFE - Normal Conditions"


def fetch_vix_data(
    start_date: date,
    end_date: date,
    max_gap_fill: int = 5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Fetch VIX and VIX3M data, calculate slope.

    Args:
        start_date: Start date
        end_date: End date
        max_gap_fill: Maximum days to forward-fill gaps

    Returns:
        Tuple of (vix_series, vix3m_series, slope_series)
        Slope = VIX3M - VIX (positive = contango = safe)
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance required for VIX data")

    # Add buffer for alignment
    buffer_start = start_date - timedelta(days=10)

    # Fetch VIX data
    vix_data = yf.download(
        "^VIX",
        start=buffer_start.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
    )

    vix3m_data = yf.download(
        "^VIX3M",
        start=buffer_start.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
    )

    if vix_data.empty or vix3m_data.empty:
        raise ValueError("Could not fetch VIX data")

    # Extract close prices
    vix = vix_data["Close"].squeeze()
    vix3m = vix3m_data["Close"].squeeze()

    # Forward-fill gaps
    vix = vix.ffill(limit=max_gap_fill)
    vix3m = vix3m.ffill(limit=max_gap_fill)

    # Align indices
    common_idx = vix.index.intersection(vix3m.index)
    vix = vix.loc[common_idx]
    vix3m = vix3m.loc[common_idx]

    # Trim to date range
    vix = vix.loc[start_date.isoformat():end_date.isoformat()]
    vix3m = vix3m.loc[start_date.isoformat():end_date.isoformat()]

    # Calculate slope (positive = contango = safe)
    slope = vix3m - vix

    return vix, vix3m, slope


def get_vix_signal(vix: float, vix3m: float) -> VixSignal:
    """Get current VIX signal from spot values.

    Args:
        vix: Current VIX value
        vix3m: Current VIX3M value

    Returns:
        VixSignal with analysis
    """
    slope = vix3m - vix
    slope_pct = (slope / vix * 100) if vix > 0 else 0
    is_inverted = slope < 0

    # Categorize level
    if vix < 15:
        level = "low"
    elif vix < 20:
        level = "normal"
    elif vix < 30:
        level = "elevated"
    else:
        level = "high"

    return VixSignal(
        vix=vix,
        vix3m=vix3m,
        slope=slope,
        slope_pct=slope_pct,
        is_inverted=is_inverted,
        level=level,
    )


def is_vix_blocked(slope: float) -> bool:
    """Check if VIX conditions block rebalancing.

    Args:
        slope: VIX3M - VIX value

    Returns:
        True if rebalancing should be blocked (inverted curve)
    """
    return slope < 0


def get_blocked_regions(slope: pd.Series) -> pd.DataFrame:
    """Identify contiguous regions where VIX was inverted.

    Args:
        slope: Time series of VIX slope

    Returns:
        DataFrame with 'start' and 'end' columns for blocked regions
    """
    is_blocked = slope < 0

    # Find transitions
    transitions = is_blocked.astype(int).diff()
    starts = slope.index[transitions == 1].tolist()
    ends = slope.index[transitions == -1].tolist()

    # Handle edge cases
    if is_blocked.iloc[0]:
        starts = [slope.index[0]] + starts
    if is_blocked.iloc[-1]:
        ends = ends + [slope.index[-1]]

    regions = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else slope.index[-1]
        regions.append({"start": start, "end": end})

    return pd.DataFrame(regions)

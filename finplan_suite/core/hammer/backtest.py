"""Backtest engine for HAMMER - completely UI-independent.

IMPORTANT: All backtests use TOTAL RETURN (adjusted close prices) which
includes dividends and stock splits. This means performance metrics reflect
what an investor would actually earn, not just price appreciation.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .portfolio import PortfolioConfig, Position, calculate_turnover
from .strategies import (
    StrategyConfig,
    StrategyMode,
    check_rebalance_trigger,
)
from .vix import fetch_vix_data, is_vix_blocked
from .data import fetch_prices, validate_prices, get_common_date_range
from .asset_classes import AssetClass, get_asset_class


@dataclass
class RebalanceEvent:
    """Record of a rebalance or blocked event."""

    date: date
    event_type: str              # 'rebalance', 'blocked', 'initial'
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    turnover: float
    vix_slope: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Configuration
    portfolio_config: PortfolioConfig
    strategy_config: StrategyConfig

    # Time series (all aligned to same DatetimeIndex)
    nav: pd.Series                    # Daily portfolio NAV
    benchmark_nav: pd.Series          # Daily benchmark NAV
    weights: pd.DataFrame             # Daily weights per ticker
    vix_slope: Optional[pd.Series]    # VIX3M - VIX (if HAMMER mode)

    # Events
    events: List[RebalanceEvent]

    # Coverage info
    data_coverage: Dict[str, Tuple[date, date]]
    effective_start: date
    effective_end: date
    data_warnings: List[str] = field(default_factory=list)

    @property
    def rebalance_events(self) -> List[RebalanceEvent]:
        """Get executed rebalance events (excluding blocked, including partial)."""
        return [e for e in self.events if e.event_type in ("rebalance", "partial")]
    
    @property
    def blocked_events(self) -> List[RebalanceEvent]:
        """Get only blocked events."""
        return [e for e in self.events if e.event_type == "blocked"]

    @property
    def partial_events(self) -> List[RebalanceEvent]:
        """Get only partial rebalance events (equity frozen)."""
        return [e for e in self.events if e.event_type == "partial"]

    @property
    def total_turnover(self) -> float:
        """Total turnover across all rebalances."""
        return sum(e.turnover for e in self.rebalance_events)

    @property
    def rebalances_per_year(self) -> float:
        """Average rebalances per year."""
        years = (self.effective_end - self.effective_start).days / 365.25
        if years <= 0:
            return 0
        actual_rebalances = len(self.rebalance_events)
        return actual_rebalances / years


def _build_partial_hammer_targets(
    old_weights: Dict[str, float],
    target_weights: Dict[str, float],
) -> Dict[str, float]:
    """Build HAMMER partial targets: freeze intra-equity, allow inter-asset shifts."""
    equity_tickers = [
        t for t in target_weights.keys()
        if get_asset_class(t) == AssetClass.EQUITY
    ]
    non_equity_tickers = [t for t in target_weights.keys() if t not in equity_tickers]

    eq_target_total = float(sum(target_weights.get(t, 0.0) for t in equity_tickers))
    old_eq_total = float(sum(old_weights.get(t, 0.0) for t in equity_tickers))

    partial = {t: float(target_weights.get(t, 0.0)) for t in non_equity_tickers}

    if equity_tickers:
        if old_eq_total > 0:
            for t in equity_tickers:
                rel = float(old_weights.get(t, 0.0)) / old_eq_total
                partial[t] = eq_target_total * rel
        else:
            # No existing equity sleeve to freeze; fall back to target split.
            for t in equity_tickers:
                partial[t] = float(target_weights.get(t, 0.0))

    total = float(sum(partial.values()))
    if total > 0:
        partial = {t: w / total for t, w in partial.items()}
    return partial

    def to_dataframe(self) -> pd.DataFrame:
        """Export results as a DataFrame for CSV export."""
        df = pd.DataFrame(index=self.nav.index)
        df["portfolio_nav"] = self.nav
        df["benchmark_nav"] = self.benchmark_nav

        # Add weights
        for ticker in self.weights.columns:
            df[f"{ticker}_weight"] = self.weights[ticker]

        # Add VIX slope if available
        if self.vix_slope is not None:
            df["vix_slope"] = self.vix_slope

        # Add event column
        event_dates = {e.date: e.event_type for e in self.events}
        df["event"] = df.index.map(
            lambda d: event_dates.get(d.date() if hasattr(d, "date") else d, "")
        )

        return df


class BacktestEngine:
    """Engine for running portfolio backtests."""

    def __init__(
        self,
        portfolio_config: PortfolioConfig,
        strategy_config: StrategyConfig,
        override_blocked_dates: Optional[set] = None,
    ):
        """Initialize backtest engine.

        Args:
            portfolio_config: Portfolio configuration
            strategy_config: Strategy configuration
            override_blocked_dates: If provided, these dates will be blocked
                instead of using VIX-based blocking. Used for permutation tests.
        """
        self.portfolio_config = portfolio_config
        self.strategy_config = strategy_config
        self.override_blocked_dates = override_blocked_dates

    def run(self) -> BacktestResult:
        """Execute the backtest.

        Returns:
            BacktestResult with all time series and events
        """
        config = self.portfolio_config
        strategy = self.strategy_config

        # 1. Fetch price data
        all_tickers = config.tickers + [config.benchmark]
        prices, coverage = fetch_prices(
            all_tickers,
            config.start_date,
            config.end_date,
        )
        data_warnings: List[str] = []

        late_assets = []
        for ticker in all_tickers:
            start_cov = coverage.get(ticker, (None, None))[0]
            if start_cov is not None and start_cov > config.start_date:
                late_assets.append((ticker, start_cov))
        if late_assets:
            for ticker, start_cov in late_assets:
                data_warnings.append(
                    f"{ticker} inception/data starts at {start_cov} after requested backtest start {config.start_date}."
                )

        common_start, _ = get_common_date_range(coverage)
        if common_start is not None and common_start > config.start_date:
            prices = prices.loc[common_start.isoformat():]
            data_warnings.append(
                f"Backtest start adjusted to {common_start} to avoid synthetic pre-inception history."
            )

        # Validate
        issues = validate_prices(prices)
        if issues["errors"]:
            raise ValueError(f"Data errors: {issues['errors']}")
        data_warnings.extend(issues.get("warnings", []))

        # 2. Fetch VIX data if needed (for HAMMER and SHIELD modes)
        vix_slope = None
        if strategy.mode in (StrategyMode.HAMMER, StrategyMode.SHIELD):
            try:
                _, _, vix_slope = fetch_vix_data(
                    config.start_date,
                    config.end_date,
                )
                # Align to price index
                vix_slope = vix_slope.reindex(prices.index, method="ffill")
            except Exception as e:
                # If VIX data fails, assume safe (don't block)
                vix_slope = pd.Series(1.0, index=prices.index)

        # 3. Initialize simulation
        position = Position()
        events: List[RebalanceEvent] = []
        nav_history = []
        weights_history = []
        benchmark_history = []

        # Track benchmark
        benchmark_start_price = prices[config.benchmark].iloc[0]
        benchmark_shares = config.initial_capital / benchmark_start_price

        # 4. Run simulation
        # Track whether we're in a "pending rebalance blocked" state to avoid
        # counting multiple blocked events for the same rebalance attempt
        pending_rebalance_blocked = False

        for i, current_date in enumerate(prices.index):
            current_prices = prices.loc[current_date].to_dict()
            is_first_day = (i == 0)

            # Calculate current portfolio value
            if is_first_day:
                portfolio_value = config.initial_capital
            else:
                portfolio_value = position.value(current_prices)

            # Calculate current drift
            current_drift = position.max_drift(
                current_prices,
                config.target_weights
            ) if not is_first_day else 0

            # Check if rebalance triggered
            should_rebalance = check_rebalance_trigger(
                current_date.date() if hasattr(current_date, "date") else current_date,
                current_drift,
                strategy,
                is_first_day,
            )

            # Check VIX gate (HAMMER and SHIELD modes)
            vix_blocked = False
            current_vix_slope = None
            current_date_val = current_date.date() if hasattr(current_date, "date") else current_date

            if (
                should_rebalance
                and not is_first_day
                and strategy.mode in (StrategyMode.HAMMER, StrategyMode.SHIELD)
            ):
                # If override_blocked_dates is set, use it instead of VIX
                if self.override_blocked_dates is not None:
                    vix_blocked = current_date_val in self.override_blocked_dates
                    if vix_slope is not None:
                        current_vix_slope = vix_slope.loc[current_date]
                elif vix_slope is not None:
                    current_vix_slope = vix_slope.loc[current_date]
                    vix_blocked = is_vix_blocked(current_vix_slope)

            # Record pre-rebalance weights
            old_weights = position.weights(current_prices) if position.shares else {}

            # If drift drops below threshold, reset the pending blocked state
            if not should_rebalance and pending_rebalance_blocked:
                pending_rebalance_blocked = False

            # Execute rebalance or record blocked event
            if should_rebalance and vix_blocked and strategy.mode == StrategyMode.SHIELD:
                # Only record blocked event if this is a NEW rebalance attempt
                # (not a continuation of an existing blocked state)
                if not pending_rebalance_blocked:
                    events.append(RebalanceEvent(
                        date=current_date.date() if hasattr(current_date, "date") else current_date,
                        event_type="blocked",
                        old_weights=old_weights,
                        new_weights=old_weights,
                        turnover=0,
                        vix_slope=current_vix_slope,
                        reason="VIX curve inverted",
                    ))
                    pending_rebalance_blocked = True

            elif should_rebalance and vix_blocked and strategy.mode == StrategyMode.HAMMER:
                partial_targets = _build_partial_hammer_targets(old_weights, config.target_weights)
                new_position = position.rebalance_to(
                    partial_targets,
                    current_prices,
                    portfolio_value,
                )
                turnover = calculate_turnover(old_weights, partial_targets)
                events.append(RebalanceEvent(
                    date=current_date.date() if hasattr(current_date, "date") else current_date,
                    event_type="partial",
                    old_weights=old_weights,
                    new_weights=partial_targets,
                    turnover=turnover,
                    vix_slope=current_vix_slope,
                    reason="VIX inverted: equity sleeve frozen, inter-asset rebalance allowed",
                ))
                position = new_position
                portfolio_value = position.value(current_prices)
                pending_rebalance_blocked = False

            elif should_rebalance:
                # Execute rebalance
                new_position = position.rebalance_to(
                    config.target_weights,
                    current_prices,
                    portfolio_value,
                )

                turnover = calculate_turnover(old_weights, config.target_weights)

                events.append(RebalanceEvent(
                    date=current_date.date() if hasattr(current_date, "date") else current_date,
                    event_type="initial" if is_first_day else "rebalance",
                    old_weights=old_weights,
                    new_weights=config.target_weights.copy(),
                    turnover=0 if is_first_day else turnover,
                    vix_slope=current_vix_slope,
                ))

                position = new_position
                portfolio_value = position.value(current_prices)
                # Reset blocked state after successful rebalance
                pending_rebalance_blocked = False

            # Record history
            nav_history.append(portfolio_value)
            weights_history.append(position.weights(current_prices))

            # Benchmark value
            benchmark_value = benchmark_shares * current_prices[config.benchmark]
            benchmark_history.append(benchmark_value)

        # 5. Build result
        nav = pd.Series(nav_history, index=prices.index, name="nav")
        benchmark_nav = pd.Series(
            benchmark_history, index=prices.index, name="benchmark"
        )
        weights_df = pd.DataFrame(weights_history, index=prices.index)

        # Effective date range
        effective_start = prices.index[0].date() if hasattr(prices.index[0], "date") else prices.index[0]
        effective_end = prices.index[-1].date() if hasattr(prices.index[-1], "date") else prices.index[-1]

        return BacktestResult(
            portfolio_config=config,
            strategy_config=strategy,
            nav=nav,
            benchmark_nav=benchmark_nav,
            weights=weights_df,
            vix_slope=vix_slope,
            events=events,
            data_coverage=coverage,
            effective_start=effective_start,
            effective_end=effective_end,
            data_warnings=data_warnings,
        )

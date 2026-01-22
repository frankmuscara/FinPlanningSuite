"""
HAMMER - Halt And Manage Market Exposure during Risk

A smart portfolio rebalancing tool that gates rebalances based on VIX term structure.
For long-term investing, not trading.

Strategies:
- HAMMER: Drift-based rebalancing with VIX gate. Blocks intra-equity rebalancing
  during market panic while allowing inter-asset-class adjustments.
- SHIELD: Periodic rebalancing with VIX gate. Skips scheduled rebalances entirely
  when VIX curve is inverted.
- DRIFT: Rebalance when any asset drifts beyond threshold.
- PERIODIC: Rebalance on fixed schedule (monthly/quarterly/annual).
- BUY_HOLD: No rebalancing after initial investment.
"""

from .portfolio import PortfolioConfig, Position
from .strategies import StrategyConfig, StrategyMode, RebalanceFrequency
from .backtest import BacktestEngine, BacktestResult, RebalanceEvent
from .metrics import PerformanceMetrics, compute_metrics
from .vix import fetch_vix_data, get_vix_signal, VixSignal
from .models import PortfolioModel, ModelManager, BUILTIN_MODELS
from .asset_classes import (
    AssetClass,
    get_asset_class,
    classify_tickers,
    get_equity_tickers,
    get_non_equity_tickers,
    DEFAULT_ASSET_CLASS_MAP,
)
from .client import (
    ClientPortfolio,
    ClientHolding,
    Trade,
    RebalanceResult,
    parse_csv_portfolio,
    compare_to_model,
    generate_trades,
)

__version__ = "1.2.0"
__all__ = [
    # Portfolio
    "PortfolioConfig",
    "Position",
    # Strategy
    "StrategyConfig",
    "StrategyMode",
    "RebalanceFrequency",
    # Backtest
    "BacktestEngine",
    "BacktestResult",
    "RebalanceEvent",
    # Metrics
    "PerformanceMetrics",
    "compute_metrics",
    # VIX
    "fetch_vix_data",
    "get_vix_signal",
    "VixSignal",
    # Asset Classes
    "AssetClass",
    "get_asset_class",
    "classify_tickers",
    "get_equity_tickers",
    "get_non_equity_tickers",
    "DEFAULT_ASSET_CLASS_MAP",
    # Models
    "PortfolioModel",
    "ModelManager",
    "BUILTIN_MODELS",
    # Client
    "ClientPortfolio",
    "ClientHolding",
    "Trade",
    "RebalanceResult",
    "parse_csv_portfolio",
    "compare_to_model",
    "generate_trades",
]

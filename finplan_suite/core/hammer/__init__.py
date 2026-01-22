"""
HAMMER - Halt And Manage Market Exposure during Risk

A smart portfolio rebalancing tool that gates rebalances based on VIX term structure.
For long-term investing, not trading.

Key feature: During VIX backwardation (market panic), HAMMER blocks intra-equity
rebalancing while still allowing inter-asset-class rebalancing. This prevents
selling equities into a downturn while maintaining overall allocation targets.
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

"""Comprehensive tests for HAMMER module integration."""

import sys
from unittest.mock import MagicMock
from datetime import date
import pytest

# Mock external dependencies before importing hammer modules
sys.modules['yfinance'] = MagicMock()

import pandas as pd
import numpy as np


class TestAssetClasses:
    def test_get_asset_class_equity(self):
        from finplan_suite.core.hammer.asset_classes import get_asset_class, AssetClass
        assert get_asset_class('VOO') == AssetClass.EQUITY
        assert get_asset_class('SPY') == AssetClass.EQUITY

    def test_get_asset_class_fixed_income(self):
        from finplan_suite.core.hammer.asset_classes import get_asset_class, AssetClass
        assert get_asset_class('BND') == AssetClass.FIXED_INCOME
        assert get_asset_class('TLT') == AssetClass.FIXED_INCOME

    def test_get_asset_class_alternatives(self):
        from finplan_suite.core.hammer.asset_classes import get_asset_class, AssetClass
        assert get_asset_class('GLD') == AssetClass.ALTERNATIVES

    def test_get_asset_class_unknown_defaults_to_equity(self):
        from finplan_suite.core.hammer.asset_classes import get_asset_class, AssetClass
        assert get_asset_class('UNKNOWN_XYZ') == AssetClass.EQUITY

    def test_classify_tickers(self):
        from finplan_suite.core.hammer.asset_classes import classify_tickers, AssetClass
        classified = classify_tickers(['VOO', 'BND', 'GLD'])
        assert 'VOO' in classified[AssetClass.EQUITY]
        assert 'BND' in classified[AssetClass.FIXED_INCOME]
        assert 'GLD' in classified[AssetClass.ALTERNATIVES]


class TestMetrics:
    def test_compute_metrics_with_partial_rebalances(self):
        from finplan_suite.core.hammer.metrics import compute_metrics
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        nav = pd.Series(100 * (1.0005 ** np.arange(len(dates))), index=dates)

        metrics = compute_metrics(
            nav,
            rebalances_per_year=4,
            blocked_rebalances=2,
            partial_rebalances=1,
            total_turnover=0.15,
        )

        assert metrics.partial_rebalances == 1
        assert 'Partial Rebalances' in metrics.to_dict()


class TestBacktestResult:
    def test_events_properties(self):
        from finplan_suite.core.hammer.backtest import BacktestResult, RebalanceEvent
        from finplan_suite.core.hammer.portfolio import PortfolioConfig
        from finplan_suite.core.hammer.strategies import StrategyConfig, StrategyMode

        config = PortfolioConfig(
            tickers=['VOO', 'BND'],
            target_weights={'VOO': 0.6, 'BND': 0.4},
            benchmark='SPY',
            initial_capital=100000,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )
        strategy = StrategyConfig(mode=StrategyMode.HAMMER, drift_threshold=0.05)
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        nav = pd.Series(100, index=dates)

        events = [
            RebalanceEvent(date(2020, 1, 1), 'initial', {}, {}, 0),
            RebalanceEvent(date(2020, 3, 1), 'rebalance', {}, {}, 0.05),
            RebalanceEvent(date(2020, 6, 1), 'blocked', {}, {}, 0),
            RebalanceEvent(date(2020, 9, 1), 'partial', {}, {}, 0.02),
        ]

        result = BacktestResult(
            portfolio_config=config,
            strategy_config=strategy,
            nav=nav, benchmark_nav=nav,
            weights=pd.DataFrame({'VOO': [0.6]*len(dates)}, index=dates),
            vix_slope=pd.Series(1.0, index=dates),
            events=events, data_coverage={},
            effective_start=date(2020, 1, 1),
            effective_end=date(2020, 12, 31),
        )

        assert len(result.blocked_events) == 1
        assert len(result.partial_events) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

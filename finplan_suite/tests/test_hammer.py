"""Comprehensive tests for HAMMER module integration."""

import sys
from unittest.mock import MagicMock
from datetime import date
import pytest

# Mock external dependencies before importing hammer modules
sys.modules['yfinance'] = MagicMock()

import pandas as pd
import numpy as np
from unittest.mock import patch


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


class TestBacktestEngineBehavior:
    def test_initial_investment_not_blocked_by_vix(self):
        from finplan_suite.core.hammer.backtest import BacktestEngine
        from finplan_suite.core.hammer.portfolio import PortfolioConfig
        from finplan_suite.core.hammer.strategies import StrategyConfig, StrategyMode

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        prices = pd.DataFrame(
            {
                "VOO": [100, 101, 102, 103, 104],
                "BND": [100, 100, 100, 100, 100],
                "SPY": [100, 100, 100, 100, 100],
            },
            index=dates,
        )
        coverage = {k: (dates[0].date(), dates[-1].date()) for k in prices.columns}
        neg_slope = pd.Series(-1.0, index=dates)

        config = PortfolioConfig(
            tickers=["VOO", "BND"],
            target_weights={"VOO": 0.5, "BND": 0.5},
            benchmark="SPY",
            initial_capital=100000,
            start_date=dates[0].date(),
            end_date=dates[-1].date(),
        )
        strategy = StrategyConfig(mode=StrategyMode.HAMMER, drift_threshold=0.01)

        with patch("finplan_suite.core.hammer.backtest.fetch_prices", return_value=(prices, coverage)), patch(
            "finplan_suite.core.hammer.backtest.fetch_vix_data", return_value=(neg_slope, neg_slope, neg_slope)
        ):
            result = BacktestEngine(config, strategy).run()

        assert result.nav.iloc[0] == pytest.approx(100000)
        assert result.nav.iloc[1] > 0
        assert any(e.event_type == "initial" for e in result.events)

    def test_hammer_generates_partial_events_under_vix_block(self):
        from finplan_suite.core.hammer.backtest import BacktestEngine
        from finplan_suite.core.hammer.portfolio import PortfolioConfig
        from finplan_suite.core.hammer.strategies import StrategyConfig, StrategyMode

        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        prices = pd.DataFrame(
            {
                "VOO": [100, 110, 120, 130, 140, 150, 160, 170],
                "BND": [100, 100, 100, 100, 100, 100, 100, 100],
                "SPY": [100, 101, 102, 103, 104, 105, 106, 107],
            },
            index=dates,
        )
        coverage = {k: (dates[0].date(), dates[-1].date()) for k in prices.columns}
        neg_slope = pd.Series(-1.0, index=dates)

        config = PortfolioConfig(
            tickers=["VOO", "BND"],
            target_weights={"VOO": 0.5, "BND": 0.5},
            benchmark="SPY",
            initial_capital=100000,
            start_date=dates[0].date(),
            end_date=dates[-1].date(),
        )
        strategy = StrategyConfig(mode=StrategyMode.HAMMER, drift_threshold=0.01)

        with patch("finplan_suite.core.hammer.backtest.fetch_prices", return_value=(prices, coverage)), patch(
            "finplan_suite.core.hammer.backtest.fetch_vix_data", return_value=(neg_slope, neg_slope, neg_slope)
        ):
            result = BacktestEngine(config, strategy).run()

        assert len(result.partial_events) >= 1
        assert all(e.event_type != "blocked" for e in result.partial_events)

    def test_warn_and_adjust_start_for_late_inception(self):
        from finplan_suite.core.hammer.backtest import BacktestEngine
        from finplan_suite.core.hammer.portfolio import PortfolioConfig
        from finplan_suite.core.hammer.strategies import StrategyConfig, StrategyMode

        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100, 101, 102, 103, 104, 105],
                "BBB": [50, 50, 50, 51, 52, 53],
                "SPY": [100, 100, 100, 100, 100, 100],
            },
            index=dates,
        )
        coverage = {
            "AAA": (dates[0].date(), dates[-1].date()),
            "BBB": (dates[2].date(), dates[-1].date()),
            "SPY": (dates[0].date(), dates[-1].date()),
        }

        config = PortfolioConfig(
            tickers=["AAA", "BBB"],
            target_weights={"AAA": 0.5, "BBB": 0.5},
            benchmark="SPY",
            initial_capital=100000,
            start_date=dates[0].date(),
            end_date=dates[-1].date(),
        )
        strategy = StrategyConfig(mode=StrategyMode.DRIFT, drift_threshold=0.5)

        with patch("finplan_suite.core.hammer.backtest.fetch_prices", return_value=(prices, coverage)):
            result = BacktestEngine(config, strategy).run()

        assert result.effective_start == dates[2].date()
        assert any("BBB inception/data starts at" in w for w in result.data_warnings)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

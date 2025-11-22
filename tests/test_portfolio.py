"""
Tests for portfolio optimization and efficient frontier.
"""
import pytest
import numpy as np
from finplan_suite.core.portfolio import (
    efficient_frontier,
    max_sharpe,
    build_frontier,
    FrontierResult
)


@pytest.mark.unit
class TestEfficientFrontier:
    """Tests for efficient frontier construction."""

    def test_single_asset(self):
        """Frontier with single asset should return that asset."""
        mu = np.array([0.08])
        cov = np.array([[0.04]])

        W, rets, risks = efficient_frontier(mu, cov, k=5)

        assert len(W) > 0
        assert len(rets) > 0
        assert len(risks) > 0
        # Single asset: all weights should be 1.0 for that asset
        np.testing.assert_array_almost_equal(W[0], [1.0])

    def test_two_assets_basic(self):
        """Basic two-asset frontier should work."""
        mu = np.array([0.06, 0.10])
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])

        W, rets, risks = efficient_frontier(mu, cov, k=10)

        assert len(W) > 0
        # All weights should sum to 1 (fully invested)
        for w in W:
            np.testing.assert_almost_equal(np.sum(w), 1.0)
        # All weights should be non-negative (long-only)
        for w in W:
            assert np.all(w >= -1e-6)  # Small tolerance for numerical errors

    def test_position_limit(self):
        """Position limits should be respected."""
        mu = np.array([0.06, 0.10])
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])

        W, rets, risks = efficient_frontier(mu, cov, k=10, w_max=0.5)

        # All weights should be <= 0.5 (with small tolerance)
        for w in W:
            assert np.all(w <= 0.5 + 1e-6)


@pytest.mark.unit
class TestMaxSharpe:
    """Tests for max Sharpe ratio optimization."""

    def test_max_sharpe_basic(self):
        """Max Sharpe should return valid weights."""
        mu = np.array([0.06, 0.10, 0.08])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.06]
        ])

        weights, ret, risk, sr = max_sharpe(mu, cov, rf=0.02)

        # Check weights are valid
        np.testing.assert_almost_equal(np.sum(weights), 1.0)
        assert np.all(weights >= -1e-6)  # Non-negative
        assert risk > 0
        assert sr > 0  # Should be positive with reasonable inputs

    def test_max_sharpe_with_position_limit(self):
        """Max Sharpe with position limits."""
        mu = np.array([0.06, 0.10])
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])

        weights, ret, risk, sr = max_sharpe(mu, cov, rf=0.02, w_max=0.6)

        # Check position limits
        assert np.all(weights <= 0.6 + 1e-6)
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_max_sharpe_higher_return_asset_preferred(self):
        """Higher return asset should get more weight (all else equal)."""
        # Asset 2 has higher return and same risk
        mu = np.array([0.05, 0.10])
        cov = np.array([
            [0.04, 0.0],
            [0.0, 0.04]
        ])

        weights, ret, risk, sr = max_sharpe(mu, cov, rf=0.02)

        # Asset 2 (higher return) should dominate
        assert weights[1] > weights[0]


@pytest.mark.unit
class TestBuildFrontier:
    """Tests for complete frontier building."""

    def test_build_frontier_returns_result(self):
        """build_frontier should return FrontierResult."""
        mu = np.array([0.06, 0.08, 0.10])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.06, 0.02],
            [0.005, 0.02, 0.09]
        ])

        result = build_frontier(mu, cov, rf=0.02, k=15)

        assert isinstance(result, FrontierResult)
        assert len(result.returns) > 0
        assert len(result.risks) > 0
        assert result.weights.shape[0] > 0
        assert len(result.max_sharpe_w) == 3
        assert result.max_sharpe_sr > 0

    def test_frontier_monotonic_risk_return(self):
        """Higher risk should generally yield higher return on frontier."""
        mu = np.array([0.05, 0.08, 0.12])
        cov = np.array([
            [0.02, 0.005, 0.001],
            [0.005, 0.05, 0.01],
            [0.001, 0.01, 0.10]
        ])

        result = build_frontier(mu, cov, k=20)

        # Returns should generally increase with risk (with small tolerance for numerical noise)
        # Check that returns are mostly increasing
        increasing_count = sum(
            result.returns[i] <= result.returns[i+1] + 1e-4
            for i in range(len(result.returns) - 1)
        )
        # At least 80% should be increasing
        assert increasing_count >= 0.8 * (len(result.returns) - 1)


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_returns(self):
        """Should handle negative expected returns."""
        mu = np.array([-0.02, 0.05])
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.06]
        ])

        result = build_frontier(mu, cov)
        assert len(result.returns) > 0

    def test_zero_correlation(self):
        """Should handle zero correlation between assets."""
        mu = np.array([0.06, 0.08])
        cov = np.array([
            [0.04, 0.0],
            [0.0, 0.06]
        ])

        W, rets, risks = efficient_frontier(mu, cov)
        assert len(W) > 0

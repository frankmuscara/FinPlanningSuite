"""
Tests for Monte Carlo retirement simulation.
"""
import pytest
import numpy as np
from finplan_suite.core.monte_carlo import MCInputs, simulate_paths


@pytest.mark.unit
class TestMCInputs:
    """Tests for MCInputs dataclass."""

    def test_default_values(self):
        """MCInputs should have sensible defaults."""
        params = MCInputs(
            init_value=100000,
            annual_invest=10000,
            horizon_years=30,
            years_to_retire=10,
            desired_income=50000,
            pension=0,
            social_security=0,
            inflation=0.02,
            mu=0.07,
            sigma=0.15
        )
        assert params.n_paths == 10000
        assert params.seed == 42


@pytest.mark.unit
class TestSimulatePathsBasic:
    """Basic tests for Monte Carlo simulation."""

    def test_accumulation_only(self):
        """Test accumulation phase with no retirement."""
        params = MCInputs(
            init_value=100000,
            annual_invest=10000,
            horizon_years=10,
            years_to_retire=10,  # Retire after horizon (never retire in this sim)
            desired_income=0,
            pension=0,
            social_security=0,
            inflation=0.02,
            mu=0.07,
            sigma=0.15,
            n_paths=1000,
            seed=42
        )

        result = simulate_paths(params)

        assert "years" in result
        assert "mean_path" in result
        assert "success_rate" in result
        assert len(result["years"]) == 10
        assert len(result["mean_path"]) == 10

        # With positive returns and contributions, mean should grow
        assert result["mean_path"][-1] > params.init_value

    def test_retirement_only(self):
        """Test retirement phase with immediate retirement."""
        params = MCInputs(
            init_value=1000000,
            annual_invest=0,
            horizon_years=20,
            years_to_retire=0,  # Retire immediately
            desired_income=40000,
            pension=0,
            social_security=0,
            inflation=0.02,
            mu=0.05,
            sigma=0.10,
            n_paths=1000,
            seed=42
        )

        result = simulate_paths(params)

        assert len(result["mean_path"]) == 20
        # Should have some success (not all ruined)
        assert result["success_rate"] > 0

    def test_reproducibility_with_seed(self):
        """Same seed should produce same results."""
        params = MCInputs(
            init_value=100000,
            annual_invest=5000,
            horizon_years=15,
            years_to_retire=5,
            desired_income=30000,
            pension=0,
            social_security=20000,
            inflation=0.025,
            mu=0.06,
            sigma=0.12,
            n_paths=1000,
            seed=123
        )

        result1 = simulate_paths(params)
        result2 = simulate_paths(params)

        np.testing.assert_array_equal(result1["mean_path"], result2["mean_path"])
        assert result1["success_rate"] == result2["success_rate"]

    def test_percentiles_ordered(self):
        """Percentiles should be properly ordered."""
        params = MCInputs(
            init_value=500000,
            annual_invest=10000,
            horizon_years=25,
            years_to_retire=10,
            desired_income=40000,
            pension=5000,
            social_security=15000,
            inflation=0.02,
            mu=0.07,
            sigma=0.15,
            n_paths=5000,
            seed=42
        )

        result = simulate_paths(params)
        pct = result["final_value_percentiles"]

        # Percentiles should be ordered
        assert pct[5] <= pct[25]
        assert pct[25] <= pct[50]
        assert pct[50] <= pct[75]
        assert pct[75] <= pct[95]


@pytest.mark.unit
class TestMonteCarloEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_return_zero_sigma(self):
        """With zero return and zero volatility, should get deterministic result."""
        params = MCInputs(
            init_value=100000,
            annual_invest=10000,
            horizon_years=5,
            years_to_retire=5,
            desired_income=0,
            pension=0,
            social_security=0,
            inflation=0.0,
            mu=0.0,
            sigma=0.0,
            n_paths=100,
            seed=42
        )

        result = simulate_paths(params)

        # With no returns, final value should be init + contributions
        expected_final = 100000 + 10000 * 5
        # All paths should be identical (deterministic)
        np.testing.assert_almost_equal(
            result["final_value_percentiles"][50],
            expected_final,
            decimal=0
        )

    def test_high_withdrawal_causes_ruin(self):
        """Excessive withdrawals should cause ruin."""
        params = MCInputs(
            init_value=100000,
            annual_invest=0,
            horizon_years=10,
            years_to_retire=0,  # Retire immediately
            desired_income=50000,  # Withdraw 50k/year from 100k
            pension=0,
            social_security=0,
            inflation=0.0,
            mu=0.01,  # Very low return
            sigma=0.05,
            n_paths=1000,
            seed=42
        )

        result = simulate_paths(params)

        # Should have significant ruin rate
        assert result["ruin_rate"] > 0.5

    def test_pension_covers_income(self):
        """When pension covers income, should not deplete portfolio."""
        params = MCInputs(
            init_value=100000,
            annual_invest=0,
            horizon_years=20,
            years_to_retire=0,
            desired_income=40000,
            pension=20000,
            social_security=20000,  # Total = 40k (covers income)
            inflation=0.0,
            mu=0.05,
            sigma=0.10,
            n_paths=1000,
            seed=42
        )

        result = simulate_paths(params)

        # Portfolio should not be drawn on, so success should be very high
        assert result["success_rate"] > 0.95

    def test_success_plus_ruin_equals_one(self):
        """Success rate + ruin rate should equal 1.0."""
        params = MCInputs(
            init_value=250000,
            annual_invest=15000,
            horizon_years=30,
            years_to_retire=15,
            desired_income=45000,
            pension=0,
            social_security=18000,
            inflation=0.025,
            mu=0.06,
            sigma=0.14,
            n_paths=2000,
            seed=42
        )

        result = simulate_paths(params)

        np.testing.assert_almost_equal(
            result["success_rate"] + result["ruin_rate"],
            1.0,
            decimal=6
        )


@pytest.mark.unit
class TestInflationHandling:
    """Tests for inflation adjustment logic."""

    def test_inflation_increases_withdrawal(self):
        """Withdrawals should increase with inflation over time."""
        # We can't directly observe withdrawals, but we can infer from
        # portfolio depletion rate with and without inflation
        params_no_inflation = MCInputs(
            init_value=500000,
            annual_invest=0,
            horizon_years=20,
            years_to_retire=0,
            desired_income=30000,
            pension=0,
            social_security=0,
            inflation=0.0,
            mu=0.04,
            sigma=0.0,  # No volatility for deterministic test
            n_paths=1,
            seed=42
        )

        params_with_inflation = MCInputs(
            init_value=500000,
            annual_invest=0,
            horizon_years=20,
            years_to_retire=0,
            desired_income=30000,
            pension=0,
            social_security=0,
            inflation=0.03,
            mu=0.04,
            sigma=0.0,
            n_paths=1,
            seed=42
        )

        result_no_infl = simulate_paths(params_no_inflation)
        result_with_infl = simulate_paths(params_with_inflation)

        # With inflation, should deplete faster (lower final value)
        assert result_with_infl["final_value_percentiles"][50] < \
               result_no_infl["final_value_percentiles"][50]

# finplan_suite/core/monte_carlo.py
"""
Vectorized Monte Carlo engine for retirement cashflow planning.

Inputs (all decimals, not %):
- init_value: starting portfolio value
- annual_invest: contribution per year until retirement
- horizon_years: total years simulated
- years_to_retire: years from now until retirement starts (integer)
- desired_income: nominal desired first-year retirement income
- pension: nominal pension first-year benefit (if any)
- social_security: nominal SS first-year benefit (if any)
- inflation: annual inflation assumption (decimal)
- mu: expected arithmetic annual return (decimal)
- sigma: annual standard deviation of returns (decimal)
- n_paths: number of Monte Carlo paths
- seed: RNG seed (optional, int)

Outputs:
- dict with:
  - years: np.array [1..horizon_years]
  - mean_path: shape (years,)
  - p5_path / p95_path: shape (years,)
  - final_value_percentiles: dict {5, 25, 50, 75, 95}
  - success_rate: fraction of paths that never hit zero before final year
  - ruin_rate: 1 - success_rate
  - ruin_year_pct: optional histogram-like array: pct of paths ruined by each year
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class MCInputs:
    init_value: float
    annual_invest: float
    horizon_years: int
    years_to_retire: int
    desired_income: float
    pension: float
    social_security: float
    inflation: float
    mu: float
    sigma: float
    n_paths: int = 10000
    seed: int | None = 42

def simulate_paths(params: MCInputs):
    n = params.n_paths
    T = params.horizon_years
    Tr = max(0, int(params.years_to_retire))

    rng = np.random.default_rng(params.seed)
    # Draw returns: shape (n, T)
    R = rng.normal(loc=params.mu, scale=params.sigma, size=(n, T))

    # Values matrix [n, T], start with init across all paths
    V = np.zeros((n, T), dtype=float)
    v = np.full(n, params.init_value, dtype=float)

    # Retirement cashflows (inflate after retirement begins)
    income = params.desired_income
    pen = params.pension
    ss = params.social_security
    infl = 1.0 + params.inflation

    # Track ruin whenever value hits 0 at or before year t
    ruined = np.zeros(n, dtype=bool)
    ruin_year_count = np.zeros(T, dtype=float)

    for t in range(T):
        if t < Tr:
            # Accumulation years: contribute then grow
            v = (v + params.annual_invest) * (1.0 + R[:, t])
        else:
            # Retirement years: inflate cashflows annually, then withdraw deficit
            if t == Tr:
                # first retirement year uses base income/pen/ss
                cur_income = income
                cur_pen = pen
                cur_ss = ss
            else:
                cur_income *= infl
                cur_pen *= infl
                cur_ss *= infl

            withdrawal = np.maximum(0.0, cur_income - cur_pen - cur_ss)
            v = (v - withdrawal) * (1.0 + R[:, t])

        # If value drops below 0, clamp to 0 and mark ruin
        newly_ruined = (v <= 0.0) & (~ruined)
        if np.any(newly_ruined):
            ruin_year_count[t] = newly_ruined.mean()  # pct ruined at year t
            ruined = ruined | newly_ruined
            v[newly_ruined] = 0.0

        V[:, t] = v

    # Stats
    mean_path = V.mean(axis=0)
    p5_path = np.percentile(V, 5, axis=0)
    p95_path = np.percentile(V, 95, axis=0)

    final_vals = V[:, -1]
    final_pct = {
        5: float(np.percentile(final_vals, 5)),
        25: float(np.percentile(final_vals, 25)),
        50: float(np.percentile(final_vals, 50)),
        75: float(np.percentile(final_vals, 75)),
        95: float(np.percentile(final_vals, 95)),
    }

    success_rate = float((~ruined).mean())
    ruin_rate = 1.0 - success_rate

    return {
        "years": np.arange(1, T + 1),
        "mean_path": mean_path,
        "p5_path": p5_path,
        "p95_path": p95_path,
        "final_value_percentiles": final_pct,
        "success_rate": success_rate,
        "ruin_rate": ruin_rate,
        "ruin_year_pct": ruin_year_count,  # pct ruined by each specific year
    }


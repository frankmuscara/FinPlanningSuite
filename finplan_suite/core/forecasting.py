# finplan_suite/core/forecasting.py
"""
Economic projections engine (lightweight, UI-friendly).

We project 10 years for:
- Real GDP growth (y/y)
- CPI inflation (y/y)
- Real short rate (ex-ante)
- Term premium (long - short)

Outputs are medians (point forecasts) with simple bands for UI display.
"""

from dataclasses import dataclass
import numpy as np

YEARS = 10

@dataclass
class MacroInputs:
    gdp_real: float       # long-run real GDP growth (e.g., 0.017 = 1.7%)
    inflation: float      # long-run CPI (e.g., 0.025 = 2.5%)
    rstar: float          # real short rate (long-run, e.g., 0.01 = 1.0%)
    term_premium: float   # long-term term premium (e.g., 0.015 = 1.5%)
    shock_level: float = 0.0  # discretionary overlay: [-1..+1] risk tone

@dataclass
class MacroProjection:
    years: np.ndarray
    gdp_path: np.ndarray
    cpi_path: np.ndarray
    real_short_path: np.ndarray
    term_premium_path: np.ndarray
    bands: dict  # {"gdp": (p10,p90), ...}

def project_macro(m: MacroInputs) -> MacroProjection:
    t = np.arange(1, YEARS + 1)

    # mean-reverting drift to long-run levels with small randomization around tone
    def mean_revert(level, speed=0.30):
        base = np.full(YEARS, level, dtype=float)
        # small tone tilt
        tilt = (m.shock_level * np.linspace(0.5, 0.0, YEARS)) * level
        return base + tilt

    gdp = mean_revert(m.gdp_real)
    cpi = mean_revert(m.inflation)
    rsh = mean_revert(m.rstar)
    tp  = mean_revert(m.term_premium)

    # create simple symmetric bands for display (not used by CMA math)
    def band(x, width=0.35):
        p10 = x * (1 - width)
        p90 = x * (1 + width)
        return p10, p90

    bands = {
        "gdp": band(gdp),
        "cpi": band(cpi),
        "rsh": band(rsh, 0.50),
        "tp":  band(tp,  0.50),
    }

    return MacroProjection(
        years=t,
        gdp_path=gdp,
        cpi_path=cpi,
        real_short_path=rsh,
        term_premium_path=tp,
        bands=bands
    )


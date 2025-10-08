# finplan_suite/core/cma.py
"""
Capital Market Assumptions (CMAs) with blendable components:
- Macro-linked model
- Valuation-anchored model
- Historical/Bayesian prior

Exports:
- build_cma_blend(...)   -> CMA object from macro/valuation/prior + weights
- derive_cma_from_macro(...) -> macro-only CMA (compat for EconomicView)
- save_cma_json, load_cma_json
- utilities: default_prior(), _default_corr(), _base_vols()
"""

from dataclasses import dataclass
import numpy as np
import json, os, time, datetime
from pathlib import Path
from typing import Optional

# ---------- Universe ----------

TICKERS = ["VOO", "IJR", "NEAR", "BOND", "TLT", "QLEIX", "VEMBX", "PDBC"]

# ---------- Dataclass ----------

@dataclass
class CMA:
    tickers: list
    exp_returns: np.ndarray  # (n,)
    cov: np.ndarray          # (n,n)
    rf: float
    notes: str = ""
    meta: Optional[dict] = None  # as_of, horizon_years, method_version, weights, etc.

# ---------- Corr/Vol helpers ----------

def _default_corr():
    return np.array([
        [1.00, 0.85, 0.00, 0.10, -0.20, 0.40, 0.30, 0.05],
        [0.85, 1.00, 0.00, 0.10, -0.15, 0.35, 0.25, 0.05],
        [0.00, 0.00, 1.00, 0.20,  0.10, 0.00, 0.10, 0.00],
        [0.10, 0.10, 0.20, 1.00,  0.40, 0.20, 0.35, 0.05],
        [-0.20,-0.15, 0.10, 0.40, 1.00, 0.00, 0.10, -0.05],
        [0.40, 0.35, 0.00, 0.20,  0.00, 1.00, 0.25, 0.10],
        [0.30, 0.25, 0.10, 0.35,  0.10, 0.25, 1.00, 0.05],
        [0.05, 0.05, 0.00, 0.05, -0.05, 0.10, 0.05, 1.00],
    ])

def _base_vols():
    return np.array([0.16, 0.21, 0.01, 0.06, 0.14, 0.10, 0.08, 0.18])

def _cov_from_vols_corr(vols, corr):
    D = np.diag(vols)
    return D @ corr @ D

def _shrink_to_identity(cov: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Shrink correlation toward identity to stabilize covariance."""
    s = np.sqrt(np.diag(cov))
    # avoid divide-by-zero if any vol is 0
    s_safe = np.where(s == 0, 1.0, s)
    corr = cov / np.outer(s_safe, s_safe)
    shrunk_corr = (1 - alpha) * corr + alpha * np.eye(len(s))
    return np.diag(s) @ shrunk_corr @ np.diag(s)

# ---------- Guardrails & Uncertainty ----------

def _clamp_expected_returns(mu: np.ndarray, tickers: list[str] = TICKERS) -> np.ndarray:
    """Clamp per-asset expected returns to guardrails."""
    bounds = {
        "VOO":  (0.03, 0.12),
        "IJR":  (0.04, 0.14),
        "NEAR": (0.00, 0.08),
        "BOND": (0.01, 0.09),
        "TLT":  (0.00, 0.10),
        "QLEIX":(0.02, 0.12),
        "VEMBX":(0.02, 0.11),
        "PDBC": (0.00, 0.08),  # generally near inflation
    }
    default_bounds = (0.00, 0.15)
    out = mu.copy()
    for i, t in enumerate(tickers):
        lo, hi = bounds.get(t, default_bounds)
        out[i] = float(np.clip(out[i], lo, hi))
    return out

def _simple_se(mu_macro: np.ndarray, mu_val: np.ndarray, priors: np.ndarray,
               w_macro: float, w_val: float, w_prior: float) -> np.ndarray:
    """Crude SE proxy; replace with parameter-uncertainty later."""
    se_macro = 0.015
    se_val   = 0.012
    se_prior = 0.008
    se = np.sqrt((w_macro*se_macro)**2 + (w_val*se_val)**2 + (w_prior*se_prior)**2)
    return np.full_like(mu_macro, se, dtype=float)

# ---------- Priors & Models ----------

def default_prior():
    """Long-run priors (nominal)."""
    return np.array([
        0.075,  # VOO
        0.085,  # IJR
        0.030,  # NEAR
        0.045,  # BOND
        0.048,  # TLT
        0.060,  # QLEIX
        0.055,  # VEMBX
        0.040,  # PDBC
    ])

def macro_model_returns(gdp, cpi, rstar, term_premium,
                        small_premium=0.012, credit_premium=0.015,
                        em_spread=0.012, inflation_beta=0.8,
                        duration_drag=0.002, spread_drag=0.003):
    """Transparent macro-linked recipe (year-1 levels)."""
    rf = float(rstar + cpi)
    ERP_large = 0.045 + 0.5 * (gdp - 0.017)   # tilt with growth vs 1.7% trend
    ERP_small = ERP_large + small_premium
    exp = {
        "VOO":   rf + ERP_large,
        "IJR":   rf + ERP_small,
        "NEAR":  rf,
        "BOND":  rf + 0.8*term_premium - spread_drag,
        "TLT":   rf + term_premium - duration_drag,
        "QLEIX": rf + 0.5*term_premium + credit_premium,
        "VEMBX": rf + 0.6*term_premium + em_spread,
        "PDBC":  inflation_beta * cpi,
    }
    mu = np.array([exp[t] for t in TICKERS])
    return rf, mu

def valuation_model_returns(
    earnings_yield_large,  # e.g., 1/CAPE or fwd EY
    small_vs_large_valuation_gap=0.0,   # + => small cheaper
    core_bond_yield=0.045,
    tlt_yield=0.048,
    credit_oas=0.015,
    em_spread=0.018,
    commodities_carry=0.0,
    half_life_years=7,
    horizon_years=10
):
    """Valuation-anchored: equities EY + mean reversion; bonds ~ yield; credit adds carry."""
    ey_target = 0.05
    alpha = 1 - 0.5 ** (1 / max(1, half_life_years))  # step toward target
    delta_val = alpha * (ey_target - earnings_yield_large)
    small_uplift = np.clip(0.01 + small_vs_large_valuation_gap, 0.0, 0.03)
    exp = {
        "VOO":   earnings_yield_large + delta_val,
        "IJR":   earnings_yield_large + delta_val + small_uplift,
        "NEAR":  core_bond_yield * 0.6,     # short-end proxy
        "BOND":  core_bond_yield + 0.4*credit_oas,
        "TLT":   tlt_yield,
        "QLEIX": core_bond_yield + 0.6*credit_oas,
        "VEMBX": core_bond_yield + 0.5*em_spread,
        "PDBC":  commodities_carry,
    }
    mu = np.array([exp[t] for t in TICKERS])
    rf = float(core_bond_yield * 0.6)  # not used if macro rf provided
    return rf, mu

# ---------- Blended CMA ----------

def build_cma_blend(
    # macro
    gdp, cpi, rstar, term_premium,
    # valuation
    earnings_yield_large,
    small_vs_large_valuation_gap,
    core_bond_yield, tlt_yield, credit_oas, em_spread, commodities_carry,
    # weights
    w_macro=0.5, w_val=0.3, w_prior=0.2,
    # constants
    small_premium=0.012, credit_premium=0.015, priors=None,
    vols=None, corr=None,
    meta_overrides: Optional[dict] = None
) -> CMA:
    priors = default_prior() if priors is None else priors
    vols = _base_vols() if vols is None else vols
    corr = _default_corr() if corr is None else corr

    rf_macro, mu_macro = macro_model_returns(
        gdp, cpi, rstar, term_premium,
        small_premium=small_premium,
        credit_premium=credit_premium,
        em_spread=em_spread
    )
    _rf_val, mu_val = valuation_model_returns(
        earnings_yield_large,
        small_vs_large_valuation_gap=small_vs_large_valuation_gap,
        core_bond_yield=core_bond_yield,
        tlt_yield=tlt_yield,
        credit_oas=credit_oas,
        em_spread=em_spread,
        commodities_carry=commodities_carry
    )

    # Blend expected returns, clamp for guardrails
    mu = (w_macro * mu_macro) + (w_val * mu_val) + (w_prior * priors)
    mu = _clamp_expected_returns(mu, TICKERS)

    # Covariance (with shrink)
    raw_cov = _cov_from_vols_corr(vols, corr)
    cov = _shrink_to_identity(raw_cov, alpha=0.25)

    # Choose rf from macro by default
    rf = float(rf_macro)

    meta = {
        "as_of": datetime.date.today().isoformat(),
        "horizon_years": 10,
        "method_version": "v1.0-blend",
        "weights": {"macro": w_macro, "valuation": w_val, "prior": w_prior},
        "se_exp_returns": _simple_se(mu_macro, mu_val, priors, w_macro, w_val, w_prior).tolist()
    }
    if meta_overrides:
        meta.update(meta_overrides)

    return CMA(tickers=TICKERS, exp_returns=mu, cov=cov, rf=rf, notes="Blended CMA", meta=meta)

# --- Compatibility wrapper for EconomicView ---

def derive_cma_from_macro(
    gdp: float,
    cpi: float,
    real_short: float,
    term_premium: float,
    *,
    vols: np.ndarray | None = None,
    corr: np.ndarray | None = None,
    method_version: str = "v1.0-macro"
) -> CMA:
    """Build a CMA using ONLY macro-linked returns (no valuation/prior blend)."""
    rf, mu_macro = macro_model_returns(gdp, cpi, real_short, term_premium)

    # guardrails
    mu_clamped = _clamp_expected_returns(mu_macro, TICKERS)

    # covariance (vols + corr, then shrink)
    vols = _base_vols() if vols is None else vols
    corr = _default_corr() if corr is None else corr
    raw_cov = _cov_from_vols_corr(vols, corr)
    cov = _shrink_to_identity(raw_cov, alpha=0.25)

    meta = {
        "as_of": datetime.date.today().isoformat(),
        "horizon_years": 10,
        "method_version": method_version,
        "weights": {"macro": 1.0, "valuation": 0.0, "prior": 0.0},
        "se_exp_returns": _simple_se(mu_macro, mu_macro, mu_macro, 1.0, 0.0, 0.0).tolist()
    }

    return CMA(
        tickers=TICKERS,
        exp_returns=mu_clamped,
        cov=cov,
        rf=float(rf),
        notes="Derived from macro inputs only",
        meta=meta
    )

# ---------- Save/Load with history ----------

def _ensure_dirs(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_cma_json(cma: CMA, path: str = "data/cma.json"):
    """Save current CMA to data/cma.json AND a timestamped copy in data/cma_history/."""
    _ensure_dirs(path)
    obj = {
        "meta": cma.meta or {},
        "notes": cma.notes,
        "tickers": cma.tickers,
        "exp_returns": cma.exp_returns.tolist(),
        "cov": cma.cov.tolist(),
        "rf": float(cma.rf),
    }
    # current
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    # history
    hist_dir = Path("data/cma_history")
    hist_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    as_of = (cma.meta or {}).get("as_of", datetime.date.today().isoformat())
    method = (cma.meta or {}).get("method_version", "v1")
    hist_path = hist_dir / f"cma_{method}_{as_of}_{ts}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_cma_json(path: str = "data/cma.json") -> Optional[CMA]:
    """Load a CMA from JSON. Returns a CMA or None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return CMA(
        tickers=obj["tickers"],
        exp_returns=np.array(obj["exp_returns"], dtype=float),
        cov=np.array(obj["cov"], dtype=float),
        rf=float(obj["rf"]),
        notes=obj.get("notes", ""),
        meta=obj.get("meta", {})
    )

# finplan_suite/core/portfolio.py
"""
Efficient frontier + max-Sharpe optimizer using cvxpy.
Constraints:
  - long-only (w >= 0)
  - fully invested (sum w = 1)
  - optional per-asset caps
"""

from dataclasses import dataclass
import numpy as np
import cvxpy as cp

@dataclass
class FrontierResult:
    returns: np.ndarray
    risks: np.ndarray
    weights: np.ndarray  # shape (k, n)
    max_sharpe_w: np.ndarray
    max_sharpe_ret: float
    max_sharpe_risk: float
    max_sharpe_sr: float

def efficient_frontier(mu, cov, k=30, w_max=None):
    """
    Build frontier by sweeping target returns between min and max attainable (long-only).
    mu: (n,)
    cov: (n,n)
    w_max: optional per-asset cap (float) -> w_i <= w_max
    """
    n = len(mu)
    Sigma = cov
    R = mu

    # ---- min-variance corner (long-only, fully invested, optional cap) ----
    w_min = cp.Variable(n)
    minvar_constraints = [cp.sum(w_min) == 1, w_min >= 0]
    if w_max is not None:
        minvar_constraints.append(w_min <= w_max)

    prob_minvar = cp.Problem(cp.Minimize(cp.quad_form(w_min, Sigma)), minvar_constraints)
    prob_minvar.solve(solver=cp.ECOS, verbose=False)

    if w_min.value is None:
        # Fallback if solver hiccups: use equal weights for r_min bound
        r_min = float(R @ (np.ones(n) / n))
    else:
        r_min = float(R @ w_min.value)

    # Max return (long-only) = allocate to the highest-return asset
    r_max = float(np.max(R))

    targets = np.linspace(r_min, r_max, k)
    risks, rets, W = [], [], []

    for r_tgt in targets:
        w = cp.Variable(n)
        cons = [cp.sum(w) == 1, w >= 0, R @ w >= r_tgt]
        if w_max is not None:
            cons.append(w <= w_max)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)
        prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            # infeasible target, skip
            continue

        w_opt = np.clip(w.value, 0, 1)
        W.append(w_opt)
        rets.append(float(R @ w_opt))
        risks.append(float(np.sqrt(w_opt @ Sigma @ w_opt)))

    return np.array(W), np.array(rets), np.array(risks)

def max_sharpe(mu, cov, rf=0.0, w_max=None):
    n = len(mu)
    w = cp.Variable(n)
    # Maximize (mu^T w - rf) / sqrt(w' Σ w)  -> maximize numerator for fixed denom via SOC trick
    # Equivalent convex form: maximize t  s.t.  (mu - rf1)^T w >= t,  ||Σ^{1/2} w||_2 <= 1,  w constraints
    # Simpler practical approach: maximize (mu - rf)^T w - λ * w'Σw and search λ. We'll do a small grid.
    lambdas = np.logspace(-4, 1, 40)
    best = None
    Sigma = cov
    excess = mu - rf

    for lam in lambdas:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= 0]
        if w_max is not None:
            constraints.append(w <= w_max)
        obj = cp.Maximize(excess @ w - lam * cp.quad_form(w, Sigma))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        if w.value is None:
            continue
        wv = np.clip(w.value, 0, 1)
        ret = float(mu @ wv)
        risk = float(np.sqrt(wv @ Sigma @ wv))
        sr = (ret - rf) / risk if risk > 0 else -1e9
        if (best is None) or (sr > best[3]):
            best = (wv, ret, risk, sr)
    if best is None:
        raise RuntimeError("Max Sharpe optimization failed")
    return best  # weights, ret, risk, SR

def build_frontier(mu, cov, rf=0.0, k=30, w_max=None) -> FrontierResult:
    W, rets, risks = efficient_frontier(mu, cov, k=k, w_max=w_max)
    w_star, r_star, s_star, sr_star = max_sharpe(mu, cov, rf=rf, w_max=w_max)
    return FrontierResult(
        returns=rets, risks=risks, weights=W,
        max_sharpe_w=w_star, max_sharpe_ret=r_star,
        max_sharpe_risk=s_star, max_sharpe_sr=sr_star
    )


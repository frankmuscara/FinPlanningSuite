# finplan_suite/core/forecast_models.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import statsmodels.api as sm

from .fred_utilities import fetch_fred, resample_quarterly_mean, pct_change_series, annualize_qoq

QUARTER_FREQ = "QE-DEC" #updating to new pandas resampling standard, avoids deprecation error

@dataclass
class GDPInputs:
    start: str = "2010-01-01"
    end: Optional[str] = None
    # mapping of series names to FRED tickers
    series: Dict[str, str] = None

    def __post_init__(self):
        if self.series is None:
            self.series = {
                "DGORDER": "DGORDER",        # Durable goods orders
                "DRCCLACBS": "DRCCLACBS",    # Credit card delinquency rate
                "RSAFS": "RSAFS",            # Retail sales
                "INDPRO": "INDPRO",          # Industrial production
                "NPPTTL": "NPPTTL",          # Private payrolls (BLS level)
                "GDPC1": "GDPC1",            # Real GDP (chain 2017$)
            }

@dataclass
class GDPForecastResult:
    df_quarterly: pd.DataFrame
    model_summary: str
    one_step_qoq: float        # QoQ (not annualized)
    one_step_annualized: float # (1+qoq)^4 - 1
    path_annualized_10y: np.ndarray  # quarterly annualized growth for 10y (len=40)

class GDPGrowthForecaster:
    """
    Builds a small, interpretable forecaster:
      - Pulls monthly series from FRED, resamples to Q
      - Creates growth features
      - Fits OLS (for summary) and RidgeCV (robustness)
      - Produces 1-step forecast and a 10y path that mean-reverts to a long-run level
    """
    def __init__(self, cfg: GDPInputs):
        self.cfg = cfg
        self.df_q = None
        self.ols = None
        self.ridge = None
        self.scaler = None

    def load_and_prepare(self) -> pd.DataFrame:
        s = self.cfg.series
        # fetch monthly
        dfm = pd.DataFrame({
            "DGORDER": fetch_fred(s["DGORDER"], start=self.cfg.start, end=self.cfg.end),
            "DRCCLACBS": fetch_fred(s["DRCCLACBS"], start=self.cfg.start, end=self.cfg.end),
            "RSAFS": fetch_fred(s["RSAFS"], start=self.cfg.start, end=self.cfg.end),
            "INDPRO": fetch_fred(s["INDPRO"], start=self.cfg.start, end=self.cfg.end),
            "NPPTTL": fetch_fred(s["NPPTTL"], start=self.cfg.start, end=self.cfg.end),
        })
        gdp = fetch_fred(s["GDPC1"], start=self.cfg.start, end=self.cfg.end)
        # quarterly resample
        q = resample_quarterly_mean(dfm)
        q["GDPC1"] = gdp.resample("QE-DEC").last()  # GDP is already Q — last aligns
        q = q.ffill().dropna()

        # features
        q["Retail_Growth"]   = pct_change_series(q["RSAFS"])
        q["IndProd_Growth"]  = pct_change_series(q["INDPRO"])
        q["Payroll_Growth"]  = pct_change_series(q["NPPTTL"])
        q["DGORDER_Growth"]  = pct_change_series(q["DGORDER"])
        # delinquency is a level %; include level and change
        q["Delinq_Level"]    = q["DRCCLACBS"]
        q["Delinq_Change"]   = q["DRCCLACBS"].diff()
        q["GDP_Growth_QoQ"]  = pct_change_series(q["GDPC1"])

        q = q.dropna()
        self.df_q = q
        return q

    def _feature_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        cols = [
            "DGORDER_Growth", "Delinq_Level", "Delinq_Change",
            "Retail_Growth", "IndProd_Growth", "Payroll_Growth"
        ]
        X = self.df_q[cols].copy()
        y = self.df_q["GDP_Growth_QoQ"].copy()
        return X, y

    def fit(self) -> str:
        X, y = self._feature_target()
        # OLS (for diagnostics)
        X_ols = sm.add_constant(X)
        self.ols = sm.OLS(y, X_ols).fit()

        # Ridge (robust predictive model with CV)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.ridge = RidgeCV(alphas=(0.1, 0.3, 1.0, 3.0, 10.0), cv=5).fit(Xs, y)

        return self.ols.summary().as_text()

    def one_step_forecast(self) -> Tuple[float, float]:
        X, _ = self._feature_target()
        latest = X.iloc[[-1]]
        latest_s = self.scaler.transform(latest)
        qoq = float(self.ridge.predict(latest_s)[0])
        return qoq, float((1.0 + qoq)**4 - 1.0)

    def path_10y(self, long_run_gdp_real: float, half_life_years: float = 4.0) -> np.ndarray:
        """
        Produce a 10-year quarterly annualized growth path:
          - Start from 1-step annualized prediction
          - Mean-revert to long-run level with a chosen half-life
        """
        qoq, ann1 = self.one_step_forecast()
        Tq = 40  # 10y * 4 quarters
        path = np.zeros(Tq, dtype=float)
        path[0] = ann1
        # compute annualized half-life step per quarter
        step = 1 - 0.5**(1.0 / (max(0.5, half_life_years) * 4.0))
        for t in range(1, Tq):
            path[t] = path[t-1] + step * (long_run_gdp_real - path[t-1])
        return path

    def run(self, long_run_gdp_real: float) -> GDPForecastResult:
        self.load_and_prepare()
        summary = self.fit()
        qoq, ann = self.one_step_forecast()
        path = self.path_10y(long_run_gdp_real=long_run_gdp_real, half_life_years=4.0)
        return GDPForecastResult(
            df_quarterly=self.df_q.copy(),
            model_summary=summary,
            one_step_qoq=qoq,
            one_step_annualized=ann,
            path_annualized_10y=path
        )
# -------------- CPI Forecaster ---------------------------------

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from .fred_utilities import fetch_fred, resample_quarterly_mean, pct_change_series

@dataclass
class CPIInputs:
    start: str = "2010-01-01"
    end: Optional[str] = None
    series: Dict[str, str] = None

    def __post_init__(self):
        # CPI headline + useful drivers: breakevens, oil, wages, unemployment
        if self.series is None:
            self.series = {
                "CPI": "CPIAUCSL",      # CPI level (SA)
                "BE5Y": "T5YIE",        # 5y breakeven inflation
                "WTI": "DCOILWTICO",    # WTI crude oil
                "WAGES": "ECIWAG",      # Employment Cost Index: Wages
                "UNRATE": "UNRATE",     # Unemployment rate
            }

@dataclass
class CPIForecastResult:
    df_quarterly: pd.DataFrame
    one_step_yoy: float          # next-quarter YoY CPI (approx)
    path_yoy_10y: np.ndarray     # quarterly YoY CPI for 10y (len=40)
    model_alpha: float           # ridge alpha chosen
    coef: np.ndarray             # ridge coefficients (standardized)

class CPIForecaster:
    """
    CPI model:
      - Monthly FRED series → quarterly features
      - Features: breakeven, oil returns, wage growth, unemployment level & change
      - Target: YoY CPI (% change from a year ago)
      - Estimator: RidgeCV for stability
      - Output: 1-step forecast + 10y mean-reverting path
    """
    def __init__(self, cfg: CPIInputs):
        self.cfg = cfg
        self.df_q = None
        self.scaler = None
        self.ridge = None

    def load_and_prepare(self) -> pd.DataFrame:
        s = self.cfg.series
        # Pull monthly series
        cpi = fetch_fred(s["CPI"], start=self.cfg.start, end=self.cfg.end)
        be5 = fetch_fred(s["BE5Y"], start=self.cfg.start, end=self.cfg.end)
        wti = fetch_fred(s["WTI"], start=self.cfg.start, end=self.cfg.end)
        wag = fetch_fred(s["WAGES"], start=self.cfg.start, end=self.cfg.end)
        un  = fetch_fred(s["UNRATE"], start=self.cfg.start, end=self.cfg.end)

        dfm = pd.DataFrame({
            "CPI": cpi,
            "BE5Y": be5,
            "WTI": wti,
            "WAGES": wag,
            "UNRATE": un,
        })

        # Quarterly alignment
        q = resample_quarterly_mean(dfm).ffill()

        # Targets & features (YoY and QoQ where appropriate)
        q["CPI_YoY"]       = q["CPI"].pct_change(4)           # YoY CPI
        q["BE5Y_Level"]    = q["BE5Y"]                        # breakeven level
        q["Oil_Growth"]    = q["WTI"].pct_change(1)           # oil QoQ
        q["Wage_YoY"]      = q["WAGES"].pct_change(4)         # wages YoY
        q["Unemp_Level"]   = q["UNRATE"]
        q["Unemp_Change"]  = q["UNRATE"].diff(1)              # change in unemp

        q = q.dropna().copy()
        self.df_q = q
        return q

    def _feature_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        cols = ["BE5Y_Level", "Oil_Growth", "Wage_YoY", "Unemp_Level", "Unemp_Change"]
        X = self.df_q[cols].copy()
        y = self.df_q["CPI_YoY"].copy()
        return X, y

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        X, y = self._feature_target()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.ridge = RidgeCV(alphas=(0.1, 0.3, 1.0, 3.0, 10.0), cv=5).fit(Xs, y)
        return self.ridge

    def one_step_forecast(self) -> float:
        X, _ = self._feature_target()
        latest = X.iloc[[-1]]
        latest_s = self.scaler.transform(latest)
        return float(self.ridge.predict(latest_s)[0])  # next-quarter YoY CPI (approx)

    def path_10y(self, long_run_cpi: float, half_life_years: float = 3.0) -> np.ndarray:
        yoy1 = self.one_step_forecast()
        Tq = 40
        path = np.zeros(Tq, dtype=float)
        path[0] = yoy1
        step = 1 - 0.5**(1.0 / (max(0.5, half_life_years) * 4.0))
        for t in range(1, Tq):
            path[t] = path[t-1] + step * (long_run_cpi - path[t-1])
        return path

    def run(self, long_run_cpi: float) -> CPIForecastResult:
        self.load_and_prepare()
        self.fit()
        yoy1 = self.one_step_forecast()
        path = self.path_10y(long_run_cpi=long_run_cpi, half_life_years=3.0)
        return CPIForecastResult(
            df_quarterly=self.df_q.copy(),
            one_step_yoy=yoy1,
            path_yoy_10y=path,
            model_alpha=float(self.ridge.alpha_),
            coef=self.ridge.coef_.copy(),
        )
# ------------- r* Forecaster -----------------------------

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from .fred_utilities import fetch_fred, resample_quarterly_mean

@dataclass
class RStarInputs:
    start: str = "2010-01-01"
    end: Optional[str] = None
    series: Dict[str, str] = None

    def __post_init__(self):
        # Core drivers; all on FRED
        if self.series is None:
            self.series = {
                "FEDFUNDS": "FEDFUNDS",  # Effective Fed Funds Rate (monthly)
                "T5YIE":    "T5YIE",     # 5y breakeven inflation (monthly, %)
                "GDPC1":    "GDPC1",     # Real GDP (quarterly)
                "GDPPOT":   "GDPPOT",    # Potential GDP (quarterly)
                "UNRATE":   "UNRATE",    # Unemployment rate (monthly)
                "NROU":     "NROU",      # NAIRU (natural unemployment, quarterly)
                "DFII5":    "DFII5",     # 5y TIPS yield (monthly)
            }

@dataclass
class RStarForecastResult:
    df_quarterly: pd.DataFrame
    one_step_rstar: float         # next-quarter r* (real)
    path_rstar_10y: np.ndarray    # quarterly r* path for 10y (len=40)
    model_alpha: float
    coef: np.ndarray

class RStarForecaster:
    """
    r* proxy: FEDFUNDS - T5YIE (policy minus expected inflation).
    Drivers: output gap (GDPC1/GDPPOT - 1), unemployment gap (UNRATE - NROU),
             breakevens (T5YIE), 5y TIPS (DFII5).
    Estimator: RidgeCV for stability.
    Output: 1-step forecast of r* and a mean-reverting 10y path toward long-run r*.
    """
    def __init__(self, cfg: RStarInputs):
        self.cfg = cfg
        self.df_q = None
        self.scaler = None
        self.ridge = None

    def load_and_prepare(self) -> pd.DataFrame:
        s = self.cfg.series
        # Monthly series
        ff   = fetch_fred(s["FEDFUNDS"], start=self.cfg.start, end=self.cfg.end) / 100.0
        be5  = fetch_fred(s["T5YIE"],    start=self.cfg.start, end=self.cfg.end) / 100.0
        tips = fetch_fred(s["DFII5"],    start=self.cfg.start, end=self.cfg.end) / 100.0
        un   = fetch_fred(s["UNRATE"],   start=self.cfg.start, end=self.cfg.end) / 100.0

        # Quarterly series
        gdp  = fetch_fred(s["GDPC1"],  start=self.cfg.start, end=self.cfg.end)
        gpot = fetch_fred(s["GDPPOT"], start=self.cfg.start, end=self.cfg.end)
        nrou = fetch_fred(s["NROU"],   start=self.cfg.start, end=self.cfg.end) / 100.0

        # Assemble monthly, then quarterly means
        dfm = (
            pd.DataFrame({
                "FEDFUNDS": ff,
                "T5YIE":    be5,
                "DFII5":    tips,
                "UNRATE":   un,
            })
        )
        qm = resample_quarterly_mean(dfm)  # quarterly mean of monthly
        q  = pd.DataFrame(index=qm.index.copy())
        q[["FEDFUNDS","T5YIE","DFII5","UNRATE"]] = qm[["FEDFUNDS","T5YIE","DFII5","UNRATE"]]

        # Align quarterly-only
        q["GDPC1"] = gdp.resample("QE-DEC").last()
        q["GDPPOT"] = gpot.resample("QE-DEC").last()
        q["NROU"]   = nrou.resample("QE-DEC").ffill()

        # Features/targets
        q["Output_Gap"]    = (q["GDPC1"] / q["GDPPOT"]) - 1.0
        q["Unemp_Gap"]     = q["UNRATE"] - q["NROU"]
        q["BE5Y_Level"]    = q["T5YIE"]
        q["TIPS5_Level"]   = q["DFII5"]
        q["RSTAR_proxy"]   = q["FEDFUNDS"] - q["T5YIE"]  # real policy ≈ r* + cyclical

        q = q.dropna().copy()
        self.df_q = q
        return q

    def _feature_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        cols = ["Output_Gap", "Unemp_Gap", "BE5Y_Level", "TIPS5_Level"]
        X = self.df_q[cols].copy()
        y = self.df_q["RSTAR_proxy"].copy()
        return X, y

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        X, y = self._feature_target()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.ridge = RidgeCV(alphas=(0.1, 0.3, 1.0, 3.0, 10.0), cv=5).fit(Xs, y)
        return self.ridge

    def one_step_forecast(self) -> float:
        X, _ = self._feature_target()
        latest = X.iloc[[-1]]
        latest_s = self.scaler.transform(latest)
        return float(self.ridge.predict(latest_s)[0])

    def path_10y(self, long_run_rstar: float, half_life_years: float = 5.0) -> np.ndarray:
        r1 = self.one_step_forecast()
        Tq = 40
        path = np.zeros(Tq, dtype=float)
        path[0] = r1
        # Slow-moving anchor; r* reverts slowly
        step = 1 - 0.5**(1.0 / (max(0.5, half_life_years) * 4.0))
        for t in range(1, Tq):
            path[t] = path[t-1] + step * (long_run_rstar - path[t-1])
        return path

    def run(self, long_run_rstar: float) -> RStarForecastResult:
        self.load_and_prepare()
        self.fit()
        r1   = self.one_step_forecast()
        path = self.path_10y(long_run_rstar=long_run_rstar, half_life_years=5.0)
        return RStarForecastResult(
            df_quarterly=self.df_q.copy(),
            one_step_rstar=r1,
            path_rstar_10y=path,
            model_alpha=float(self.ridge.alpha_),
            coef=self.ridge.coef_.copy(),
        )

# ------------- Term Premium (TP) Forecaster ----------------------------

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from .fred_utilities import fetch_fred, resample_quarterly_mean

@dataclass
class TPInputs:
    start: str = "2010-01-01"
    end: Optional[str] = None
    series: Dict[str, str] = None

    def __post_init__(self):
        if self.series is None:
            self.series = {
                # Target (proxy): Instantaneous forward term premium 10y hence
                "TP_TARGET": "THREEFFTP10",
                # Drivers
                "T10Y2Y":  "T10Y2Y",   # 10y - 2y slope
                "T10Y3M":  "T10Y3M",   # 10y - 3m slope
                "DGS10":   "DGS10",    # 10y nominal yield
                "T5YIE":   "T5YIE",    # 5y breakeven inflation
            }

@dataclass
class TPForecastResult:
    df_quarterly: pd.DataFrame
    one_step_tp: float         # next-quarter TP level (decimal)
    path_tp_10y: np.ndarray    # quarterly TP for 10y (len=40)
    model_alpha: float
    coef: np.ndarray

class TPForecaster:
    """
    Target: THREEFFTP10 (instantaneous fwd term premium 10y hence) as a proxy for TP dynamics.
    Drivers: curve slopes (T10Y2Y, T10Y3M), level (DGS10), breakevens (T5YIE).
    Estimator: RidgeCV for robustness.
    Output: 1-step forecast and a 10y mean-reverting path toward a long-run TP.
    """
    def __init__(self, cfg: TPInputs):
        self.cfg = cfg
        self.df_q = None
        self.scaler = None
        self.ridge = None

    def load_and_prepare(self) -> pd.DataFrame:
        s = self.cfg.series
        # FRED series are % levels -> convert to decimals
        # THREEFFTP10 is published in percent (can be negative)
        try:
            tp_target = fetch_fred(s["TP_TARGET"], start=self.cfg.start, end=self.cfg.end) / 100.0
        except Exception:
            tp_target = None  # we will proxy below if needed

        y10   = fetch_fred(s["DGS10"],   start=self.cfg.start, end=self.cfg.end) / 100.0
        be5   = fetch_fred(s["T5YIE"],   start=self.cfg.start, end=self.cfg.end) / 100.0
        s10_2 = fetch_fred(s["T10Y2Y"],  start=self.cfg.start, end=self.cfg.end) / 100.0
        s10_3 = fetch_fred(s["T10Y3M"],  start=self.cfg.start, end=self.cfg.end) / 100.0

        dfm = pd.DataFrame({
            "TP_target": tp_target,   # may be None
            "DGS10": y10,
            "T5YIE": be5,
            "T10Y2Y": s10_2,
            "T10Y3M": s10_3,
        })

        q = resample_quarterly_mean(dfm)  # uses Q-DEC in your helper
        # If target missing, create a simple proxy (centered slope as rough TP proxy)
        if q["TP_target"].isna().all():
            # proxy: a weighted slope minus breakeven adjustments, scaled
            proxy = 0.6 * q["T10Y2Y"] + 0.4 * q["T10Y3M"] - 0.10 * (q["T5YIE"] - q["T5YIE"].rolling(20, min_periods=5).mean())
            q["TP_Target"] = proxy.fillna(method="ffill").fillna(0.0)
        else:
            q["TP_Target"] = q["TP_target"]

        # Features
        q["Slope_10_2"]   = q["T10Y2Y"]
        q["Slope_10_3m"]  = q["T10Y3M"]
        q["Y10_Level"]    = q["DGS10"]
        q["BE5_Level"]    = q["T5YIE"]
        q["Slope_Change"] = q["T10Y2Y"].diff()

        q = q.dropna().copy()
        self.df_q = q
        return q

    def _feature_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        cols = ["Slope_10_2", "Slope_10_3m", "Y10_Level", "BE5_Level", "Slope_Change"]
        X = self.df_q[cols].copy()
        y = self.df_q["TP_Target"].copy()
        return X, y

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        X, y = self._feature_target()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.ridge = RidgeCV(alphas=(0.1, 0.3, 1.0, 3.0, 10.0), cv=5).fit(Xs, y)
        return self.ridge

    def one_step_forecast(self) -> float:
        X, _ = self._feature_target()
        latest = X.iloc[[-1]]
        latest_s = self.scaler.transform(latest)
        return float(self.ridge.predict(latest_s)[0])

    def path_10y(self, long_run_tp: float, half_life_years: float = 4.0) -> np.ndarray:
        tp1 = self.one_step_forecast()
        Tq = 40
        path = np.zeros(Tq, dtype=float)
        path[0] = tp1
        step = 1 - 0.5**(1.0 / (max(0.5, half_life_years) * 4.0))
        for t in range(1, Tq):
            path[t] = path[t-1] + step * (long_run_tp - path[t-1])
        return path

    def run(self, long_run_tp: float) -> TPForecastResult:
        self.load_and_prepare()
        self.fit()
        tp1  = self.one_step_forecast()
        path = self.path_10y(long_run_tp=long_run_tp, half_life_years=4.0)
        return TPForecastResult(
            df_quarterly=self.df_q.copy(),
            one_step_tp=tp1,
            path_tp_10y=path,
            model_alpha=float(self.ridge.alpha_),
            coef=self.ridge.coef_.copy(),
        )

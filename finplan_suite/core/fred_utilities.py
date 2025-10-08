# finplan_suite/core/fred_utilities.py
from __future__ import annotations
import os
import pandas as pd
from pandas_datareader import data as pdr

CACHE_DIR = "data/fred_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(code: str) -> str:
    return os.path.join(CACHE_DIR, f"{code}.csv")

def _p(symbol: str, ext: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}.{ext}")

def _save_cache(df: pd.Series | pd.DataFrame, code: str):
    p = _cache_path(code)
    try:
        # write index for time series
        df.to_csv(p, index=True)
    except Exception as e:
        print(f"[FRED cache] failed to save {code}: {e}")

def _load_cache(code: str) -> pd.Series | pd.DataFrame | None:
    p = _cache_path(code)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        # if you cached a Series, re-cast it
        if df.shape[1] == 1:
            return df.iloc[:, 0]
        return df
    except Exception as e:
        print(f"[FRED cache] failed to load {code}: {e}")
        return None

def fetch_fred(symbol: str, start: str = "2000-01-01", end: str | None = None, force: bool = False) -> pd.Series:
    """
    Fetch a FRED series with on-disk caching.
    - If `force` is False, returns cache immediately if present.
    - If FRED call fails, returns cache if present; otherwise raises.
    Always returns a pandas Series named `symbol`.
    """
    if not force:
        cached = _load_cache(symbol)
        if cached is not None:
            return cached

    try:
        df = pdr.DataReader(symbol, "fred", start=start, end=end)  # DataFrame
        s = df.iloc[:, 0]
        s.name = symbol
        _save_cache(symbol, s)
        return s
    except Exception as e:
        # On any failure, try cache; if none, raise a clean error
        cached = _load_cache(symbol)
        if cached is not None:
            return cached
        raise RuntimeError(f"FRED fetch failed for {symbol} and no cache available: {e}")

def resample_quarterly_mean(df: pd.DataFrame, freq: str = "QE-DEC") -> pd.DataFrame:
    q = df.resample(freq).mean()
    return q.ffill()

def pct_change_series(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods)

def annualize_qoq(qoq: pd.Series) -> pd.Series:
    return (1.0 + qoq) ** 4 - 1.0

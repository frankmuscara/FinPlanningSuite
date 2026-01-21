"""Asset class taxonomy for HAMMER.

Defines asset classes and provides mapping of tickers to their respective
asset classes. Used for asset-class-aware rebalancing during VIX blocking.

When VIX term structure is inverted (backwardation):
- Intra-equity rebalancing is BLOCKED (don't sell one stock to buy another)
- Inter-asset-class rebalancing is ALLOWED (can shift from equities to bonds)
- Intra-bond/alternatives rebalancing is ALLOWED
"""

from enum import Enum
from typing import Dict, Optional, Set


class AssetClass(Enum):
    """Asset class categories."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    ALTERNATIVES = "alternatives"
    CASH = "cash"


DEFAULT_ASSET_CLASS_MAP: Dict[str, AssetClass] = {
    # US Large Cap Equity
    "VOO": AssetClass.EQUITY,
    "VTI": AssetClass.EQUITY,
    "SPY": AssetClass.EQUITY,
    "IVV": AssetClass.EQUITY,
    "SCHX": AssetClass.EQUITY,
    "QQQ": AssetClass.EQUITY,
    "COWZ": AssetClass.EQUITY,
    "VUG": AssetClass.EQUITY,
    "VTV": AssetClass.EQUITY,
    "MGK": AssetClass.EQUITY,
    "MTUM": AssetClass.EQUITY,
    "QUAL": AssetClass.EQUITY,

    # US Small/Mid Cap Equity
    "IJR": AssetClass.EQUITY,
    "VB": AssetClass.EQUITY,
    "VXF": AssetClass.EQUITY,
    "SCHA": AssetClass.EQUITY,
    "IJH": AssetClass.EQUITY,
    "VO": AssetClass.EQUITY,

    # International Developed Equity
    "VXUS": AssetClass.EQUITY,
    "VEA": AssetClass.EQUITY,
    "EFA": AssetClass.EQUITY,
    "IEFA": AssetClass.EQUITY,
    "SCHF": AssetClass.EQUITY,
    "VYMI": AssetClass.EQUITY,
    "VGK": AssetClass.EQUITY,
    "VPL": AssetClass.EQUITY,

    # Emerging Markets Equity
    "VWO": AssetClass.EQUITY,
    "EEM": AssetClass.EQUITY,
    "IEMG": AssetClass.EQUITY,
    "SCHE": AssetClass.EQUITY,

    # Sector Equity
    "XLF": AssetClass.EQUITY,
    "XLK": AssetClass.EQUITY,
    "XLE": AssetClass.EQUITY,
    "XLV": AssetClass.EQUITY,
    "XLI": AssetClass.EQUITY,
    "XLY": AssetClass.EQUITY,
    "XLP": AssetClass.EQUITY,
    "XLU": AssetClass.EQUITY,
    "XLRE": AssetClass.EQUITY,
    "XLB": AssetClass.EQUITY,
    "XLC": AssetClass.EQUITY,

    # US Investment Grade Bonds
    "BND": AssetClass.FIXED_INCOME,
    "BOND": AssetClass.FIXED_INCOME,
    "AGG": AssetClass.FIXED_INCOME,
    "SCHZ": AssetClass.FIXED_INCOME,
    "BSV": AssetClass.FIXED_INCOME,
    "BIV": AssetClass.FIXED_INCOME,
    "BLV": AssetClass.FIXED_INCOME,

    # US Treasury Bonds
    "TLT": AssetClass.FIXED_INCOME,
    "IEF": AssetClass.FIXED_INCOME,
    "SHY": AssetClass.FIXED_INCOME,
    "GOVT": AssetClass.FIXED_INCOME,
    "VGSH": AssetClass.FIXED_INCOME,
    "VGIT": AssetClass.FIXED_INCOME,
    "VGLT": AssetClass.FIXED_INCOME,
    "SCHO": AssetClass.FIXED_INCOME,
    "SCHR": AssetClass.FIXED_INCOME,

    # Short-Term / Ultra-Short Bonds
    "NEAR": AssetClass.FIXED_INCOME,
    "SHV": AssetClass.FIXED_INCOME,
    "MINT": AssetClass.FIXED_INCOME,
    "JPST": AssetClass.FIXED_INCOME,
    "ICSH": AssetClass.FIXED_INCOME,

    # TIPS / Inflation Protected
    "TIP": AssetClass.FIXED_INCOME,
    "SCHP": AssetClass.FIXED_INCOME,
    "VTIP": AssetClass.FIXED_INCOME,

    # Corporate Bonds
    "LQD": AssetClass.FIXED_INCOME,
    "VCIT": AssetClass.FIXED_INCOME,
    "VCSH": AssetClass.FIXED_INCOME,
    "VCLT": AssetClass.FIXED_INCOME,

    # High Yield / Credit
    "HYG": AssetClass.FIXED_INCOME,
    "JNK": AssetClass.FIXED_INCOME,
    "SHYG": AssetClass.FIXED_INCOME,

    # International / EM Bonds
    "VEMBX": AssetClass.FIXED_INCOME,
    "BNDX": AssetClass.FIXED_INCOME,
    "EMB": AssetClass.FIXED_INCOME,
    "IAGG": AssetClass.FIXED_INCOME,

    # Muni Bonds
    "MUB": AssetClass.FIXED_INCOME,
    "VTEB": AssetClass.FIXED_INCOME,
    "TFI": AssetClass.FIXED_INCOME,

    # Multi-Asset / Balanced
    "QLEIX": AssetClass.FIXED_INCOME,

    # Commodities
    "PDBC": AssetClass.ALTERNATIVES,
    "DBC": AssetClass.ALTERNATIVES,
    "GSG": AssetClass.ALTERNATIVES,
    "COMT": AssetClass.ALTERNATIVES,

    # Gold / Precious Metals
    "GLD": AssetClass.ALTERNATIVES,
    "IAU": AssetClass.ALTERNATIVES,
    "GLDM": AssetClass.ALTERNATIVES,
    "SLV": AssetClass.ALTERNATIVES,

    # Real Estate
    "VNQ": AssetClass.ALTERNATIVES,
    "SCHH": AssetClass.ALTERNATIVES,
    "IYR": AssetClass.ALTERNATIVES,
    "VNQI": AssetClass.ALTERNATIVES,

    # Cash / Money Market
    "SGOV": AssetClass.CASH,
    "BIL": AssetClass.CASH,
    "USFR": AssetClass.CASH,
}


def get_asset_class(
    ticker: str,
    custom_map: Optional[Dict[str, AssetClass]] = None,
) -> AssetClass:
    ticker_upper = ticker.upper()
    if custom_map and ticker_upper in custom_map:
        return custom_map[ticker_upper]
    if ticker_upper in DEFAULT_ASSET_CLASS_MAP:
        return DEFAULT_ASSET_CLASS_MAP[ticker_upper]
    return AssetClass.EQUITY


def classify_tickers(
    tickers: list,
    custom_map: Optional[Dict[str, AssetClass]] = None,
) -> Dict[AssetClass, Set[str]]:
    result: Dict[AssetClass, Set[str]] = {ac: set() for ac in AssetClass}
    for ticker in tickers:
        asset_class = get_asset_class(ticker, custom_map)
        result[asset_class].add(ticker.upper())
    return result


def get_equity_tickers(
    tickers: list,
    custom_map: Optional[Dict[str, AssetClass]] = None,
) -> Set[str]:
    return {
        t.upper() for t in tickers
        if get_asset_class(t, custom_map) == AssetClass.EQUITY
    }


def get_non_equity_tickers(
    tickers: list,
    custom_map: Optional[Dict[str, AssetClass]] = None,
) -> Set[str]:
    return {
        t.upper() for t in tickers
        if get_asset_class(t, custom_map) != AssetClass.EQUITY
    }

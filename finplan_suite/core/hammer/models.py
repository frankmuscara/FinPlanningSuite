"""
Portfolio Models/Templates

Predefined and custom portfolio models for client portfolio management.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class PortfolioModel:
    """A portfolio model/template with target allocations."""

    name: str
    description: str
    allocations: Dict[str, float]  # ticker -> weight (0-100)
    benchmark: str = "SPY"
    risk_level: str = "moderate"  # conservative, moderate, aggressive
    is_builtin: bool = False

    def __post_init__(self):
        """Normalize allocations to sum to 100."""
        total = sum(self.allocations.values())
        if total > 0 and abs(total - 100) > 0.01:
            self.allocations = {
                k: v / total * 100 for k, v in self.allocations.items()
            }

    @property
    def tickers(self) -> List[str]:
        """Get list of tickers."""
        return list(self.allocations.keys())

    @property
    def weights_normalized(self) -> Dict[str, float]:
        """Get weights as decimals (0-1)."""
        return {k: v / 100 for k, v in self.allocations.items()}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "allocations": self.allocations,
            "benchmark": self.benchmark,
            "risk_level": self.risk_level,
            "is_builtin": self.is_builtin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioModel":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            allocations=data["allocations"],
            benchmark=data.get("benchmark", "SPY"),
            risk_level=data.get("risk_level", "moderate"),
            is_builtin=data.get("is_builtin", False),
        )


# Built-in portfolio models
BUILTIN_MODELS: Dict[str, PortfolioModel] = {
    "conservative": PortfolioModel(
        name="Conservative",
        description="Low risk, income-focused. Heavy bond allocation for stability.",
        allocations={
            "BND": 50,   # Total Bond Market
            "VTI": 25,   # Total Stock Market
            "VTIP": 15,  # TIPS (inflation protected)
            "VXUS": 10,  # International Stocks
        },
        benchmark="SPY",
        risk_level="conservative",
        is_builtin=True,
    ),
    "balanced": PortfolioModel(
        name="Balanced",
        description="Classic 60/40 portfolio. Balanced growth and stability.",
        allocations={
            "VTI": 40,   # Total Stock Market
            "VXUS": 20,  # International Stocks
            "BND": 30,   # Total Bond Market
            "BNDX": 10,  # International Bonds
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "growth": PortfolioModel(
        name="Growth",
        description="Higher equity allocation for long-term growth.",
        allocations={
            "VTI": 50,   # Total Stock Market
            "VXUS": 25,  # International Stocks
            "VGT": 10,   # Tech Sector
            "BND": 15,   # Total Bond Market
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "aggressive": PortfolioModel(
        name="Aggressive Growth",
        description="Maximum growth focus. Higher volatility, higher potential returns.",
        allocations={
            "VTI": 45,   # Total Stock Market
            "VXUS": 25,  # International Stocks
            "VGT": 15,   # Tech Sector
            "VWO": 10,   # Emerging Markets
            "BND": 5,    # Minimal Bonds
        },
        benchmark="QQQ",
        risk_level="aggressive",
        is_builtin=True,
    ),
    "income": PortfolioModel(
        name="Income",
        description="Dividend and income focused for cash flow generation.",
        allocations={
            "VYM": 30,   # High Dividend Yield
            "SCHD": 20,  # Dividend Growth
            "BND": 25,   # Total Bond Market
            "VCIT": 15,  # Corporate Bonds
            "VNQ": 10,   # Real Estate (REITs)
        },
        benchmark="SPY",
        risk_level="conservative",
        is_builtin=True,
    ),
    "all_weather": PortfolioModel(
        name="All Weather",
        description="Ray Dalio inspired. Designed to perform in any economic environment.",
        allocations={
            "VTI": 30,   # Stocks
            "TLT": 40,   # Long-term Treasuries
            "IEF": 15,   # Intermediate Treasuries
            "GLD": 7.5,  # Gold
            "DBC": 7.5,  # Commodities
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "three_fund": PortfolioModel(
        name="Three Fund",
        description="Bogleheads classic. Simple, diversified, low-cost.",
        allocations={
            "VTI": 50,   # Total Stock Market
            "VXUS": 30,  # International Stocks
            "BND": 20,   # Total Bond Market
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "target_2030": PortfolioModel(
        name="Target 2030",
        description="Target date style for retirement around 2030. Moderately conservative.",
        allocations={
            "VTI": 35,
            "VXUS": 20,
            "BND": 35,
            "BNDX": 10,
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "target_2040": PortfolioModel(
        name="Target 2040",
        description="Target date style for retirement around 2040. Growth focused.",
        allocations={
            "VTI": 45,
            "VXUS": 25,
            "BND": 22,
            "BNDX": 8,
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "target_2050": PortfolioModel(
        name="Target 2050",
        description="Target date style for retirement around 2050. Aggressive growth.",
        allocations={
            "VTI": 52,
            "VXUS": 28,
            "BND": 15,
            "BNDX": 5,
        },
        benchmark="SPY",
        risk_level="aggressive",
        is_builtin=True,
    ),
    # GC Proprietary Models
    # Structure: Equities split 75/25 domestic/int'l
    # Domestic: 50/50 COWZ/QQQ, Int'l: VYMI, Bonds: BOND, Alts: QLEIX
    "gc_conservative": PortfolioModel(
        name="GC Proprietary Conservative",
        description="35/55/10 Equity/Bond/Alt. Conservative allocation with alternatives sleeve.",
        allocations={
            "COWZ": 13.125,  # Domestic (35% * 75% * 50%)
            "QQQ": 13.125,   # Domestic (35% * 75% * 50%)
            "VYMI": 8.75,    # International (35% * 25%)
            "BOND": 55,      # Fixed Income
            "QLEIX": 10,     # Alternatives
        },
        benchmark="SPY",
        risk_level="conservative",
        is_builtin=True,
    ),
    "gc_moderate": PortfolioModel(
        name="GC Proprietary Moderate",
        description="55/35/10 Equity/Bond/Alt. Balanced allocation with alternatives sleeve.",
        allocations={
            "COWZ": 20.625,  # Domestic (55% * 75% * 50%)
            "QQQ": 20.625,   # Domestic (55% * 75% * 50%)
            "VYMI": 13.75,   # International (55% * 25%)
            "BOND": 35,      # Fixed Income
            "QLEIX": 10,     # Alternatives
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "gc_growth": PortfolioModel(
        name="GC Proprietary Growth",
        description="75/15/10 Equity/Bond/Alt. Growth-focused with alternatives sleeve.",
        allocations={
            "COWZ": 28.125,  # Domestic (75% * 75% * 50%)
            "QQQ": 28.125,   # Domestic (75% * 75% * 50%)
            "VYMI": 18.75,   # International (75% * 25%)
            "BOND": 15,      # Fixed Income
            "QLEIX": 10,     # Alternatives
        },
        benchmark="SPY",
        risk_level="moderate",
        is_builtin=True,
    ),
    "gc_aggressive": PortfolioModel(
        name="GC Proprietary Aggressive Growth",
        description="85/5/10 Equity/Bond/Alt. Maximum growth with alternatives sleeve.",
        allocations={
            "COWZ": 31.875,  # Domestic (85% * 75% * 50%)
            "QQQ": 31.875,   # Domestic (85% * 75% * 50%)
            "VYMI": 21.25,   # International (85% * 25%)
            "BOND": 5,       # Fixed Income
            "QLEIX": 10,     # Alternatives
        },
        benchmark="QQQ",
        risk_level="aggressive",
        is_builtin=True,
    ),
}


class ModelManager:
    """Manages portfolio models (built-in and custom)."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize with optional custom storage path."""
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to user's home directory
            self.storage_path = Path.home() / ".hammer" / "models.json"

        self._custom_models: Dict[str, PortfolioModel] = {}
        self._load_custom_models()

    def _load_custom_models(self):
        """Load custom models from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for key, model_data in data.items():
                        self._custom_models[key] = PortfolioModel.from_dict(model_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load custom models: {e}")

    def _save_custom_models(self):
        """Save custom models to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: model.to_dict() for key, model in self._custom_models.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_all_models(self) -> Dict[str, PortfolioModel]:
        """Get all models (built-in + custom)."""
        all_models = dict(BUILTIN_MODELS)
        all_models.update(self._custom_models)
        return all_models

    def get_builtin_models(self) -> Dict[str, PortfolioModel]:
        """Get only built-in models."""
        return dict(BUILTIN_MODELS)

    def get_custom_models(self) -> Dict[str, PortfolioModel]:
        """Get only custom models."""
        return dict(self._custom_models)

    def get_model(self, key: str) -> Optional[PortfolioModel]:
        """Get a model by key."""
        if key in BUILTIN_MODELS:
            return BUILTIN_MODELS[key]
        return self._custom_models.get(key)

    def save_model(self, model: PortfolioModel) -> str:
        """Save a custom model. Returns the key."""
        # Generate key from name
        key = model.name.lower().replace(" ", "_")

        # Don't allow overwriting built-in models
        if key in BUILTIN_MODELS:
            key = f"custom_{key}"

        model.is_builtin = False
        self._custom_models[key] = model
        self._save_custom_models()
        return key

    def delete_model(self, key: str) -> bool:
        """Delete a custom model. Returns True if deleted."""
        if key in BUILTIN_MODELS:
            return False  # Can't delete built-in
        if key in self._custom_models:
            del self._custom_models[key]
            self._save_custom_models()
            return True
        return False

    def model_exists(self, name: str) -> bool:
        """Check if a model with this name exists."""
        key = name.lower().replace(" ", "_")
        return key in BUILTIN_MODELS or key in self._custom_models


def get_model_choices() -> List[tuple]:
    """Get model choices for dropdown (key, display_name)."""
    manager = ModelManager()
    choices = []

    # Built-in models first
    for key, model in manager.get_builtin_models().items():
        choices.append((key, f"{model.name} (Built-in)"))

    # Then custom models
    for key, model in manager.get_custom_models().items():
        choices.append((key, f"{model.name} (Custom)"))

    return choices

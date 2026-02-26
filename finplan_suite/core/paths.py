"""Project path helpers for stable data file access."""

from pathlib import Path


def project_root() -> Path:
    """Return repository root path based on this package location."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the root data directory."""
    return project_root() / "data"


def data_dir(*parts: str) -> Path:
    """Return a data directory path and ensure it exists."""
    path = data_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_file(*parts: str) -> Path:
    """Return a path under data/ without creating parent directories."""
    return data_root().joinpath(*parts)

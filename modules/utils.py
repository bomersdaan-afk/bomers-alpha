"""
utils.py — Shared utility functions for QuantEdge
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple


# ─── Number Formatting ────────────────────────────────────────────────────────

def format_large_number(value: Optional[float], decimals: int = 2) -> str:
    """Format large numbers with T/B/M/K suffixes."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"

    sign = "-" if v < 0 else ""
    abs_v = abs(v)

    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:.{decimals}f}T"
    elif abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:.{decimals}f}B"
    elif abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:.{decimals}f}M"
    elif abs_v >= 1e3:
        return f"{sign}${abs_v / 1e3:.{decimals}f}K"
    else:
        return f"{sign}${abs_v:.{decimals}f}"


def format_percentage(value: Optional[float], decimals: int = 1, multiply: bool = False) -> str:
    """Format value as percentage string."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
        if multiply:
            v *= 100
        return f"{v:+.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_multiple(value: Optional[float], decimals: int = 1) -> str:
    """Format a valuation multiple (e.g., 15.3x)."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or v <= 0:
            return "N/A"
        return f"{v:.{decimals}f}x"
    except (TypeError, ValueError):
        return "N/A"


def format_price(value: Optional[float]) -> str:
    """Format a dollar price."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
        return f"${v:,.2f}"
    except (TypeError, ValueError):
        return "N/A"


# ─── Math Helpers ─────────────────────────────────────────────────────────────

def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
    default: Optional[float] = None,
) -> Optional[float]:
    """Division that returns `default` on zero / None / NaN."""
    try:
        if denominator is None or denominator == 0:
            return default
        result = float(numerator) / float(denominator)
        return result if not np.isnan(result) else default
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float between min and max."""
    return max(min_val, min(max_val, float(value)))


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """Convert a value to float, returning default on failure."""
    try:
        v = float(value)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


# ─── Signal / Score Helpers ───────────────────────────────────────────────────

def score_to_signal(score: float) -> Tuple[str, str]:
    """Convert a 0-100 composite score to (signal, hex_color)."""
    if score >= 65:
        return "BUY", "#00E676"
    elif score >= 40:
        return "HOLD", "#FFD600"
    else:
        return "SELL", "#FF1744"


def score_to_color(score: float) -> str:
    """Return a hex color for any score 0-100."""
    if score >= 65:
        return "#00E676"
    elif score >= 40:
        return "#FFD600"
    return "#FF1744"


def margin_of_safety_color(mos: float) -> str:
    """Green / yellow / red depending on margin of safety %."""
    if mos >= 20:
        return "#00E676"
    elif mos >= 0:
        return "#FFD600"
    return "#FF1744"

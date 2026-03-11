"""
screener.py — Multi-stock scanner and ranker
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from modules.data_fetcher import get_full_stock_data, extract_key_metrics
from modules.valuation import run_full_valuation
from modules.utils import safe_float

# ─── Default Universe ─────────────────────────────────────────────────────────

DEFAULT_TICKERS: List[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "GS", "V", "MA", "BRK-B",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK",
    # Consumer
    "HD", "MCD", "KO", "PG", "WMT", "COST", "NKE",
    # Industrials
    "CAT", "HON", "GE", "BA",
    # Energy
    "XOM", "CVX",
    # Communication
    "NFLX", "DIS",
]


# ─── Single-Stock Screen ──────────────────────────────────────────────────────

def screen_stock(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Run the full valuation pipeline on one ticker and return a flat record.
    Returns None on any failure so the screener can skip bad tickers.
    """
    try:
        data = get_full_stock_data(ticker)
        if "error" in data:
            return None

        val = run_full_valuation(data)
        m   = val["metrics"]
        sc  = val["scores"]
        h   = val["health"]
        dcf = val["dcf"]

        def pct(v):
            f = safe_float(v)
            return round(f * 100, 2) if f is not None else None

        def rnd(v, d=2):
            f = safe_float(v)
            return round(f, d) if f is not None else None

        return {
            "Ticker":        ticker,
            "Company":       (m.get("company_name") or ticker)[:28],
            "Sector":        m.get("sector", "N/A"),
            "Price":         rnd(m.get("current_price")),
            "Mkt Cap ($B)":  rnd((m.get("market_cap") or 0) / 1e9, 1),
            "P/E":           rnd(m.get("pe_ratio"), 1),
            "Fwd P/E":       rnd(m.get("forward_pe"), 1),
            "EV/EBITDA":     rnd(m.get("ev_ebitda"), 1),
            "P/S":           rnd(m.get("ps_ratio"), 1),
            "Rev Growth %":  pct(m.get("revenue_growth")),
            "EPS Growth %":  pct(m.get("earnings_growth")),
            "Gross Margin %":pct(m.get("gross_margin")),
            "Net Margin %":  pct(m.get("profit_margin")),
            "ROE %":         pct(m.get("roe")),
            "ROIC %":        pct(h.get("roic")),
            "Debt/EBITDA":   rnd(h.get("debt_to_ebitda"), 1),
            "Intrinsic Val": rnd(dcf.get("intrinsic_value")),
            "MoS %":         rnd(val.get("margin_of_safety"), 1),
            "Value Score":   round(sc.get("value", 0), 1),
            "Growth Score":  round(sc.get("growth", 0), 1),
            "Quality Score": round(sc.get("quality", 0), 1),
            "Macro Score":   round(sc.get("macro", 0), 1),
            "Final Score":   round(sc.get("final", 0), 1),
            "Signal":        val.get("signal", "N/A"),
        }
    except Exception:
        return None


# ─── Batch Screener ───────────────────────────────────────────────────────────

def run_screener(
    tickers: List[str],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_stocks: int = 35,
) -> pd.DataFrame:
    """
    Screen a list of tickers and return a DataFrame sorted by Final Score.

    Args:
        tickers:           List of ticker symbols.
        progress_callback: Optional (pct_float, message_str) callable.
        max_stocks:        Hard cap on universe size.
    """
    tickers = [t.upper().strip() for t in tickers][:max_stocks]
    results: List[Dict[str, Any]] = []

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i / len(tickers), f"Scanning {ticker}…")
        record = screen_stock(ticker)
        if record:
            results.append(record)

    if progress_callback:
        progress_callback(1.0, f"Done — {len(results)} stocks screened.")

    if not results:
        return pd.DataFrame()

    df = (
        pd.DataFrame(results)
        .sort_values("Final Score", ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1   # 1-based rank
    return df


# ─── Filter Helpers ───────────────────────────────────────────────────────────

def filter_screener(
    df: pd.DataFrame,
    min_score: float = 0,
    signals: Optional[List[str]] = None,
    sectors: Optional[List[str]] = None,
    min_mkt_cap: float = 0,
    max_pe: float = 999,
) -> pd.DataFrame:
    """Apply common screener filters to a results DataFrame."""
    if df.empty:
        return df

    mask = (df["Final Score"] >= min_score)

    if signals:
        mask &= df["Signal"].isin(signals)
    if sectors:
        mask &= df["Sector"].isin(sectors)
    if "Mkt Cap ($B)" in df.columns:
        mask &= df["Mkt Cap ($B)"].fillna(0) >= min_mkt_cap
    if "P/E" in df.columns:
        mask &= (df["P/E"].fillna(999) <= max_pe)

    return df[mask]

"""
data_fetcher.py — All data retrieval from Yahoo Finance via yfinance
"""

from __future__ import annotations
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import streamlit as st


# ─── Raw Data Fetching ────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_full_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Pull all available data for a ticker in one shot.
    Returns a dict with keys: ticker, info, price_history_2y,
    price_history_5y, financials, balance_sheet, cash_flow.
    """
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Validate: yfinance returns a stub dict for bad tickers
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None and info.get("previousClose") is None:
            # Try one more field
            if not info.get("longName") and not info.get("shortName"):
                return {"error": f"Ticker '{ticker}' not found.", "ticker": ticker}

        hist_2y = stock.history(period="2y")
        hist_5y = stock.history(period="5y")
        financials = stock.financials          # annual income statement
        balance_sheet = stock.balance_sheet    # annual balance sheet
        cash_flow = stock.cashflow             # annual cash flow

        return {
            "ticker": ticker,
            "info": info,
            "price_history_2y": hist_2y,
            "price_history_5y": hist_5y,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
        }
    except Exception as exc:
        return {"error": str(exc), "ticker": ticker}


# ─── Key-Metric Extraction ────────────────────────────────────────────────────

def extract_key_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the raw yfinance `info` dict into a clean, typed metrics dict.
    All numeric fields default to None when unavailable.
    """
    info: Dict[str, Any] = data.get("info", {})

    def _get(*keys, cast=None, default=None):
        """Try multiple key names and optionally cast the result."""
        for k in keys:
            v = info.get(k)
            if v is not None:
                try:
                    return cast(v) if cast else v
                except (TypeError, ValueError):
                    pass
        return default

    # ── Price & Market ────────────────────────────────────────────────────────
    current_price = _get("currentPrice", "regularMarketPrice", "previousClose", cast=float)
    market_cap    = _get("marketCap", cast=float)
    shares_out    = _get("sharesOutstanding", "impliedSharesOutstanding", cast=float)
    enterprise_v  = _get("enterpriseValue", cast=float)

    # ── Valuation Multiples ───────────────────────────────────────────────────
    pe_ratio      = _get("trailingPE", cast=float)
    forward_pe    = _get("forwardPE", cast=float)
    pb_ratio      = _get("priceToBook", cast=float)
    ps_ratio      = _get("priceToSalesTrailing12Months", cast=float)
    ev_ebitda     = _get("enterpriseToEbitda", cast=float)
    ev_revenue    = _get("enterpriseToRevenue", cast=float)
    peg_ratio     = _get("pegRatio", cast=float)

    # ── Income / Cash Flow ────────────────────────────────────────────────────
    revenue       = _get("totalRevenue", cast=float)
    ebitda        = _get("ebitda", cast=float)
    net_income    = _get("netIncomeToCommon", cast=float)
    free_cash_flow = _get("freeCashflow", cast=float)
    operating_cf  = _get("operatingCashflow", cast=float)

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    total_debt    = _get("totalDebt", cast=float)
    total_cash    = _get("totalCash", cast=float)

    # ── Per-Share ─────────────────────────────────────────────────────────────
    eps_ttm       = _get("trailingEps", cast=float)
    eps_forward   = _get("forwardEps", cast=float)
    book_value    = _get("bookValue", cast=float)

    # ── Growth ────────────────────────────────────────────────────────────────
    revenue_growth   = _get("revenueGrowth", cast=float)
    earnings_growth  = _get("earningsGrowth", cast=float)
    earnings_quarterly_growth = _get("earningsQuarterlyGrowth", cast=float)

    # ── Profitability ─────────────────────────────────────────────────────────
    gross_margin     = _get("grossMargins", cast=float)
    operating_margin = _get("operatingMargins", cast=float)
    profit_margin    = _get("profitMargins", cast=float)
    roe              = _get("returnOnEquity", cast=float)
    roa              = _get("returnOnAssets", cast=float)

    # ── Analyst / Market ──────────────────────────────────────────────────────
    target_price     = _get("targetMeanPrice", "targetHighPrice", cast=float)
    target_low       = _get("targetLowPrice", cast=float)
    target_high      = _get("targetHighPrice", cast=float)
    analyst_rec      = _get("recommendationMean", cast=float)   # 1=Strong Buy … 5=Strong Sell
    num_analysts     = _get("numberOfAnalystOpinions", cast=int)

    # ── Dividend & Risk ───────────────────────────────────────────────────────
    dividend_yield   = _get("dividendYield", cast=float)
    payout_ratio     = _get("payoutRatio", cast=float)
    beta             = _get("beta", cast=float)

    # ── 52-Week Range ─────────────────────────────────────────────────────────
    week_52_high     = _get("fiftyTwoWeekHigh", cast=float)
    week_52_low      = _get("fiftyTwoWeekLow", cast=float)
    day_high         = _get("dayHigh", "regularMarketDayHigh", cast=float)
    day_low          = _get("dayLow", "regularMarketDayLow", cast=float)

    return {
        # Identity
        "ticker":           data.get("ticker", ""),
        "company_name":     info.get("longName") or info.get("shortName") or data.get("ticker", ""),
        "sector":           info.get("sector", "N/A"),
        "industry":         info.get("industry", "N/A"),
        "country":          info.get("country", "N/A"),
        "employees":        info.get("fullTimeEmployees"),
        "description":      info.get("longBusinessSummary", ""),
        "website":          info.get("website", ""),
        # Price & Market
        "current_price":    current_price,
        "market_cap":       market_cap,
        "shares_outstanding": shares_out,
        "enterprise_value": enterprise_v,
        # Multiples
        "pe_ratio":         pe_ratio,
        "forward_pe":       forward_pe,
        "pb_ratio":         pb_ratio,
        "ps_ratio":         ps_ratio,
        "ev_ebitda":        ev_ebitda,
        "ev_revenue":       ev_revenue,
        "peg_ratio":        peg_ratio,
        # Financials
        "revenue":          revenue,
        "ebitda":           ebitda,
        "net_income":       net_income,
        "free_cash_flow":   free_cash_flow,
        "operating_cf":     operating_cf,
        "total_debt":       total_debt,
        "total_cash":       total_cash,
        # Per-share
        "eps_ttm":          eps_ttm,
        "eps_forward":      eps_forward,
        "book_value":       book_value,
        # Growth
        "revenue_growth":   revenue_growth,
        "earnings_growth":  earnings_growth,
        "earnings_quarterly_growth": earnings_quarterly_growth,
        # Profitability
        "gross_margin":     gross_margin,
        "operating_margin": operating_margin,
        "profit_margin":    profit_margin,
        "roe":              roe,
        "roa":              roa,
        # Analyst
        "target_price":     target_price,
        "target_low":       target_low,
        "target_high":      target_high,
        "analyst_rec":      analyst_rec,
        "num_analysts":     num_analysts,
        # Dividend / risk
        "dividend_yield":   dividend_yield,
        "payout_ratio":     payout_ratio,
        "beta":             beta,
        # Range
        "week_52_high":     week_52_high,
        "week_52_low":      week_52_low,
        "day_high":         day_high,
        "day_low":          day_low,
    }

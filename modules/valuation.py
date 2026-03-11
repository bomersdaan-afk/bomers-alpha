"""
valuation.py — DCF, comparable multiples, scoring, and signal generation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from modules.utils import clamp, safe_divide, safe_float
from modules.data_fetcher import extract_key_metrics


# ─── WACC & Cost of Capital ───────────────────────────────────────────────────

def estimate_cost_of_equity(
    beta: float = 1.0,
    risk_free_rate: float = 0.045,   # ~10-yr US Treasury (2025)
    equity_risk_premium: float = 0.055,
) -> float:
    """CAPM: E(r) = Rf + β × ERP"""
    beta = clamp(float(beta or 1.0), 0.3, 3.0)
    return risk_free_rate + beta * equity_risk_premium


def calculate_wacc(
    equity: float,
    debt: float,
    cost_of_equity: float,
    cost_of_debt: float = 0.045,
    tax_rate: float = 0.21,
) -> float:
    """Weighted Average Cost of Capital."""
    total = equity + debt
    if total <= 0:
        return 0.10
    we = equity / total
    wd = debt / total
    wacc = we * cost_of_equity + wd * cost_of_debt * (1 - tax_rate)
    return clamp(wacc, 0.06, 0.20)


# ─── DCF Model ────────────────────────────────────────────────────────────────

def calculate_dcf(
    free_cash_flow: Optional[float],
    revenue: Optional[float],
    revenue_growth: Optional[float],
    shares_outstanding: Optional[float],
    beta: float = 1.0,
    total_debt: float = 0.0,
    total_cash: float = 0.0,
    market_cap: float = 0.0,
    forecast_years: int = 10,
    terminal_growth: float = 0.025,
    tax_rate: float = 0.21,
) -> Dict[str, Any]:
    """
    Two-stage DCF model.

    Stage 1 (years 1-5): high growth derived from revenue_growth.
    Stage 2 (years 6-10): linearly declining toward terminal growth.
    Returns intrinsic value per share and full breakdown.
    """
    if not free_cash_flow or not shares_outstanding or shares_outstanding <= 0:
        return {"error": "Insufficient FCF or share count data.", "intrinsic_value": None}

    fcf = float(free_cash_flow)
    shares = float(shares_outstanding)
    debt = float(total_debt or 0)
    cash = float(total_cash or 0)
    mcap = float(market_cap or shares * 50)  # rough fallback

    # Cost of capital
    coe = estimate_cost_of_equity(beta)
    wacc = calculate_wacc(
        equity=mcap,
        debt=debt,
        cost_of_equity=coe,
        cost_of_debt=0.045,
        tax_rate=tax_rate,
    )

    # Base FCF growth: anchor to revenue growth, clamp aggressively
    base_growth = clamp(float(revenue_growth or 0.05), -0.15, 0.35)

    # Build year-by-year forecast
    fcf_forecasts: List[Dict[str, float]] = []
    current_fcf = fcf

    for yr in range(1, forecast_years + 1):
        if yr <= 5:
            g = base_growth
        else:
            # Linear fade from base_growth → terminal_growth over years 6-10
            fade = (yr - 5) / 5
            g = base_growth * (1 - fade) + terminal_growth * fade

        current_fcf = current_fcf * (1 + g)
        df = (1 + wacc) ** yr
        pv = current_fcf / df
        fcf_forecasts.append(
            {"year": yr, "fcf": current_fcf, "growth_rate": g, "discount_factor": df, "pv_fcf": pv}
        )

    # Terminal value (Gordon Growth)
    terminal_fcf = fcf_forecasts[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** forecast_years

    # Sum of PV of free cash flows
    pv_fcfs_sum = sum(f["pv_fcf"] for f in fcf_forecasts)

    # Enterprise value → equity value → per-share
    enterprise_value = pv_fcfs_sum + pv_terminal
    net_debt = debt - cash
    equity_value = enterprise_value - net_debt
    intrinsic_value = equity_value / shares

    return {
        "intrinsic_value": intrinsic_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "pv_fcfs": pv_fcfs_sum,
        "pv_terminal": pv_terminal,
        "terminal_value": terminal_value,
        "wacc": wacc,
        "cost_of_equity": coe,
        "terminal_growth": terminal_growth,
        "base_growth": base_growth,
        "fcf_forecasts": fcf_forecasts,
        "forecast_years": forecast_years,
    }


def calculate_margin_of_safety(current_price: float, intrinsic_value: float) -> Optional[float]:
    """MoS = (IV - Price) / IV × 100. Positive = undervalued."""
    if not intrinsic_value or intrinsic_value <= 0 or not current_price:
        return None
    return (intrinsic_value - current_price) / intrinsic_value * 100


# ─── Comparable Multiples ─────────────────────────────────────────────────────

# S&P 500 long-run averages used as default benchmarks
_SP500_BENCHMARKS = {
    "pe":         20.0,
    "forward_pe": 18.0,
    "pb":          3.5,
    "ps":          2.5,
    "ev_ebitda":  14.0,
    "ev_revenue":  2.5,
    "peg":         1.5,
}


def _multiple_score(value: Optional[float], benchmark: float) -> int:
    """Score a multiple vs benchmark. Lower is better (cheaper). Returns 0-100."""
    if value is None or value <= 0:
        return 50  # neutral when no data
    ratio = value / benchmark
    thresholds = [(0.5, 90), (0.70, 80), (0.85, 68), (1.0, 55), (1.20, 40), (1.50, 25)]
    for threshold, score in thresholds:
        if ratio <= threshold:
            return score
    return 12


def calculate_comparable_multiples(
    metrics: Dict[str, Any],
    benchmarks: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score each valuation multiple against market benchmarks."""
    bm = benchmarks or _SP500_BENCHMARKS

    multiples = {
        "pe_ratio":   metrics.get("pe_ratio"),
        "forward_pe": metrics.get("forward_pe"),
        "pb_ratio":   metrics.get("pb_ratio"),
        "ps_ratio":   metrics.get("ps_ratio"),
        "ev_ebitda":  metrics.get("ev_ebitda"),
        "ev_revenue": metrics.get("ev_revenue"),
        "peg_ratio":  metrics.get("peg_ratio"),
    }

    scores = {
        "pe_score":         _multiple_score(multiples["pe_ratio"],   bm["pe"]),
        "forward_pe_score": _multiple_score(multiples["forward_pe"], bm["forward_pe"]),
        "pb_score":         _multiple_score(multiples["pb_ratio"],   bm["pb"]),
        "ps_score":         _multiple_score(multiples["ps_ratio"],   bm["ps"]),
        "ev_ebitda_score":  _multiple_score(multiples["ev_ebitda"],  bm["ev_ebitda"]),
        "ev_revenue_score": _multiple_score(multiples["ev_revenue"], bm["ev_revenue"]),
        "peg_score":        _multiple_score(multiples["peg_ratio"],  bm["peg"]),
    }

    valid_scores = [v for v in scores.values()]
    avg_score = float(np.mean(valid_scores)) if valid_scores else 50.0

    return {"multiples": multiples, "scores": scores, "value_score": avg_score, "benchmarks": bm}


# ─── Growth Analysis ──────────────────────────────────────────────────────────

def _extract_series(df: pd.DataFrame, *labels: str) -> List[float]:
    """Try multiple row labels, return list of floats (newest → oldest)."""
    if df is None or df.empty:
        return []
    for lbl in labels:
        if lbl in df.index:
            values = []
            for col in df.columns:
                try:
                    v = float(df.loc[lbl, col])
                    if not np.isnan(v):
                        values.append(v)
                except (TypeError, ValueError):
                    pass
            if values:
                return values
    return []


def _growth_rates(series: List[float]) -> List[float]:
    """Compute YoY growth from a newest-first series."""
    if len(series) < 2:
        return []
    # Reverse to oldest-first for growth calculation
    s = list(reversed(series))
    rates = []
    for i in range(1, len(s)):
        denom = abs(s[i - 1])
        if denom > 0:
            rates.append((s[i] - s[i - 1]) / denom)
    return rates


def calculate_growth_analysis(
    financials: Optional[pd.DataFrame],
    cash_flow: Optional[pd.DataFrame],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract historical growth from financial statements."""
    result: Dict[str, Any] = {}

    # Revenue
    rev_series = _extract_series(financials, "Total Revenue", "Revenue")
    rev_growths = _growth_rates(rev_series)
    if rev_series:
        result["revenue_history"] = list(reversed(rev_series))   # oldest → newest
        result["revenue_growth_rates"] = rev_growths
        result["avg_revenue_growth"] = float(np.mean(rev_growths)) if rev_growths else None

    # Net income
    ni_series = _extract_series(financials, "Net Income", "Net Income Common Stockholders")
    ni_growths = _growth_rates(ni_series)
    if ni_series:
        result["net_income_history"] = list(reversed(ni_series))
        result["ni_growth_rates"] = ni_growths
        result["avg_ni_growth"] = float(np.mean(ni_growths)) if ni_growths else None

    # Free Cash Flow (OCF + CapEx, where CapEx is usually negative)
    ocf_series = _extract_series(
        cash_flow,
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash From Operations",
    )
    capex_series = _extract_series(
        cash_flow,
        "Capital Expenditures",
        "Capital Expenditure",
        "Purchase Of Property Plant And Equipment",
    )
    if ocf_series:
        if capex_series and len(capex_series) == len(ocf_series):
            fcf_series = [o + c for o, c in zip(ocf_series, capex_series)]
        else:
            fcf_series = ocf_series
        fcf_growths = _growth_rates(fcf_series)
        result["fcf_history"] = list(reversed(fcf_series))
        result["fcf_growth_rates"] = fcf_growths
        result["avg_fcf_growth"] = float(np.mean(fcf_growths)) if fcf_growths else None

    # Fill-in from info when statements are sparse
    result["revenue_growth_yoy"]  = metrics.get("revenue_growth")
    result["earnings_growth_yoy"] = metrics.get("earnings_growth")

    return result


# ─── Financial Health ─────────────────────────────────────────────────────────

def calculate_financial_health(
    metrics: Dict[str, Any],
    balance_sheet: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute debt ratios, ROIC and margin snapshots."""
    debt  = safe_float(metrics.get("total_debt"), 0.0) or 0.0
    cash  = safe_float(metrics.get("total_cash"), 0.0) or 0.0
    ebitda = safe_float(metrics.get("ebitda"))
    ni    = safe_float(metrics.get("net_income"))
    mcap  = safe_float(metrics.get("market_cap"), 0.0) or 0.0

    net_debt = debt - cash
    debt_to_ebitda     = safe_divide(debt, ebitda)
    net_debt_to_ebitda = safe_divide(net_debt, ebitda)

    # ROIC: NOPAT / Invested Capital  (simplified)
    invested_capital = debt + mcap - cash
    roic = safe_divide(ni, invested_capital) if ni else None

    # Interest coverage (ebit / interest) — approximate
    ebit = None
    if ebitda and metrics.get("operating_margin") and metrics.get("revenue"):
        ebit = metrics["revenue"] * metrics["operating_margin"]

    return {
        "net_debt":            net_debt,
        "debt_to_ebitda":      debt_to_ebitda,
        "net_debt_to_ebitda":  net_debt_to_ebitda,
        "roic":                roic,
        "gross_margin":        metrics.get("gross_margin"),
        "operating_margin":    metrics.get("operating_margin"),
        "profit_margin":       metrics.get("profit_margin"),
        "roe":                 metrics.get("roe"),
        "roa":                 metrics.get("roa"),
    }


# ─── Scoring Dimensions ───────────────────────────────────────────────────────

def score_value(
    metrics: Dict[str, Any],
    dcf_result: Dict[str, Any],
    multiples_result: Dict[str, Any],
) -> float:
    """Value dimension score 0-100."""
    scores: List[float] = []

    # 1. DCF margin of safety
    price = safe_float(metrics.get("current_price"))
    iv    = safe_float(dcf_result.get("intrinsic_value"))
    if price and iv and iv > 0:
        mos = (iv - price) / iv * 100
        if mos >= 40:   scores.append(95)
        elif mos >= 25: scores.append(82)
        elif mos >= 15: scores.append(70)
        elif mos >= 5:  scores.append(58)
        elif mos >= -5: scores.append(45)
        elif mos >= -20:scores.append(30)
        else:           scores.append(12)

    # 2. Multiples vs market
    scores.append(multiples_result.get("value_score", 50))

    # 3. Analyst target upside
    target = safe_float(metrics.get("target_price"))
    if target and price:
        upside = (target - price) / price * 100
        if upside >= 25:    scores.append(90)
        elif upside >= 15:  scores.append(72)
        elif upside >= 5:   scores.append(57)
        elif upside >= 0:   scores.append(44)
        else:               scores.append(22)

    # 4. PEG ratio
    peg = safe_float(metrics.get("peg_ratio"))
    if peg and peg > 0:
        if peg <= 0.75:  scores.append(90)
        elif peg <= 1.0: scores.append(75)
        elif peg <= 1.5: scores.append(58)
        elif peg <= 2.0: scores.append(42)
        else:            scores.append(22)

    return clamp(float(np.mean(scores)) if scores else 50.0, 0, 100)


def score_growth(metrics: Dict[str, Any], growth_data: Dict[str, Any]) -> float:
    """Growth dimension score 0-100."""
    scores: List[float] = []

    def _rate_score(rate: Optional[float]) -> Optional[float]:
        if rate is None:
            return None
        if rate >= 0.30:   return 95.0
        elif rate >= 0.20: return 82.0
        elif rate >= 0.12: return 68.0
        elif rate >= 0.06: return 54.0
        elif rate >= 0.01: return 40.0
        elif rate >= 0.00: return 28.0
        else:              return 14.0

    rev_g = metrics.get("revenue_growth") or growth_data.get("avg_revenue_growth")
    eps_g = metrics.get("earnings_growth") or growth_data.get("avg_ni_growth")
    fcf_g = growth_data.get("avg_fcf_growth")
    eq_g  = metrics.get("earnings_quarterly_growth")

    for g in [rev_g, eps_g, fcf_g, eq_g]:
        s = _rate_score(g)
        if s is not None:
            scores.append(s)

    return clamp(float(np.mean(scores)) if scores else 50.0, 0, 100)


def score_quality(metrics: Dict[str, Any], health: Dict[str, Any]) -> float:
    """Quality dimension score 0-100 (margins, ROE, ROIC, balance sheet)."""
    scores: List[float] = []

    # Net margin
    nm = safe_float(metrics.get("profit_margin"))
    if nm is not None:
        if nm >= 0.25:   scores.append(95)
        elif nm >= 0.15: scores.append(80)
        elif nm >= 0.08: scores.append(65)
        elif nm >= 0.03: scores.append(50)
        elif nm >= 0.0:  scores.append(35)
        else:            scores.append(15)

    # Gross margin
    gm = safe_float(metrics.get("gross_margin"))
    if gm is not None:
        if gm >= 0.60:   scores.append(92)
        elif gm >= 0.40: scores.append(78)
        elif gm >= 0.25: scores.append(62)
        elif gm >= 0.15: scores.append(48)
        else:            scores.append(28)

    # ROE
    roe = safe_float(metrics.get("roe"))
    if roe is not None:
        if roe >= 0.30:   scores.append(92)
        elif roe >= 0.20: scores.append(78)
        elif roe >= 0.12: scores.append(64)
        elif roe >= 0.05: scores.append(48)
        elif roe >= 0.0:  scores.append(30)
        else:             scores.append(12)

    # ROIC
    roic = safe_float(health.get("roic"))
    if roic is not None:
        if roic >= 0.25:   scores.append(92)
        elif roic >= 0.15: scores.append(78)
        elif roic >= 0.09: scores.append(62)
        elif roic >= 0.0:  scores.append(40)
        else:              scores.append(18)

    # Debt/EBITDA (lower = better)
    d_ebitda = safe_float(health.get("debt_to_ebitda"))
    if d_ebitda is not None and d_ebitda > 0:
        if d_ebitda <= 0.5: scores.append(95)
        elif d_ebitda <= 1.5: scores.append(80)
        elif d_ebitda <= 2.5: scores.append(62)
        elif d_ebitda <= 3.5: scores.append(45)
        elif d_ebitda <= 5.0: scores.append(28)
        else:                  scores.append(12)

    return clamp(float(np.mean(scores)) if scores else 50.0, 0, 100)


def score_macro(metrics: Dict[str, Any]) -> float:
    """Macro/sentiment dimension score 0-100."""
    scores: List[float] = []

    # Beta
    beta = safe_float(metrics.get("beta"))
    if beta is not None:
        if 0.4 <= beta <= 0.9:   scores.append(80)
        elif 0.9 < beta <= 1.2:  scores.append(65)
        elif beta < 0.4:         scores.append(60)
        elif 1.2 < beta <= 1.6:  scores.append(48)
        else:                     scores.append(30)

    # Analyst consensus (1=Strong Buy … 5=Strong Sell)
    rec = safe_float(metrics.get("analyst_rec"))
    if rec is not None:
        if rec <= 1.5:   scores.append(92)
        elif rec <= 2.0: scores.append(78)
        elif rec <= 2.5: scores.append(62)
        elif rec <= 3.0: scores.append(48)
        elif rec <= 3.5: scores.append(32)
        else:             scores.append(12)

    # 52-week price position (near low = opportunity)
    price = safe_float(metrics.get("current_price"))
    hi    = safe_float(metrics.get("week_52_high"))
    lo    = safe_float(metrics.get("week_52_low"))
    if price and hi and lo and hi > lo:
        pos = (price - lo) / (hi - lo)
        if pos <= 0.20:   scores.append(80)
        elif pos <= 0.40: scores.append(68)
        elif pos <= 0.60: scores.append(55)
        elif pos <= 0.80: scores.append(45)
        else:              scores.append(35)

    return clamp(float(np.mean(scores)) if scores else 50.0, 0, 100)


def calculate_final_score(
    value: float,
    growth: float,
    quality: float,
    macro: float,
) -> float:
    """Weighted composite score."""
    return clamp(
        value   * 0.30
        + growth  * 0.30
        + quality * 0.25
        + macro   * 0.15,
        0, 100,
    )


# ─── Signal Generation ────────────────────────────────────────────────────────

def generate_signal(
    score: float,
    mos: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    """Return (signal, color, reasoning)."""
    if score >= 65:
        signal, color = "BUY", "#00E676"
        if mos and mos >= 20:
            reason = f"Strong margin of safety ({mos:.1f}%) with solid composite score of {score:.0f}/100"
        elif score >= 80:
            reason = "Exceptional fundamentals across value, growth, and quality dimensions"
        else:
            reason = f"Attractive risk-reward profile — composite score {score:.0f}/100"
    elif score >= 40:
        signal, color = "HOLD", "#FFD600"
        reason = f"Mixed signals — score {score:.0f}/100. Monitor for better entry or catalyst"
    else:
        signal, color = "SELL", "#FF1744"
        if mos and mos < -25:
            reason = f"Significant overvaluation ({-mos:.1f}% premium to intrinsic value)"
        else:
            reason = f"Weak fundamentals or unfavorable risk-reward — score {score:.0f}/100"

    return signal, color, reason


# ─── Full Valuation Pipeline ──────────────────────────────────────────────────

def run_full_valuation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Master entry point. Returns a complete valuation dict including
    metrics, DCF, multiples, growth, health, scores, and signal.
    """
    metrics = extract_key_metrics(data)

    # DCF
    dcf = calculate_dcf(
        free_cash_flow=metrics.get("free_cash_flow"),
        revenue=metrics.get("revenue"),
        revenue_growth=metrics.get("revenue_growth") or 0.05,
        shares_outstanding=metrics.get("shares_outstanding"),
        beta=metrics.get("beta") or 1.0,
        total_debt=metrics.get("total_debt") or 0.0,
        total_cash=metrics.get("total_cash") or 0.0,
        market_cap=metrics.get("market_cap") or 0.0,
    )

    mos = calculate_margin_of_safety(
        current_price=metrics.get("current_price") or 0,
        intrinsic_value=dcf.get("intrinsic_value") or 0,
    )

    # Multiples
    multiples = calculate_comparable_multiples(metrics)

    # Growth
    growth = calculate_growth_analysis(
        data.get("financials"),
        data.get("cash_flow"),
        metrics,
    )

    # Health
    health = calculate_financial_health(metrics, data.get("balance_sheet"))

    # Scores
    v_score = score_value(metrics, dcf, multiples)
    g_score = score_growth(metrics, growth)
    q_score = score_quality(metrics, health)
    m_score = score_macro(metrics)
    final   = calculate_final_score(v_score, g_score, q_score, m_score)

    signal, sig_color, sig_reason = generate_signal(final, mos, metrics)

    return {
        "metrics":          metrics,
        "dcf":              dcf,
        "margin_of_safety": mos,
        "multiples":        multiples,
        "growth":           growth,
        "health":           health,
        "scores": {
            "value":   v_score,
            "growth":  g_score,
            "quality": q_score,
            "macro":   m_score,
            "final":   final,
        },
        "signal":        signal,
        "signal_color":  sig_color,
        "signal_reason": sig_reason,
    }

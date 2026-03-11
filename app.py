"""
app.py — Bomers Alpha Institutional Stock Analysis Platform
Bloomberg-style dark dashboard built with Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np

from modules.data_fetcher import get_full_stock_data, extract_key_metrics
from modules.valuation import run_full_valuation
from modules.charts import (
    price_history_chart,
    valuation_comparison_chart,
    dcf_waterfall_chart,
    revenue_growth_chart,
    margin_trends_chart,
    score_radar_chart,
    intrinsic_value_gauge,
    screener_scatter,
)
from modules.ai_analysis import generate_ai_analysis, generate_beginner_explanation
from modules.screener import run_screener, filter_screener, DEFAULT_TICKERS
from modules.utils import (
    format_large_number,
    format_percentage,
    format_multiple,
    format_price,
    score_to_color,
    safe_float,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bomers Alpha — Institutional Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset / Base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp                     { background: #0A0E1A; color: #E0E6F0; }
.main .block-container     { padding: 1.2rem 2rem 2rem 2rem; max-width: 100%; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"]  { background: #0C1020; border-right: 1px solid #1E2235; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── st.metric — dark theme override ─────────────────────── */
[data-testid="stMetric"] {
    background: #131722 !important;
    border: 1px solid #1E2235 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    transition: border-color .2s !important;
}
[data-testid="stMetric"]:hover { border-color: #2962FF !important; }
[data-testid="stMetricLabel"] p {
    color: #8892A4 !important; font-size: 10px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: .09em !important; margin: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 700 !important;
    color: #E0E6F0 !important; line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }
[data-testid="stMetricDelta"] > div {
    font-size: 11px !important; margin-top: 4px !important;
}
/* Tooltip help icon */
[data-testid="stMetricHelpIcon"] { color: #3A4562 !important; }
[data-testid="stMetricHelpIcon"]:hover { color: #8892A4 !important; }

/* ── Inputs ───────────────────────────────────────────────── */
.stTextInput input {
    background: #131722 !important; border: 1px solid #2A3050 !important;
    color: #E0E6F0 !important; border-radius: 6px !important; font-size: 14px !important;
}
.stTextInput input:focus {
    border-color: #2962FF !important;
    box-shadow: 0 0 0 2px rgba(41,98,255,0.18) !important;
}
.stSelectbox > div > div {
    background: #131722 !important; border: 1px solid #2A3050 !important;
    color: #E0E6F0 !important; border-radius: 6px !important;
}

/* ── Mode toggle (radio) ──────────────────────────────────── */
[data-testid="stSidebar"] .stRadio > label {
    color: #8892A4 !important; font-size: 10px !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .09em !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
    display: none !important;
}
.mode-chip {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600; cursor: pointer;
    border: 1px solid #2A3050; color: #8892A4; background: transparent;
}
.mode-chip-active {
    background: rgba(41,98,255,0.18) !important;
    border-color: #2962FF !important; color: #E0E6F0 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: #2962FF !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 8px 22px !important; transition: background .2s !important;
}
.stButton > button:hover { background: #1A50DD !important; }

/* Secondary button style (Explain this analysis) */
.btn-secondary > button {
    background: transparent !important;
    border: 1px solid #2A3050 !important;
    color: #C0CAD8 !important;
}
.btn-secondary > button:hover {
    background: rgba(41,98,255,0.1) !important;
    border-color: #2962FF !important; color: #E0E6F0 !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid #1E2235; gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent; color: #8892A4; border-radius: 4px 4px 0 0;
    font-size: 12px; font-weight: 500; padding: 8px 16px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #131722 !important; color: #E0E6F0 !important;
    border-bottom: 2px solid #2962FF !important;
}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0f1424 !important;
    border: 1px solid #1E2235 !important;
    border-radius: 8px !important;
    margin-bottom: 16px !important;
}
[data-testid="stExpander"] summary {
    color: #8892A4 !important; font-size: 12.5px !important;
    font-weight: 500 !important; padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover { color: #E0E6F0 !important; }

/* ── Progress bar ─────────────────────────────────────────── */
.stProgress > div > div { background: #2962FF !important; }

/* ── Custom components ────────────────────────────────────── */
.kpi-card {
    background: #131722; border: 1px solid #1E2235; border-radius: 8px;
    padding: 14px 18px; transition: border-color .2s; height: 100%;
}
.kpi-card:hover { border-color: #2962FF; }
.kpi-label {
    color: #8892A4; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .09em; margin-bottom: 6px;
}
.kpi-value { color: #E0E6F0; font-size: 24px; font-weight: 700; line-height: 1.15; }
.kpi-sub   { color: #8892A4; font-size: 11px; margin-top: 5px; }

.score-card {
    background: #131722; border: 1px solid #1E2235; border-radius: 8px;
    padding: 16px; text-align: center;
}
.score-num { font-size: 34px; font-weight: 700; margin: 6px 0 4px; }
.score-bar-bg { background: #1E2235; border-radius: 3px; height: 4px; }

/* ── Signal badges ────────────────────────────────────────── */
.sig-buy  { background:rgba(0,230,118,.13); border:1px solid #00E676;
            color:#00E676; padding:3px 14px; border-radius:4px;
            font-weight:700; font-size:14px; display:inline-block; }
.sig-hold { background:rgba(255,214,0,.13); border:1px solid #FFD600;
            color:#FFD600; padding:3px 14px; border-radius:4px;
            font-weight:700; font-size:14px; display:inline-block; }
.sig-sell { background:rgba(255,23,68,.13);  border:1px solid #FF1744;
            color:#FF1744; padding:3px 14px; border-radius:4px;
            font-weight:700; font-size:14px; display:inline-block; }

/* ── Fin rows (right panel) ───────────────────────────────── */
.fin-row {
    display:flex; justify-content:space-between; align-items:center;
    padding: 5px 0; border-bottom: 1px solid #1a1f30; font-size: 12.5px;
}
.fin-lbl { color: #8892A4; }
.fin-val { color: #E0E6F0; font-weight: 500; font-variant-numeric: tabular-nums; }
.fin-pos  { color: #00E676; font-weight: 500; }
.fin-neg  { color: #FF1744; font-weight: 500; }

/* ── Section headers ──────────────────────────────────────── */
.section-hdr {
    color: #8892A4; font-size: 9.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .12em;
    padding: 10px 0 6px; border-bottom: 1px solid #1E2235; margin-bottom: 10px;
}

/* ── Info callout boxes ───────────────────────────────────── */
.callout {
    background: #0f1424; border-left: 3px solid #2962FF;
    border-radius: 0 6px 6px 0; padding: 12px 16px;
    margin: 12px 0; font-size: 12.5px; color: #8892A4; line-height: 1.7;
}
.callout-green  { border-color: #00E676 !important; }
.callout-yellow { border-color: #FFD600 !important; }
.callout-red    { border-color: #FF1744 !important; }

/* ── How-to pill labels ───────────────────────────────────── */
.pill {
    display: inline-block; background: rgba(41,98,255,0.12);
    border: 1px solid rgba(41,98,255,0.3); color: #7B9FFF;
    padding: 1px 8px; border-radius: 10px; font-size: 10px;
    font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
    margin-right: 6px; vertical-align: middle;
}
.pill-green  { background:rgba(0,230,118,.1);  border-color:rgba(0,230,118,.3);  color:#00E676; }
.pill-yellow { background:rgba(255,214,0,.1);  border-color:rgba(255,214,0,.3);  color:#FFD600; }
.pill-red    { background:rgba(255,23,68,.1);  border-color:rgba(255,23,68,.3);  color:#FF1744; }

.panel {
    background: #131722; border: 1px solid #1E2235;
    border-radius: 8px; padding: 18px;
}

/* ── Beginner explanation box ─────────────────────────────── */
.explain-box {
    background: #0d1526; border: 1px solid #1E3A5F;
    border-radius: 8px; padding: 20px 22px;
    font-size: 13.5px; color: #C4CEDB; line-height: 1.8;
}
.explain-box h4 { color: #E0E6F0; margin: 0 0 10px; font-size: 14px; }

/* ── Misc ─────────────────────────────────────────────────── */
hr       { border-color: #1E2235 !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #2A3050; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:6px 0 22px;">
        <div style="font-size:22px;font-weight:700;color:#E0E6F0;letter-spacing:-0.5px;">
            📊 Bomers Alpha
        </div>
        <div style="font-size:10.5px;color:#8892A4;margin-top:3px;letter-spacing:.04em;">
            INSTITUTIONAL ANALYSIS PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["Stock Analysis", "Stock Screener"],
                    label_visibility="collapsed")

    # ── Beginner / Advanced Mode toggle ───────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hdr">View Mode</div>
    """, unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["Beginner", "Advanced"],
        index=0,
        horizontal=True,
        help="Beginner: simplified view with plain-English explanations. "
             "Advanced: full financial detail, DCF model, and all metrics.",
        label_visibility="collapsed",
    )
    beginner = (mode == "Beginner")

    if beginner:
        st.markdown("""
        <div style="font-size:11px;color:#8892A4;margin-top:6px;line-height:1.6;
                    background:#0f1424;border:1px solid #1E2235;border-radius:5px;padding:8px 10px;">
            Showing key signals only.<br>Switch to <b style="color:#E0E6F0;">Advanced</b>
            for full models and metrics.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:11px;color:#8892A4;margin-top:6px;line-height:1.6;
                    background:#0f1424;border:1px solid #1E2235;border-radius:5px;padding:8px 10px;">
            Full analysis mode — all models, multiples, and financial ratios visible.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">API Settings</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Claude API Key (optional)",
        type="password",
        placeholder="sk-ant-api03-…",
        help="Optional — enables AI-powered narrative analysis via Claude claude-opus-4-6.",
    )

    st.markdown("""
    <div style="margin-top:22px;padding:12px;background:#0f1424;border:1px solid #1E2235;border-radius:6px;">
        <div class="section-hdr" style="margin:0 0 8px;">Data Sources</div>
        <div style="font-size:11px;color:#8892A4;line-height:1.9;">
            • Yahoo Finance (yfinance)<br>
            • Real-time &amp; delayed quotes<br>
            • Annual financial statements<br>
            • Analyst estimates &amp; targets
        </div>
    </div>
    <div style="margin-top:12px;padding:12px;background:#0f1424;border:1px solid #1E2235;border-radius:6px;">
        <div class="section-hdr" style="margin:0 0 8px;">Valuation Models</div>
        <div style="font-size:11px;color:#8892A4;line-height:1.9;">
            • Two-Stage DCF (FCF-based)<br>
            • Comparable Multiples<br>
            • Growth Scoring<br>
            • Quality &amp; Balance Sheet<br>
            • Macro / Sentiment Layer
        </div>
    </div>
    <div style="margin-top:20px;font-size:9.5px;color:#2A3050;text-align:center;line-height:1.6;">
        For informational purposes only.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ─── Shared Helpers ───────────────────────────────────────────────────────────

def fin_row(label: str, value_html: str):
    st.markdown(
        f'<div class="fin-row"><span class="fin-lbl">{label}</span>'
        f'<span>{value_html}</span></div>',
        unsafe_allow_html=True,
    )


def score_card_html(label: str, score: float, color: str) -> str:
    return f"""
    <div class="score-card">
        <div class="kpi-label">{label}</div>
        <div class="score-num" style="color:{color};">{score:.0f}</div>
        <div class="score-bar-bg">
            <div style="background:{color};width:{score:.0f}%;height:4px;border-radius:3px;"></div>
        </div>
    </div>"""


def _metric_html(val, pct=False, mult=False, dollar=False, invert=False) -> str:
    if val is None:
        return '<span class="fin-val">—</span>'
    v = safe_float(val)
    if v is None:
        return '<span class="fin-val">—</span>'
    if dollar:
        txt = format_price(v)
        cls = "fin-val"
    elif pct:
        txt = f"{v * 100:+.1f}%"
        cls = ("fin-neg" if (v > 0) else "fin-pos") if invert else ("fin-pos" if v > 0 else "fin-neg")
    elif mult:
        txt = f"{v:.1f}x" if v > 0 else "—"
        cls = "fin-val"
    else:
        txt = f"{v:.2f}"
        cls = "fin-val"
    return f'<span class="{cls}">{txt}</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if page == "Stock Analysis":

    # ── 1. HOW TO READ THIS DASHBOARD ─────────────────────────────────────────
    with st.expander("📖  How to read this dashboard", expanded=False):
        st.markdown("""
        <div style="font-size:13px;color:#C0CAD8;line-height:1.8;padding:4px 0;">

        <b style="color:#E0E6F0;font-size:14px;">A quick guide to the key numbers</b><br><br>

        <span class="pill">Current Price</span>
        The price you would pay right now to buy one share of the company on the stock market.
        Think of it as the price tag on the shelf.<br><br>

        <span class="pill">Intrinsic Value</span>
        Our model's estimate of what the stock is <em>actually worth</em>, based on the company's
        expected future cash flows. If intrinsic value is higher than the current price, the stock
        may be undervalued — like finding something on sale.<br><br>

        <span class="pill">Margin of Safety</span>
        The gap between intrinsic value and current price, shown as a percentage.
        <span class="pill-green">Green / positive</span> means the stock looks cheap vs our estimate.
        <span class="pill-red">Red / negative</span> means it looks expensive.
        A margin of safety above 20% is generally considered attractive by value investors.<br><br>

        <span class="pill">Investment Score</span>
        A single number from 0 to 100 that combines valuation, growth, profitability, and market
        signals. Think of it like a hotel star rating — higher is better.
        <span class="pill-green">65–100 = strong</span> &nbsp;
        <span class="pill-yellow">40–64 = mixed</span> &nbsp;
        <span class="pill-red">0–39 = weak</span><br><br>

        <span class="pill">BUY / HOLD / SELL Signal</span>
        A plain-English recommendation derived from the score and margin of safety.
        <span class="pill-green">BUY</span> — the model finds the stock attractively priced.
        <span class="pill-yellow">HOLD</span> — fairly valued; no urgent action needed.
        <span class="pill-red">SELL</span> — the stock looks overpriced or fundamentally weak.<br><br>

        <b style="color:#8892A4;font-size:11px;">
        ⚠️ All analysis is for informational purposes only and is not financial advice.
        Always do your own research before investing.
        </b>

        </div>
        """, unsafe_allow_html=True)

    # ── Ticker Search Bar ──────────────────────────────────────────────────────
    c_ticker, c_period = st.columns([4, 1])
    with c_ticker:
        ticker_input = st.text_input(
            "Ticker",
            value=st.session_state.get("last_ticker", "AAPL"),
            placeholder="Enter ticker symbol  (e.g. AAPL · MSFT · NVDA · TSLA)",
            label_visibility="collapsed",
        )
    with c_period:
        chart_period = st.selectbox(
            "Period", ["2y", "5y", "1y", "6mo", "3mo"],
            label_visibility="collapsed",
        )

    if not ticker_input:
        st.stop()

    ticker = ticker_input.upper().strip()
    st.session_state["last_ticker"] = ticker

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for **{ticker}** …"):
        data = get_full_stock_data(ticker)

    if "error" in data and not data.get("info"):
        st.error(f"Could not load data for **{ticker}**. Please verify the symbol and try again.")
        st.stop()

    with st.spinner("Running valuation engine …"):
        try:
            valuation = run_full_valuation(data)
        except Exception as exc:
            st.error(f"Valuation error: {exc}")
            st.stop()

    metrics     = valuation["metrics"]
    scores      = valuation["scores"]
    dcf         = valuation["dcf"]
    health      = valuation["health"]
    growth_data = valuation["growth"]
    signal      = valuation["signal"]
    mos         = valuation.get("margin_of_safety")
    final_score = scores["final"]

    price  = safe_float(metrics.get("current_price"))
    iv     = safe_float(dcf.get("intrinsic_value"))
    target = safe_float(metrics.get("target_price"))

    # Clear cached AI output when ticker changes
    if st.session_state.get("ai_ticker") != ticker:
        st.session_state.pop("ai_analysis", None)
        st.session_state.pop("beginner_explanation", None)
        st.session_state["ai_ticker"] = ticker

    # ── Company Header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:18px;">
        <div style="font-size:11.5px;color:#8892A4;margin-bottom:5px;">
            {metrics.get('sector','—')} &nbsp;·&nbsp; {metrics.get('industry','—')} &nbsp;·&nbsp; {metrics.get('country','—')}
        </div>
        <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
            <span style="font-size:34px;font-weight:700;color:#E0E6F0;letter-spacing:-1px;">{ticker}</span>
            <span style="font-size:17px;color:#C0CAD8;font-weight:400;">{metrics.get('company_name','')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. KPI ROW — st.metric() with help tooltips ────────────────────────────
    mos_color = "#00E676" if (mos or 0) >= 20 else ("#FFD600" if (mos or 0) >= 0 else "#FF1744")
    mos_str   = f"{mos:+.1f}%" if mos is not None else "N/A"

    if target and price:
        upside     = (target - price) / price * 100
        target_str = format_price(target)
        target_delta = f"{upside:+.1f}% upside"
    else:
        target_str   = "N/A"
        target_delta = None

    sc_color = score_to_color(final_score)

    if beginner:
        # Beginner: 4 cards — Price, IV, MoS, Signal
        kc = st.columns(4)
    else:
        # Advanced: 6 cards
        kc = st.columns(6)

    with kc[0]:
        st.metric(
            "Current Price",
            format_price(price),
            help=(
                "The live market price of one share. "
                "This is what you would pay to buy the stock right now."
            ),
        )

    with kc[1]:
        st.metric(
            "Intrinsic Value (DCF)",
            format_price(iv),
            delta=mos_str if mos is not None else None,
            delta_color="normal" if (mos or 0) >= 0 else "inverse",
            help=(
                "Our estimate of the stock's fair value, calculated by projecting the company's "
                "future free cash flows and discounting them back to today. "
                "If this is higher than the current price, the stock may be undervalued."
            ),
        )

    with kc[2]:
        st.metric(
            "Margin of Safety",
            mos_str,
            help=(
                "The percentage gap between intrinsic value and current price. "
                "Positive = stock looks cheap vs our model. "
                "Negative = stock looks expensive. "
                "A value above 20% is generally considered a comfortable buffer for error."
            ),
        )

    if beginner:
        with kc[3]:
            sig_cls = f"sig-{signal.lower()}"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Signal</div>
                <div style="margin-top:7px;"><span class="{sig_cls}">{signal}</span></div>
                <div class="kpi-sub">{valuation.get('signal_reason','')[:55]}…</div>
            </div>""", unsafe_allow_html=True)
    else:
        with kc[3]:
            st.metric(
                "Analyst Target",
                target_str,
                delta=target_delta,
                delta_color="normal",
                help=(
                    "The average 12-month price target set by Wall Street analysts. "
                    "This is not the same as our DCF intrinsic value — it reflects "
                    "professional analyst consensus."
                ),
            )

        with kc[4]:
            st.metric(
                "Overall Score",
                f"{final_score:.0f} / 100",
                help=(
                    "A composite score from 0 to 100 combining valuation (30%), growth (30%), "
                    "quality (25%), and macro factors (15%). "
                    "65+ = BUY territory. 40–64 = HOLD. Below 40 = SELL."
                ),
            )

        with kc[5]:
            sig_cls = f"sig-{signal.lower()}"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Signal</div>
                <div style="margin-top:7px;"><span class="{sig_cls}">{signal}</span></div>
                <div class="kpi-sub">{valuation.get('signal_reason','')[:55]}…</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # ── Price Chart + Right Panel ──────────────────────────────────────────────
    if beginner:
        # Beginner: full-width chart, no right panel
        chart_col = st.container()
        panel_visible = False
    else:
        col_chart, col_panel = st.columns([2.6, 1])
        chart_col   = col_chart
        panel_visible = True

    with chart_col:
        hist_key = "price_history_5y" if chart_period == "5y" else "price_history_2y"
        hist = data.get(hist_key, pd.DataFrame())
        if chart_period in ("1y", "6mo", "3mo"):
            days = {"1y": 365, "6mo": 183, "3mo": 91}[chart_period]
            if not hist.empty:
                hist = hist.tail(days)

        fig_price = price_history_chart(hist, ticker, intrinsic_value=iv, target_price=target)
        st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

        # Beginner: show a brief plain-English chart caption
        if beginner and iv and price:
            diff_pct = (iv - price) / price * 100
            if diff_pct > 10:
                caption_color = "#00E676"
                caption = (
                    f"The green dashed line shows our intrinsic value estimate of {format_price(iv)}. "
                    f"The stock is currently trading about {abs(diff_pct):.0f}% <b>below</b> "
                    f"that estimate — suggesting it may be undervalued."
                )
            elif diff_pct < -10:
                caption_color = "#FF1744"
                caption = (
                    f"The green dashed line shows our intrinsic value estimate of {format_price(iv)}. "
                    f"The stock is currently trading about {abs(diff_pct):.0f}% <b>above</b> "
                    f"that estimate — suggesting it may be overvalued."
                )
            else:
                caption_color = "#FFD600"
                caption = (
                    f"The green dashed line shows our intrinsic value estimate of {format_price(iv)}. "
                    f"The stock is trading close to that level — suggesting it is fairly valued."
                )
            st.markdown(
                f'<div class="callout" style="border-color:{caption_color};">{caption}</div>',
                unsafe_allow_html=True,
            )

    if panel_visible:
        with col_panel:
            st.markdown('<div class="section-hdr">Valuation Multiples</div>', unsafe_allow_html=True)
            fin_row("P/E (TTM)",      _metric_html(metrics.get("pe_ratio"),    mult=True))
            fin_row("Forward P/E",    _metric_html(metrics.get("forward_pe"),  mult=True))
            fin_row("EV / EBITDA",    _metric_html(metrics.get("ev_ebitda"),   mult=True))
            fin_row("EV / Revenue",   _metric_html(metrics.get("ev_revenue"),  mult=True))
            fin_row("Price / Book",   _metric_html(metrics.get("pb_ratio"),    mult=True))
            fin_row("Price / Sales",  _metric_html(metrics.get("ps_ratio"),    mult=True))
            fin_row("PEG Ratio",      _metric_html(metrics.get("peg_ratio"),   mult=True))

            st.markdown('<div class="section-hdr" style="margin-top:14px;">Financials</div>',
                        unsafe_allow_html=True)
            fin_row("Market Cap",
                    f'<span class="fin-val">{format_large_number(metrics.get("market_cap"))}</span>')
            fin_row("Revenue (TTM)",
                    f'<span class="fin-val">{format_large_number(metrics.get("revenue"))}</span>')
            fin_row("EBITDA",
                    f'<span class="fin-val">{format_large_number(metrics.get("ebitda"))}</span>')
            fin_row("Free Cash Flow",
                    f'<span class="fin-val">{format_large_number(metrics.get("free_cash_flow"))}</span>')
            fin_row("Net Debt",
                    f'<span class="fin-val">{format_large_number(health.get("net_debt"))}</span>')

            st.markdown('<div class="section-hdr" style="margin-top:14px;">Profitability</div>',
                        unsafe_allow_html=True)
            fin_row("Revenue Growth",   _metric_html(metrics.get("revenue_growth"),   pct=True))
            fin_row("EPS Growth",       _metric_html(metrics.get("earnings_growth"),  pct=True))
            fin_row("Gross Margin",     _metric_html(metrics.get("gross_margin"),     pct=True))
            fin_row("Operating Margin", _metric_html(metrics.get("operating_margin"), pct=True))
            fin_row("Net Margin",       _metric_html(metrics.get("profit_margin"),    pct=True))
            fin_row("ROE",              _metric_html(metrics.get("roe"),              pct=True))
            fin_row("ROIC",             _metric_html(health.get("roic"),              pct=True))
            fin_row("Debt / EBITDA",    _metric_html(health.get("debt_to_ebitda"),   mult=True))
            fin_row("Dividend Yield",   _metric_html(metrics.get("dividend_yield"),  pct=True))

            beta = safe_float(metrics.get("beta"))
            if beta:
                fin_row("Beta", f'<span class="fin-val">{beta:.2f}</span>')

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── 5. SCORING MODEL — description + cards ────────────────────────────────
    st.markdown('<div class="section-hdr">Scoring Model</div>', unsafe_allow_html=True)

    # Plain-English description callout
    st.markdown("""
    <div class="callout" style="margin-bottom:14px;">
        The overall score is built from four dimensions, each measuring a different aspect of the stock:<br>
        <b style="color:#E0E6F0;">Value (30%)</b> — Is the price reasonable? Compares current price to our DCF estimate and market averages.<br>
        <b style="color:#E0E6F0;">Growth (30%)</b> — Is the business growing? Measures revenue, earnings, and cash flow growth.<br>
        <b style="color:#E0E6F0;">Quality (25%)</b> — Is it a good business? Checks profit margins, return on capital, and debt levels.<br>
        <b style="color:#E0E6F0;">Macro (15%)</b> — What does the market think? Uses analyst ratings, price momentum, and volatility (Beta).
    </div>
    """, unsafe_allow_html=True)

    if beginner:
        # Beginner: show only final score, large and clear
        fc = scores.get("final", 0)
        fc_color = score_to_color(fc)
        sig_word = "Strong" if fc >= 65 else ("Mixed" if fc >= 40 else "Weak")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;background:#131722;
                    border:1px solid #1E2235;border-radius:8px;padding:20px 24px;margin-bottom:8px;">
            <div style="text-align:center;min-width:90px;">
                <div style="font-size:52px;font-weight:700;color:{fc_color};line-height:1;">{fc:.0f}</div>
                <div style="font-size:10px;color:#8892A4;text-transform:uppercase;letter-spacing:.1em;margin-top:4px;">out of 100</div>
            </div>
            <div>
                <div style="font-size:18px;font-weight:600;color:#E0E6F0;margin-bottom:4px;">
                    {sig_word} Overall Rating
                </div>
                <div style="font-size:13px;color:#8892A4;line-height:1.6;">
                    {"Scores above 65 indicate a potentially attractive investment opportunity." if fc >= 65
                     else "Scores between 40–64 suggest the stock is fairly valued — no urgent action." if fc >= 40
                     else "Scores below 40 suggest caution — the stock may be overvalued or have weak fundamentals."
                    }
                </div>
                <div style="background:#1E2235;border-radius:4px;height:6px;margin-top:10px;max-width:300px;">
                    <div style="background:{fc_color};width:{fc:.0f}%;height:6px;border-radius:4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Advanced: all 5 score cards
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        for col, label, key in [
            (sc1, "Value",       "value"),
            (sc2, "Growth",      "growth"),
            (sc3, "Quality",     "quality"),
            (sc4, "Macro",       "macro"),
            (sc5, "FINAL SCORE", "final"),
        ]:
            s = scores.get(key, 0)
            c = score_to_color(s)
            with col:
                st.markdown(score_card_html(label, s, c), unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # ── Chart Tabs ─────────────────────────────────────────────────────────────
    if beginner:
        # Beginner mode: show only a compact valuation summary tab
        st.markdown('<div class="section-hdr">Valuation Summary</div>', unsafe_allow_html=True)
        if price and iv:
            col_gauge, col_expl = st.columns([1, 1.4])
            with col_gauge:
                st.plotly_chart(intrinsic_value_gauge(price, iv),
                                use_container_width=True, config={"displayModeBar": False})
            with col_expl:
                mos_pct = mos or 0
                if mos_pct >= 20:
                    verdict = "potentially <b style='color:#00E676;'>undervalued</b>"
                    msg = f"Our model suggests the stock is trading about {mos_pct:.0f}% below its estimated fair value. This represents a meaningful margin of safety."
                elif mos_pct >= 0:
                    verdict = "<b style='color:#FFD600;'>fairly valued</b>"
                    msg = f"The stock is trading close to our intrinsic value estimate. There is a small buffer of {mos_pct:.0f}%, but limited upside from a valuation perspective."
                else:
                    verdict = "potentially <b style='color:#FF1744;'>overvalued</b>"
                    msg = f"The stock is trading about {abs(mos_pct):.0f}% above our estimated fair value. You may be paying a premium vs the model's assessment."
                st.markdown(f"""
                <div class="explain-box" style="margin-top:10px;">
                    <h4>What this means</h4>
                    Based on our discounted cash flow model, {metrics.get('company_name', ticker)}
                    appears {verdict} at the current price of {format_price(price)}.<br><br>
                    {msg}<br><br>
                    <b style="color:#8892A4;font-size:11px;">
                    Note: DCF models are estimates. The actual fair value can differ based on
                    future growth and economic conditions.
                    </b>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Intrinsic value could not be calculated — the company may not have sufficient free cash flow history.")
    else:
        # Advanced mode: all four tabs
        tab_val, tab_dcf, tab_growth, tab_radar = st.tabs(
            ["📊  Valuation", "💰  DCF Model", "📈  Growth & Margins", "🎯  Score Radar"]
        )

        with tab_val:
            col_v1, col_v2 = st.columns([1.3, 1])
            with col_v1:
                st.plotly_chart(valuation_comparison_chart(valuation),
                                use_container_width=True, config={"displayModeBar": False})
            with col_v2:
                if price and iv:
                    st.plotly_chart(intrinsic_value_gauge(price, iv),
                                    use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("DCF intrinsic value not available — check FCF data.")

        with tab_dcf:
            if dcf.get("fcf_forecasts"):
                st.plotly_chart(dcf_waterfall_chart(dcf),
                                use_container_width=True, config={"displayModeBar": False})

                # ── 4. DCF CHART EXPLANATIONS ──────────────────────────────────
                st.markdown("""
                <div class="callout" style="margin-top:4px;">
                    <b style="color:#E0E6F0;">What is Free Cash Flow (FCF)?</b><br>
                    FCF is the cash a company generates after paying its operating costs and
                    capital expenditures. It's the money left over that can be returned to
                    shareholders, used to pay down debt, or reinvested for growth.
                    The blue bars show our projected FCF for each year.
                </div>
                <div class="callout" style="margin-top:6px;">
                    <b style="color:#E0E6F0;">Why discount future cash flows?</b><br>
                    A dollar received in 10 years is worth less than a dollar today —
                    because of inflation, risk, and opportunity cost. Discounting converts
                    future cash flows into today's equivalent value. The cyan bars show
                    each year's FCF adjusted to present value.
                </div>
                <div class="callout" style="margin-top:6px;">
                    <b style="color:#E0E6F0;">How does WACC affect valuation?</b><br>
                    WACC (Weighted Average Cost of Capital) is the discount rate used.
                    A higher WACC means future cash flows are discounted more heavily,
                    resulting in a lower intrinsic value. It represents the return
                    investors require to hold the stock. Currently set at
                    <b style="color:#E0E6F0;">{wacc:.1f}%</b>.
                </div>
                """.format(wacc=(dcf.get("wacc", 0) * 100)), unsafe_allow_html=True)

                d1, d2, d3, d4, d5 = st.columns(5)
                with d1:
                    st.metric(
                        "WACC",
                        f"{dcf.get('wacc', 0)*100:.1f}%",
                        help="Weighted Average Cost of Capital — the discount rate applied to future "
                             "cash flows. Reflects the blended required return for equity and debt holders. "
                             "Lower WACC → higher intrinsic value.",
                    )
                with d2:
                    st.metric(
                        "Terminal Growth",
                        f"{dcf.get('terminal_growth', 0)*100:.1f}%",
                        help="The assumed annual growth rate of cash flows beyond the 10-year forecast "
                             "horizon — essentially the long-run steady-state growth rate. "
                             "Typically set close to long-run GDP growth (2–3%).",
                    )
                with d3:
                    st.metric(
                        "Base FCF Growth",
                        f"{dcf.get('base_growth', 0)*100:.1f}%",
                        help="The assumed free cash flow growth rate in years 1–5, "
                             "anchored to the company's recent revenue growth. "
                             "This rate gradually declines toward terminal growth in years 6–10.",
                    )
                with d4:
                    st.metric(
                        "Enterprise Value",
                        format_large_number(dcf.get("enterprise_value")),
                        help="Total estimated value of the business (equity + net debt), "
                             "calculated as the sum of all discounted future cash flows plus terminal value.",
                    )
                with d5:
                    st.metric(
                        "Intrinsic Value / Share",
                        format_price(iv),
                        help="Enterprise value minus net debt, divided by shares outstanding. "
                             "This is the per-share fair value estimate our model produces.",
                    )
            else:
                st.info(
                    "Insufficient free cash flow data to run DCF model. "
                    "This typically occurs for financial companies or early-stage firms."
                )

        with tab_growth:
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(revenue_growth_chart(growth_data, metrics),
                                use_container_width=True, config={"displayModeBar": False})
            with g2:
                st.plotly_chart(margin_trends_chart(data.get("financials"), metrics),
                                use_container_width=True, config={"displayModeBar": False})

        with tab_radar:
            r1, r2 = st.columns([1, 1])
            with r1:
                st.plotly_chart(score_radar_chart(scores),
                                use_container_width=True, config={"displayModeBar": False})
            with r2:
                st.markdown("""
                <div class="panel" style="margin-top:10px;font-size:12px;color:#8892A4;line-height:1.9;">
                    <div class="section-hdr">Score Methodology</div>
                    <b style="color:#E0E6F0;">Value (30%)</b><br>
                    DCF margin of safety, valuation multiples vs S&amp;P 500 averages,
                    analyst target upside, PEG ratio.<br><br>
                    <b style="color:#E0E6F0;">Growth (30%)</b><br>
                    Revenue growth, EPS growth, FCF growth — historical and forward estimates.<br><br>
                    <b style="color:#E0E6F0;">Quality (25%)</b><br>
                    Net/gross margins, ROE, ROIC, debt-to-EBITDA balance sheet health.<br><br>
                    <b style="color:#E0E6F0;">Macro (15%)</b><br>
                    Beta-adjusted risk, analyst consensus rating, 52-week price positioning.
                </div>""", unsafe_allow_html=True)

    # ── Separator ──────────────────────────────────────────────────────────────
    st.markdown("---")

    # ── AI ANALYSIS + EXPLAIN THIS ANALYSIS ────────────────────────────────────
    st.markdown('<div class="section-hdr">AI Investment Analysis</div>', unsafe_allow_html=True)

    if beginner:
        # ── 7. BEGINNER MODE — "Explain this analysis" button ─────────────────
        st.markdown("""
        <div style="font-size:12.5px;color:#8892A4;margin-bottom:12px;">
            Click the button below to get a plain-English summary of what this analysis means —
            no financial jargon.
        </div>""", unsafe_allow_html=True)

        explain_btn = st.button("💡  Explain this analysis", key="explain_btn")

        if explain_btn:
            with st.spinner("Generating plain-English explanation …"):
                explanation = generate_beginner_explanation(
                    ticker=ticker,
                    metrics=metrics,
                    valuation_result=valuation,
                    api_key=api_key or None,
                )
            st.session_state["beginner_explanation"] = explanation

        if "beginner_explanation" in st.session_state:
            st.markdown(
                f'<div class="explain-box">'
                + st.session_state["beginner_explanation"].replace("\n", "<br>")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div class="panel" style="text-align:center;padding:28px;color:#8892A4;font-size:13px;">
                Click <b style="color:#E0E6F0;">Explain this analysis</b> for a plain-English summary.<br>
                <span style="font-size:11px;">Works without an API key — AI-enhanced with a Claude key.</span>
            </div>""", unsafe_allow_html=True)

    else:
        # ── ADVANCED MODE — full AI analysis + explain button ─────────────────
        ai_col, stats_col = st.columns([2, 1])

        with ai_col:
            btn_row_l, btn_row_r = st.columns([1, 1])
            with btn_row_l:
                gen_btn = st.button("⚡  Generate Analysis", key="gen_ai")
            with btn_row_r:
                with st.container():
                    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
                    explain_adv_btn = st.button("💡  Explain this analysis", key="explain_adv")
                    st.markdown("</div>", unsafe_allow_html=True)

            if gen_btn:
                with st.spinner("Generating analysis …"):
                    analysis = generate_ai_analysis(
                        ticker=ticker,
                        metrics=metrics,
                        valuation_result=valuation,
                        api_key=api_key or None,
                    )
                st.session_state["ai_analysis"] = analysis

            if explain_adv_btn:
                with st.spinner("Generating plain-English explanation …"):
                    explanation = generate_beginner_explanation(
                        ticker=ticker,
                        metrics=metrics,
                        valuation_result=valuation,
                        api_key=api_key or None,
                    )
                st.session_state["beginner_explanation"] = explanation

            # Show whichever was generated last
            if "ai_analysis" in st.session_state and not explain_adv_btn:
                st.markdown(
                    f'<div class="panel" style="font-size:13px;line-height:1.75;color:#C4CEDB;">'
                    + st.session_state["ai_analysis"].replace("\n", "<br>")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            elif "beginner_explanation" in st.session_state:
                st.markdown(
                    f'<div class="explain-box">'
                    + st.session_state["beginner_explanation"].replace("\n", "<br>")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("""
                <div class="panel" style="text-align:center;padding:32px;color:#8892A4;font-size:13px;">
                    Click <b style="color:#E0E6F0;">Generate Analysis</b> for an institutional note,
                    or <b style="color:#E0E6F0;">Explain this analysis</b> for a plain-English summary.<br>
                    <span style="font-size:11px;">
                        Add a Claude API key in the sidebar for AI-powered narrative.
                        Both buttons work without an API key.
                    </span>
                </div>""", unsafe_allow_html=True)

        with stats_col:
            st.markdown('<div class="section-hdr">Snapshot</div>', unsafe_allow_html=True)

            hi52 = safe_float(metrics.get("week_52_high"))
            lo52 = safe_float(metrics.get("week_52_low"))
            if hi52 and lo52 and price and hi52 > lo52:
                pos = (price - lo52) / (hi52 - lo52) * 100
                fin_row("52W Low",  f'<span class="fin-val">{format_price(lo52)}</span>')
                fin_row("52W High", f'<span class="fin-val">{format_price(hi52)}</span>')
                st.markdown(f"""
                <div style="background:#1E2235;border-radius:3px;height:5px;margin:6px 0 12px;">
                    <div style="background:#2962FF;width:{pos:.0f}%;height:5px;border-radius:3px;"></div>
                </div>""", unsafe_allow_html=True)

            eps     = safe_float(metrics.get("eps_ttm"))
            eps_fwd = safe_float(metrics.get("eps_forward"))
            bv      = safe_float(metrics.get("book_value"))
            dy      = safe_float(metrics.get("dividend_yield"))
            emp     = metrics.get("employees")

            if eps:     fin_row("EPS (TTM)",     f'<span class="fin-val">${eps:.2f}</span>')
            if eps_fwd: fin_row("EPS (Fwd)",     f'<span class="fin-val">${eps_fwd:.2f}</span>')
            if bv:      fin_row("Book Value",    f'<span class="fin-val">${bv:.2f}</span>')
            if dy:      fin_row("Dividend Yield",f'<span class="fin-pos">{dy*100:.2f}%</span>')
            if emp:     fin_row("Employees",     f'<span class="fin-val">{emp:,}</span>')

            num_analysts = metrics.get("num_analysts")
            if num_analysts:
                fin_row("# Analysts", f'<span class="fin-val">{num_analysts}</span>')

            analyst_rec = safe_float(metrics.get("analyst_rec"))
            if analyst_rec:
                rec_map = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
                rec_lbl = rec_map.get(round(analyst_rec), f"{analyst_rec:.1f}")
                rec_col = "#00E676" if analyst_rec <= 2 else ("#FFD600" if analyst_rec <= 3 else "#FF1744")
                fin_row("Analyst Rec", f'<span style="color:{rec_col};font-weight:600;">{rec_lbl}</span>')


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — STOCK SCREENER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Stock Screener":

    st.markdown("""
    <div style="margin-bottom:22px;">
        <div style="font-size:28px;font-weight:700;color:#E0E6F0;letter-spacing:-0.5px;">
            Stock Screener
        </div>
        <div style="font-size:12.5px;color:#8892A4;margin-top:4px;">
            Scan a universe of stocks and rank them by valuation, growth, quality, and composite score
        </div>
    </div>
    """, unsafe_allow_html=True)

    if beginner:
        st.markdown("""
        <div class="callout" style="margin-bottom:16px;">
            <b style="color:#E0E6F0;">How to use the screener</b><br>
            Enter a list of stock ticker symbols (or leave blank for our default universe),
            then click <b style="color:#E0E6F0;">▶ Run</b>. The table will rank every stock
            from highest to lowest score so you can quickly spot the best-looking opportunities.
            <b style="color:#00E676;">Green scores</b> are strong, <b style="color:#FFD600;">yellow</b>
            are mixed, and <b style="color:#FF1744;">red</b> are weak.
        </div>
        """, unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 1.2, 1.2, 0.8])

    with ctrl1:
        custom_input = st.text_input(
            "Tickers",
            placeholder="Custom tickers: AAPL, MSFT, NVDA … (blank = default universe)",
            label_visibility="collapsed",
        )
    with ctrl2:
        sort_col = st.selectbox(
            "Sort",
            ["Final Score", "Value Score", "Growth Score", "Quality Score", "MoS %", "Mkt Cap ($B)"],
            label_visibility="collapsed",
        )
    with ctrl3:
        sig_filter = st.multiselect(
            "Signal",
            ["BUY", "HOLD", "SELL"],
            default=["BUY", "HOLD", "SELL"],
            label_visibility="collapsed",
        )
    with ctrl4:
        run_btn = st.button("▶  Run", type="primary")

    if run_btn:
        if custom_input.strip():
            tickers_to_screen = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        else:
            tickers_to_screen = DEFAULT_TICKERS

        tickers_to_screen = tickers_to_screen[:35]

        prog   = st.progress(0.0)
        status = st.empty()

        def _cb(pct: float, msg: str):
            prog.progress(min(pct, 1.0))
            status.markdown(
                f'<span style="font-size:12px;color:#8892A4;">{msg}</span>',
                unsafe_allow_html=True,
            )

        df_raw = run_screener(tickers_to_screen, progress_callback=_cb)
        prog.empty()
        status.empty()
        st.session_state["screener_df"] = df_raw

    df_raw = st.session_state.get("screener_df", pd.DataFrame())

    if not df_raw.empty:
        df = filter_screener(df_raw, signals=sig_filter)
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
            df.index = df.index + 1

        st.markdown(
            f'<div style="font-size:11.5px;color:#8892A4;margin-bottom:10px;">'
            f'Showing {len(df)} of {len(df_raw)} stocks</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📈  Score vs Market Cap Chart", expanded=False):
            st.plotly_chart(screener_scatter(df), use_container_width=True,
                            config={"displayModeBar": False})

        # Choose columns based on mode
        if beginner:
            display_cols = [
                "Ticker", "Company", "Sector", "Price",
                "Rev Growth %", "Net Margin %",
                "MoS %", "Final Score", "Signal",
            ]
        else:
            display_cols = [
                "Ticker", "Company", "Sector", "Price", "Mkt Cap ($B)",
                "P/E", "Fwd P/E", "EV/EBITDA",
                "Rev Growth %", "Net Margin %", "ROE %",
                "Intrinsic Val", "MoS %",
                "Value Score", "Growth Score", "Quality Score", "Final Score", "Signal",
            ]
        display_cols = [c for c in display_cols if c in df.columns]

        def _color_signal(val):
            return {"BUY":  "color:#00E676;font-weight:700",
                    "SELL": "color:#FF1744;font-weight:700",
                    "HOLD": "color:#FFD600;font-weight:700"}.get(val, "")

        def _color_score(val):
            try:
                v = float(val)
                if v >= 65:   return "color:#00E676"
                elif v >= 40: return "color:#FFD600"
                else:         return "color:#FF1744"
            except Exception:
                return ""

        score_cols = [c for c in ["Value Score", "Growth Score", "Quality Score", "Final Score"]
                      if c in display_cols]

        styled = (
            df[display_cols]
            .style
            .applymap(_color_signal, subset=["Signal"])
            .applymap(_color_score, subset=score_cols)
            .set_properties(**{
                "background-color": "#131722",
                "color": "#E0E6F0",
                "border-color": "#1E2235",
                "font-size": "12px",
            })
            .format(precision=1, na_rep="—")
        )

        st.dataframe(styled, use_container_width=True, height=540)

        buys = df[df["Signal"] == "BUY"].head(6)
        if not buys.empty:
            st.markdown('<div class="section-hdr" style="margin-top:20px;">Top BUY Signals</div>',
                        unsafe_allow_html=True)
            cols = st.columns(len(buys))
            for col, (_, row) in zip(cols, buys.iterrows()):
                s = row.get("Final Score", 0)
                with col:
                    st.markdown(f"""
                    <div class="kpi-card" style="text-align:center;">
                        <div style="font-size:19px;font-weight:700;color:#E0E6F0;">{row['Ticker']}</div>
                        <div style="font-size:10px;color:#8892A4;margin:2px 0;">{str(row.get('Sector',''))[:16]}</div>
                        <div style="font-size:26px;font-weight:700;color:#00E676;margin:6px 0;">{s:.0f}</div>
                        <span class="sig-buy">BUY</span>
                    </div>""", unsafe_allow_html=True)

    elif not df_raw.empty:
        st.warning("No stocks match the current filters.")
    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:40px;color:#8892A4;">
            <div style="font-size:15px;margin-bottom:8px;">No results yet</div>
            <div style="font-size:12px;">Click <b style="color:#E0E6F0;">▶ Run</b> to scan the universe.</div>
        </div>""", unsafe_allow_html=True)

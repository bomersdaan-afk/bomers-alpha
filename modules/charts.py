"""
charts.py — All Plotly charts for the QuantEdge dashboard
"""

from __future__ import annotations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

# ─── Design System ────────────────────────────────────────────────────────────

C = {
    "bg":       "#0A0E1A",
    "surface":  "#131722",
    "surface2": "#1E2235",
    "border":   "#2A3050",
    "text":     "#E0E6F0",
    "dim":      "#8892A4",
    "blue":     "#2962FF",
    "green":    "#00E676",
    "red":      "#FF1744",
    "yellow":   "#FFD600",
    "orange":   "#FF6D00",
    "cyan":     "#00B0FF",
    "purple":   "#AA00FF",
}

_BASE_LAYOUT = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Inter, Arial, sans-serif", size=12),
    margin=dict(l=55, r=25, t=52, b=45),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"], tickcolor=C["dim"], showgrid=True),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"], tickcolor=C["dim"], showgrid=True),
    legend=dict(
        bgcolor=C["surface2"],
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=11),
    ),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["border"], font=dict(color=C["text"])),
)


def _apply_base(fig: go.Figure, **extra) -> go.Figure:
    layout = dict(_BASE_LAYOUT)
    layout.update(extra)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=C["border"], linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border"])
    return fig


# ─── Price History ────────────────────────────────────────────────────────────

def price_history_chart(
    price_history: pd.DataFrame,
    ticker: str,
    intrinsic_value: Optional[float] = None,
    target_price: Optional[float] = None,
) -> go.Figure:
    """Price line + moving averages + volume + IV / target overlays."""

    if price_history is None or price_history.empty:
        fig = go.Figure()
        fig.update_layout(**_BASE_LAYOUT, title=f"{ticker} — No price data", height=480)
        return fig

    close = price_history["Close"].dropna()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.76, 0.24],
    )

    # Price area
    fig.add_trace(
        go.Scatter(
            x=price_history.index, y=close,
            name="Price",
            line=dict(color=C["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(41,98,255,0.07)",
        ),
        row=1, col=1,
    )

    # Moving averages
    if len(close) >= 50:
        ma50 = close.rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=price_history.index, y=ma50, name="MA 50",
                       line=dict(color=C["yellow"], width=1.2, dash="dash"), opacity=0.85),
            row=1, col=1,
        )
    if len(close) >= 200:
        ma200 = close.rolling(200).mean()
        fig.add_trace(
            go.Scatter(x=price_history.index, y=ma200, name="MA 200",
                       line=dict(color=C["orange"], width=1.2, dash="dot"), opacity=0.85),
            row=1, col=1,
        )

    # Intrinsic value line
    if intrinsic_value and intrinsic_value > 0:
        fig.add_hline(
            y=intrinsic_value,
            line=dict(color=C["green"], width=1.8, dash="dash"),
            annotation_text=f"  IV ${intrinsic_value:.2f}",
            annotation_font=dict(color=C["green"], size=11),
            annotation_position="top right",
            row=1, col=1,
        )

    # Analyst target
    if target_price and target_price > 0:
        fig.add_hline(
            y=target_price,
            line=dict(color=C["cyan"], width=1.4, dash="dot"),
            annotation_text=f"  Target ${target_price:.2f}",
            annotation_font=dict(color=C["cyan"], size=11),
            annotation_position="bottom right",
            row=1, col=1,
        )

    # Volume bars
    if "Volume" in price_history.columns:
        vol = price_history["Volume"].fillna(0)
        vol_colors = [
            C["green"] if close.iloc[i] >= close.iloc[i - 1] else C["red"]
            for i in range(len(close))
        ]
        fig.add_trace(
            go.Bar(x=price_history.index, y=vol, name="Volume",
                   marker_color=vol_colors, opacity=0.55, showlegend=False),
            row=2, col=1,
        )

    _apply_base(
        fig,
        title=dict(text=f"<b>{ticker}</b> — Price History", font=dict(size=15, color=C["text"])),
        height=490,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickprefix="$")
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    return fig


# ─── Valuation Comparison Bar Chart ──────────────────────────────────────────

def valuation_comparison_chart(valuation_result: Dict[str, Any]) -> go.Figure:
    """Group-bar: stock multiples vs S&P 500 benchmarks."""
    multiples = valuation_result.get("multiples", {}).get("multiples", {})
    benchmarks = valuation_result.get("multiples", {}).get("benchmarks", {})

    mapping = [
        ("P/E",       "pe_ratio",   "pe"),
        ("Fwd P/E",   "forward_pe", "forward_pe"),
        ("P/B",       "pb_ratio",   "pb"),
        ("P/S",       "ps_ratio",   "ps"),
        ("EV/EBITDA", "ev_ebitda",  "ev_ebitda"),
        ("EV/Rev",    "ev_revenue", "ev_revenue"),
    ]

    labels, stock_vals, bench_vals = [], [], []
    for label, sk, bk in mapping:
        sv = multiples.get(sk)
        bv = benchmarks.get(bk)
        if sv and sv > 0:
            labels.append(label)
            stock_vals.append(round(sv, 2))
            bench_vals.append(round(bv or 0, 2))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Stock", x=labels, y=stock_vals,
        marker_color=C["blue"], opacity=0.9,
        text=[f"{v:.1f}x" for v in stock_vals],
        textposition="outside",
        textfont=dict(size=11, color=C["text"]),
    ))
    fig.add_trace(go.Bar(
        name="S&P 500 Avg", x=labels, y=bench_vals,
        marker_color=C["dim"], opacity=0.6,
        text=[f"{v:.1f}x" for v in bench_vals],
        textposition="outside",
        textfont=dict(size=11, color=C["dim"]),
    ))

    _apply_base(fig,
        title=dict(text="Valuation Multiples vs Market Average", font=dict(size=14)),
        barmode="group", height=360,
        yaxis=dict(title="Multiple (x)", gridcolor=C["border"]),
    )
    return fig


# ─── DCF Waterfall / Bar Chart ────────────────────────────────────────────────

def dcf_waterfall_chart(dcf_result: Dict[str, Any]) -> go.Figure:
    """Forecasted FCF vs present value, year by year."""
    forecasts = dcf_result.get("fcf_forecasts", [])
    if not forecasts:
        return go.Figure()

    years   = [f"Yr {f['year']}" for f in forecasts]
    fcf_b   = [f["fcf"] / 1e9     for f in forecasts]
    pv_b    = [f["pv_fcf"] / 1e9  for f in forecasts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Forecasted FCF", x=years, y=fcf_b,
        marker_color=C["blue"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="PV of FCF", x=years, y=pv_b,
        marker_color=C["cyan"], opacity=0.75,
    ))

    _apply_base(fig,
        title=dict(text="DCF — Forecasted FCF vs Present Value", font=dict(size=14)),
        barmode="group", height=355,
        yaxis=dict(title="$B", ticksuffix="B"),
    )
    return fig


# ─── Revenue & Growth Chart ───────────────────────────────────────────────────

def revenue_growth_chart(growth_data: Dict[str, Any], metrics: Dict[str, Any]) -> go.Figure:
    """Dual-axis: revenue bars (left) + YoY growth line (right)."""
    rev_hist = growth_data.get("revenue_history", [])
    rev_rates = growth_data.get("revenue_growth_rates", [])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if rev_hist:
        n = len(rev_hist)
        year_labels = [f"FY-{n - 1 - i}" if i < n - 1 else "TTM" for i in range(n)]
        fig.add_trace(
            go.Bar(
                x=year_labels, y=[r / 1e9 for r in rev_hist],
                name="Revenue ($B)",
                marker_color=C["blue"], opacity=0.85,
            ),
            secondary_y=False,
        )
        if rev_rates:
            growth_labels = year_labels[1:]
            fig.add_trace(
                go.Scatter(
                    x=growth_labels, y=[r * 100 for r in rev_rates],
                    name="YoY Growth %",
                    line=dict(color=C["green"], width=2.2),
                    mode="lines+markers",
                    marker=dict(size=7, color=C["green"]),
                ),
                secondary_y=True,
            )
    elif metrics.get("revenue_growth"):
        # Fallback: single-bar for YoY
        fig.add_trace(
            go.Bar(x=["YoY"], y=[metrics["revenue_growth"] * 100],
                   name="Revenue Growth %", marker_color=C["blue"]),
            secondary_y=True,
        )

    _apply_base(fig,
        title=dict(text="Revenue History & Growth", font=dict(size=14)),
        height=355,
    )
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=False,
                     gridcolor=C["border"], ticksuffix="B")
    fig.update_yaxes(title_text="Growth %", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", ticksuffix="%")
    return fig


# ─── Margin Trends ────────────────────────────────────────────────────────────

def margin_trends_chart(
    financials: Optional[pd.DataFrame],
    metrics: Dict[str, Any],
) -> go.Figure:
    """Horizontal grouped bars for gross / operating / net margins."""

    labels = ["Gross Margin", "Operating Margin", "Net Margin"]
    colors = [C["green"], C["cyan"], C["blue"]]
    keys   = ["gross_margin", "operating_margin", "profit_margin"]

    values = [(metrics.get(k) or 0) * 100 for k in keys]

    fig = go.Figure()
    for lbl, val, col in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            name=lbl,
            x=[val],
            y=[lbl],
            orientation="h",
            marker_color=col,
            opacity=0.85,
            text=f"{val:.1f}%",
            textposition="outside",
            textfont=dict(size=12, color=C["text"]),
        ))

    _apply_base(fig,
        title=dict(text="Profit Margins (TTM)", font=dict(size=14)),
        barmode="group",
        height=320,
        xaxis=dict(title="%", ticksuffix="%", gridcolor=C["border"]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


# ─── Score Radar ──────────────────────────────────────────────────────────────

def score_radar_chart(scores: Dict[str, float]) -> go.Figure:
    """Spider / radar chart of the four scoring dimensions."""
    cats   = ["Value", "Growth", "Quality", "Macro"]
    vals   = [scores.get("value", 0), scores.get("growth", 0),
              scores.get("quality", 0), scores.get("macro", 0)]
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]

    fig = go.Figure()

    # Background full polygon
    fig.add_trace(go.Scatterpolar(
        r=[100] * 5, theta=cats_c,
        fill="toself",
        fillcolor="rgba(42,48,80,0.4)",
        line=dict(color=C["border"], width=1),
        showlegend=False,
    ))
    # Mid-reference at 50
    fig.add_trace(go.Scatterpolar(
        r=[50] * 5, theta=cats_c,
        fill="none",
        line=dict(color=C["dim"], width=1, dash="dot"),
        showlegend=False,
    ))
    # Score polygon
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c,
        fill="toself",
        fillcolor="rgba(41,98,255,0.22)",
        line=dict(color=C["blue"], width=2.5),
        name="Score",
        showlegend=False,
        mode="lines+markers",
        marker=dict(size=8, color=C["blue"]),
    ))

    fig.update_layout(
        paper_bgcolor=C["bg"],
        font=dict(color=C["text"], family="Inter, Arial, sans-serif"),
        polar=dict(
            bgcolor=C["surface"],
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor=C["border"],
                tickfont=dict(size=9, color=C["dim"]),
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(gridcolor=C["border"], tickfont=dict(size=12, color=C["text"])),
            gridshape="linear",
        ),
        title=dict(text="Score Breakdown", font=dict(size=14, color=C["text"])),
        height=330,
        margin=dict(l=70, r=70, t=60, b=60),
    )
    return fig


# ─── Intrinsic Value Gauge ────────────────────────────────────────────────────

def intrinsic_value_gauge(current_price: float, intrinsic_value: float) -> go.Figure:
    """Gauge indicator showing price vs intrinsic value."""
    if not intrinsic_value or intrinsic_value <= 0:
        return go.Figure()

    mos = (intrinsic_value - current_price) / intrinsic_value * 100
    bar_color = C["green"] if mos >= 15 else (C["yellow"] if mos >= 0 else C["red"])
    iv2 = intrinsic_value * 2

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_price,
        number={"prefix": "$", "font": {"color": C["text"], "size": 28}},
        delta={
            "reference": intrinsic_value,
            "relative": True,
            "valueformat": ".1%",
            "increasing": {"color": C["red"]},
            "decreasing": {"color": C["green"]},
        },
        title={"text": "Current Price vs Intrinsic Value", "font": {"size": 13, "color": C["dim"]}},
        gauge={
            "axis": {
                "range": [0, iv2],
                "tickcolor": C["dim"],
                "tickfont": {"color": C["dim"], "size": 10},
            },
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": C["surface2"],
            "borderwidth": 1,
            "bordercolor": C["border"],
            "steps": [
                {"range": [0,              intrinsic_value * 0.7], "color": "rgba(0,230,118,0.12)"},
                {"range": [intrinsic_value * 0.7, intrinsic_value], "color": "rgba(255,214,0,0.09)"},
                {"range": [intrinsic_value, iv2],                   "color": "rgba(255,23,68,0.10)"},
            ],
            "threshold": {
                "line": {"color": C["green"], "width": 3},
                "thickness": 0.78,
                "value": intrinsic_value,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor=C["bg"],
        font=dict(color=C["text"]),
        height=268,
        margin=dict(l=30, r=30, t=55, b=20),
    )
    return fig


# ─── Screener Scatter ─────────────────────────────────────────────────────────

def screener_scatter(df: "pd.DataFrame") -> go.Figure:
    """Score vs Market Cap scatter for screener results."""
    if df.empty:
        return go.Figure()

    signal_colors = {"BUY": C["green"], "HOLD": C["yellow"], "SELL": C["red"]}
    colors = [signal_colors.get(s, C["dim"]) for s in df.get("Signal", [])]

    fig = go.Figure(go.Scatter(
        x=df.get("Mkt Cap ($B)", []),
        y=df.get("Final Score", []),
        mode="markers+text",
        text=df.get("Ticker", []),
        textposition="top center",
        textfont=dict(size=10, color=C["text"]),
        marker=dict(
            size=11,
            color=colors,
            opacity=0.85,
            line=dict(width=1, color=C["border"]),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Market Cap: $%{x:.1f}B<br>"
            "Score: %{y:.1f}<extra></extra>"
        ),
    ))

    _apply_base(fig,
        title=dict(text="Score vs Market Cap", font=dict(size=14)),
        height=420,
        xaxis=dict(title="Market Cap ($B)", type="log", gridcolor=C["border"]),
        yaxis=dict(title="Final Score", gridcolor=C["border"], range=[0, 100]),
    )
    return fig

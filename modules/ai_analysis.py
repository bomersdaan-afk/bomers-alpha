"""
ai_analysis.py — AI-powered narrative analysis using Claude
Falls back gracefully to a rule-based summary when no API key is supplied.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from modules.utils import safe_float


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(val, pct: bool = False, mult: bool = False, dollar: bool = False) -> str:
    v = safe_float(val)
    if v is None:
        return "N/A"
    if pct:
        return f"{v * 100:+.1f}%"
    if mult:
        return f"{v:.1f}x"
    if dollar:
        return f"${v:,.2f}"
    return str(v)


# ─── AI Analysis (Claude API) ─────────────────────────────────────────────────

def generate_ai_analysis(
    ticker: str,
    metrics: Dict[str, Any],
    valuation_result: Dict[str, Any],
    api_key: Optional[str] = None,
) -> str:
    """
    Generate investment narrative via Claude claude-opus-4-6.
    Falls back to rule-based analysis if no API key is provided.
    """
    if not api_key:
        return _rule_based_analysis(ticker, metrics, valuation_result)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(ticker, metrics, valuation_result)
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception:
        return _rule_based_analysis(ticker, metrics, valuation_result)


def _build_prompt(ticker: str, metrics: Dict[str, Any], val: Dict[str, Any]) -> str:
    scores  = val.get("scores", {})
    dcf     = val.get("dcf", {})
    health  = val.get("health", {})
    mos     = val.get("margin_of_safety")

    iv_str  = _fmt(dcf.get("intrinsic_value"), dollar=True)
    mos_str = f"{mos:+.1f}%" if mos is not None else "N/A"

    return f"""You are a senior equity analyst at a global macro hedge fund. Provide a concise, institutional-quality investment note for {metrics.get('company_name', ticker)} ({ticker}).

COMPANY SNAPSHOT
Sector: {metrics.get('sector', 'N/A')} | Industry: {metrics.get('industry', 'N/A')}
Current Price: {_fmt(metrics.get('current_price'), dollar=True)} | Intrinsic Value (DCF): {iv_str}
Margin of Safety: {mos_str} | Signal: {val.get('signal', 'N/A')} | Score: {scores.get('final', 0):.0f}/100

VALUATION
P/E: {_fmt(metrics.get('pe_ratio'), mult=True)} | Fwd P/E: {_fmt(metrics.get('forward_pe'), mult=True)} | EV/EBITDA: {_fmt(metrics.get('ev_ebitda'), mult=True)}
EV/Revenue: {_fmt(metrics.get('ev_revenue'), mult=True)} | P/B: {_fmt(metrics.get('pb_ratio'), mult=True)} | PEG: {_fmt(metrics.get('peg_ratio'), mult=True)}

GROWTH & PROFITABILITY
Revenue Growth YoY: {_fmt(metrics.get('revenue_growth'), pct=True)}
EPS Growth YoY: {_fmt(metrics.get('earnings_growth'), pct=True)}
Gross Margin: {_fmt(metrics.get('gross_margin'), pct=True)} | Op. Margin: {_fmt(metrics.get('operating_margin'), pct=True)} | Net Margin: {_fmt(metrics.get('profit_margin'), pct=True)}
ROE: {_fmt(metrics.get('roe'), pct=True)} | ROIC: {_fmt(health.get('roic'), pct=True)}

BALANCE SHEET
Debt/EBITDA: {_fmt(health.get('debt_to_ebitda'), mult=True)} | Net Debt/EBITDA: {_fmt(health.get('net_debt_to_ebitda'), mult=True)}

SCORE BREAKDOWN
Value: {scores.get('value', 0):.0f} | Growth: {scores.get('growth', 0):.0f} | Quality: {scores.get('quality', 0):.0f} | Macro: {scores.get('macro', 0):.0f}

Write the investment note using EXACTLY the following structure (use markdown bold for headers):

**Investment Thesis** — 2-3 sentences stating the core bull/bear case.

**Key Growth Drivers**
• [driver 1]
• [driver 2]
• [driver 3]

**Primary Risks**
• [risk 1]
• [risk 2]
• [risk 3]

**Valuation Assessment** — 2 sentences on whether the stock is cheap, fair, or expensive vs fundamentals.

**Recommendation** — One definitive sentence with {val.get('signal', 'HOLD')} stance.

Be direct, data-driven, and precise. No filler language."""


# ─── Rule-Based Fallback ──────────────────────────────────────────────────────

def _rule_based_analysis(
    ticker: str,
    metrics: Dict[str, Any],
    val: Dict[str, Any],
) -> str:
    scores  = val.get("scores", {})
    dcf     = val.get("dcf", {})
    health  = val.get("health", {})
    mos     = val.get("margin_of_safety")
    signal  = val.get("signal", "HOLD")
    reason  = val.get("signal_reason", "")
    company = metrics.get("company_name", ticker)
    final   = scores.get("final", 50)

    # Investment thesis
    if signal == "BUY":
        mos_clause = f" Our DCF model indicates a {mos:.1f}% margin of safety." if mos and mos > 0 else ""
        thesis = (
            f"{company} is rated **{signal}** with a composite score of **{final:.0f}/100**."
            f"{mos_clause} The risk-reward profile is attractive at current levels."
        )
    elif signal == "SELL":
        mos_clause = f" The stock trades at a {-mos:.1f}% premium to intrinsic value." if mos and mos < 0 else ""
        thesis = (
            f"{company} is rated **{signal}** with a composite score of **{final:.0f}/100**."
            f"{mos_clause} Current fundamentals do not justify the market valuation."
        )
    else:
        thesis = (
            f"{company} is rated **{signal}** with a composite score of **{final:.0f}/100**. "
            f"The stock shows mixed signals and is best monitored for a more compelling entry point."
        )

    # Growth drivers
    rev_g   = safe_float(metrics.get("revenue_growth"))
    eps_g   = safe_float(metrics.get("earnings_growth"))
    gm      = safe_float(metrics.get("gross_margin"))
    om      = safe_float(metrics.get("operating_margin"))
    sector  = metrics.get("sector", "its sector")

    drivers = []
    if rev_g and rev_g > 0.05:
        drivers.append(f"Solid revenue momentum with {rev_g*100:.1f}% YoY top-line growth")
    if eps_g and eps_g > 0.05:
        drivers.append(f"Earnings acceleration — EPS growing at {eps_g*100:.1f}% YoY")
    if gm and gm > 0.40:
        drivers.append(f"High-quality business model with {gm*100:.1f}% gross margins")
    if om and om > 0.15:
        drivers.append(f"Operational efficiency evidenced by {om*100:.1f}% operating margin")
    drivers.append(f"Continued market share expansion within {sector}")
    while len(drivers) < 3:
        drivers.append("Management execution on strategic growth initiatives")

    # Risks
    risks = []
    d_ebitda = safe_float(health.get("debt_to_ebitda"))
    pe = safe_float(metrics.get("pe_ratio"))
    if d_ebitda and d_ebitda > 3:
        risks.append(f"Elevated leverage — Debt/EBITDA of {d_ebitda:.1f}x constrains financial flexibility")
    if mos and mos < -15:
        risks.append(f"Valuation risk — stock trades {-mos:.1f}% above our intrinsic value estimate")
    if pe and pe > 30:
        risks.append(f"Premium P/E of {pe:.1f}x is vulnerable to multiple compression")
    risks.append("Macro headwinds — rising rates and slower consumer spending")
    risks.append("Competitive disruption within the industry")
    risks.append("Regulatory / geopolitical risk exposure")

    # Valuation
    iv = safe_float(dcf.get("intrinsic_value"))
    price = safe_float(metrics.get("current_price"))
    ev_ebitda = safe_float(metrics.get("ev_ebitda"))

    if iv and price:
        prem = (price - iv) / iv * 100
        val_sentence = (
            f"Our DCF model yields an intrinsic value of **${iv:.2f}**, implying the stock is "
            f"{'undervalued by' if prem < 0 else 'overvalued by'} **{abs(prem):.1f}%**. "
        )
    else:
        val_sentence = "DCF intrinsic value could not be computed due to insufficient FCF data. "

    if ev_ebitda:
        val_sentence += f"EV/EBITDA of {ev_ebitda:.1f}x {'is reasonable' if ev_ebitda < 15 else 'is elevated'} relative to historical norms."

    # Format
    def bullets(items: list, n: int = 3) -> str:
        return "\n".join(f"• {i}" for i in items[:n])

    return f"""**Investment Thesis**
{thesis}

**Key Growth Drivers**
{bullets(drivers)}

**Primary Risks**
{bullets(risks)}

**Valuation Assessment**
{val_sentence}

**Recommendation**
{signal} — {reason}

---
*Rule-based analysis. Add a Claude API key in the sidebar for AI-powered insights.*"""


# ─── Beginner Explanation ─────────────────────────────────────────────────────

def generate_beginner_explanation(
    ticker: str,
    metrics: Dict[str, Any],
    valuation_result: Dict[str, Any],
    api_key: Optional[str] = None,
) -> str:
    """
    Return a plain-English explanation of the analysis for beginner investors.
    No financial jargon. Uses the Claude API when a key is supplied;
    otherwise falls back to a fully rule-based version.
    """
    if not api_key:
        return _rule_based_beginner_explanation(ticker, metrics, valuation_result)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_beginner_prompt(ticker, metrics, valuation_result)
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception:
        return _rule_based_beginner_explanation(ticker, metrics, valuation_result)


def _build_beginner_prompt(
    ticker: str,
    metrics: Dict[str, Any],
    val: Dict[str, Any],
) -> str:
    """Claude prompt that requests a plain-English beginner explanation."""
    scores  = val.get("scores", {})
    dcf     = val.get("dcf", {})
    health  = val.get("health", {})
    mos     = val.get("margin_of_safety")

    return f"""You are explaining a stock analysis to someone who has never invested before.
Use simple, everyday language. Avoid financial jargon. If you must use a technical term,
explain it immediately in plain words.

Here is the data for {metrics.get('company_name', ticker)} ({ticker}):

- Current share price: {_fmt(metrics.get('current_price'), dollar=True)}
- Our estimated fair value: {_fmt(dcf.get('intrinsic_value'), dollar=True)}
- Margin of safety: {f"{mos:+.1f}%" if mos is not None else "N/A"} (positive = stock looks cheap, negative = looks expensive)
- Overall score: {scores.get('final', 0):.0f} out of 100
- Signal: {val.get('signal', 'HOLD')}
- Value score: {scores.get('value', 0):.0f}/100  (is the price reasonable?)
- Growth score: {scores.get('growth', 0):.0f}/100  (is the business growing?)
- Quality score: {scores.get('quality', 0):.0f}/100  (is it a profitable, financially healthy business?)
- Macro score: {scores.get('macro', 0):.0f}/100  (what does the broader market think?)
- Revenue growth: {_fmt(metrics.get('revenue_growth'), pct=True)}
- Net profit margin: {_fmt(metrics.get('profit_margin'), pct=True)}
- Debt level: {_fmt(health.get('debt_to_ebitda'), mult=True)} debt-to-earnings ratio
- Sector: {metrics.get('sector', 'N/A')}

Write a friendly, jargon-free explanation using EXACTLY this structure:

**Is this stock cheap or expensive right now?**
One or two sentences. Compare price to fair value in plain English. Use an analogy if helpful (e.g. "like finding a jacket on sale").

**What does the {scores.get('final', 0):.0f}/100 score mean?**
Two to three sentences. Explain the score in everyday terms. Mention which dimension is strongest and which is weakest.

**What's going well**
• [strength 1 — plain English, no jargon]
• [strength 2]
• [strength 3]

**What to watch out for**
• [risk 1 — plain English, explain why it matters to a beginner]
• [risk 2]

**Bottom line**
One clear sentence summarising whether a beginner investor should be interested,
cautious, or should avoid — based on the {val.get('signal', 'HOLD')} signal.
End with a reminder that this is a model output, not personal financial advice.

Tone: clear, honest, encouraging. Maximum reading age: 16. No bullet points outside the designated sections."""


def _rule_based_beginner_explanation(
    ticker: str,
    metrics: Dict[str, Any],
    val: Dict[str, Any],
) -> str:
    """
    Fully rule-based plain-English explanation — no API key required.
    Covers: price vs fair value, score meaning, strengths, risks, bottom line.
    """
    scores  = val.get("scores", {})
    dcf     = val.get("dcf", {})
    health  = val.get("health", {})
    mos     = val.get("margin_of_safety")
    signal  = val.get("signal", "HOLD")
    company = metrics.get("company_name", ticker)
    final   = scores.get("final", 50)

    price = safe_float(metrics.get("current_price"))
    iv    = safe_float(dcf.get("intrinsic_value"))

    # ── Section 1: price vs fair value ────────────────────────────────────────
    if iv and price:
        diff_pct = abs((iv - price) / iv * 100)
        if (mos or 0) >= 20:
            price_section = (
                f"Our model estimates the fair value of {company} at **${iv:.2f}** per share, "
                f"while the stock currently trades at **${price:.2f}**. "
                f"That means you could be buying it at roughly a **{diff_pct:.0f}% discount** to what we think it's worth — "
                f"a bit like finding a brand-name item on sale. The bigger the discount, "
                f"the more cushion you have if something goes wrong."
            )
        elif (mos or 0) >= 0:
            price_section = (
                f"Our model estimates the fair value of {company} at **${iv:.2f}** per share, "
                f"and the stock trades close to that at **${price:.2f}**. "
                f"There is a small discount of about **{diff_pct:.0f}%**, "
                f"which provides a modest buffer but not a large one. "
                f"The stock appears reasonably priced, not a bargain but not overpriced either."
            )
        else:
            price_section = (
                f"Our model estimates the fair value of {company} at **${iv:.2f}** per share, "
                f"but the stock is currently trading higher at **${price:.2f}** — "
                f"about **{diff_pct:.0f}% above** our estimate. "
                f"This means you may be paying a premium. "
                f"Think of it like buying something at full price when the sale just ended."
            )
    else:
        price_section = (
            f"{company} is currently priced at **${price:.2f}** per share. "
            f"We were unable to calculate a precise fair value estimate because the company "
            f"does not have enough free cash flow history for our model — "
            f"this is common for early-stage or financial-sector companies."
        )

    # ── Section 2: what the score means ───────────────────────────────────────
    v_score = scores.get("value",   50)
    g_score = scores.get("growth",  50)
    q_score = scores.get("quality", 50)
    m_score = scores.get("macro",   50)

    dim_scores = {"Value": v_score, "Growth": g_score, "Quality": q_score, "Macro": m_score}
    best_dim  = max(dim_scores, key=dim_scores.get)
    worst_dim = min(dim_scores, key=dim_scores.get)

    _dim_plain = {
        "Value":   "whether the price looks reasonable",
        "Growth":  "how fast the business is growing",
        "Quality": "how profitable and financially healthy the company is",
        "Macro":   "how the broader market and analysts view the stock",
    }

    if final >= 65:
        score_verdict = "a strong result"
        score_label   = "(strong)"
    elif final >= 40:
        score_verdict = "a mixed result"
        score_label   = "(mixed)"
    else:
        score_verdict = "a weak result"
        score_label   = "(weak)"

    score_section = (
        f"A score of **{final:.0f} out of 100** is {score_verdict} {score_label}. "
        f"Think of it like a report card — the higher the number, the better the stock "
        f"looks across all the things we measure. "
        f"The strongest area is **{best_dim}** ({dim_scores[best_dim]:.0f}/100), meaning "
        f"{_dim_plain[best_dim]} looks particularly good. "
        f"The weakest area is **{worst_dim}** ({dim_scores[worst_dim]:.0f}/100), meaning "
        f"{_dim_plain[worst_dim]} is less impressive and worth keeping an eye on."
    )

    # ── Section 3: strengths ──────────────────────────────────────────────────
    strengths = []

    rev_g = safe_float(metrics.get("revenue_growth"))
    if rev_g and rev_g > 0.08:
        strengths.append(
            f"The company's sales are growing at **{rev_g*100:.0f}% per year** — "
            f"that's a healthy pace, meaning more people are buying its products or services."
        )
    elif rev_g and rev_g > 0:
        strengths.append(
            f"Sales are growing at **{rev_g*100:.0f}% per year** — modest but positive momentum."
        )

    nm = safe_float(metrics.get("profit_margin"))
    if nm and nm > 0.15:
        strengths.append(
            f"For every £/$ of sales, the company keeps **{nm*100:.0f} cents as profit** — "
            f"that's a high margin, suggesting pricing power and an efficient business."
        )
    elif nm and nm > 0.05:
        strengths.append(
            f"The company keeps **{nm*100:.0f} cents of profit per dollar of sales**, "
            f"which is a decent margin for its industry."
        )

    roe = safe_float(metrics.get("roe"))
    if roe and roe > 0.15:
        strengths.append(
            f"The business earns a strong **{roe*100:.0f}% return on shareholders' money** — "
            f"meaning management is putting capital to work effectively."
        )

    d_ebitda = safe_float(health.get("debt_to_ebitda"))
    if d_ebitda is not None and d_ebitda < 1.5:
        strengths.append(
            f"The company carries very little debt relative to its earnings — "
            f"a healthy balance sheet that reduces financial risk."
        )

    # Always have at least 2 strengths
    if len(strengths) < 2:
        strengths.append(
            f"{company} operates in the **{metrics.get('sector', 'broader market')}** sector "
            f"and has an established market presence."
        )
    if len(strengths) < 2:
        strengths.append("The business generates revenue and has analyst coverage.")

    # ── Section 4: risks ──────────────────────────────────────────────────────
    risks = []

    if d_ebitda and d_ebitda > 3:
        risks.append(
            f"The company carries a significant debt load (about **{d_ebitda:.1f}×** its annual earnings). "
            f"High debt can be risky if business conditions worsen, because interest payments still have to be made."
        )

    if (mos or 0) < -15:
        risks.append(
            f"The stock is trading above our fair value estimate. "
            f"This means you could be paying too much — if the company disappoints, "
            f"the share price could fall back toward our estimated fair value."
        )

    pe = safe_float(metrics.get("pe_ratio"))
    if pe and pe > 35:
        risks.append(
            f"Investors are paying a high price relative to earnings (P/E of {pe:.0f}×). "
            f"Stocks with high valuations can drop sharply if growth slows down."
        )

    if rev_g is not None and rev_g < 0:
        risks.append(
            f"Revenue is actually shrinking ({rev_g*100:.1f}% year-on-year). "
            f"Declining sales can be an early warning sign that a business is losing ground."
        )

    # Always include a general market risk
    risks.append(
        "All stocks carry general market risk — prices can fall due to economic downturns, "
        "interest rate changes, or broader investor sentiment, regardless of company performance."
    )

    # ── Section 5: bottom line ────────────────────────────────────────────────
    _signal_plain = {
        "BUY":  (
            f"Based on this analysis, {company} looks like a potentially interesting opportunity "
            f"for investors willing to accept normal stock market risk. "
            f"The model gives it a **BUY** signal — but always invest only what you can afford to lose "
            f"and consider speaking to a financial adviser before making any decisions."
        ),
        "HOLD": (
            f"Based on this analysis, {company} looks fairly valued right now — "
            f"neither an obvious bargain nor clearly overpriced. "
            f"The model gives it a **HOLD** signal, which means there is no urgent reason to buy or sell. "
            f"It may be worth watching for a better entry price."
        ),
        "SELL": (
            f"Based on this analysis, the model gives {company} a **SELL** signal, "
            f"suggesting the stock may be overpriced or that the business has concerning weaknesses. "
            f"This does not mean the company is going bankrupt — it simply means the current price "
            f"does not offer an attractive risk-reward balance according to our model."
        ),
    }
    bottom_line = _signal_plain.get(signal, _signal_plain["HOLD"])

    # ── Assemble ──────────────────────────────────────────────────────────────
    def _bullets(items, n=3):
        return "\n".join(f"• {item}" for item in items[:n])

    return f"""**Is this stock cheap or expensive right now?**
{price_section}

**What does the {final:.0f}/100 score mean?**
{score_section}

**What's going well**
{_bullets(strengths)}

**What to watch out for**
{_bullets(risks, n=3)}

**Bottom line**
{bottom_line}

---
*This explanation was generated automatically. It is not personal financial advice.*"""

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(page_title="FinBERT Narrative Options Matrix", layout="wide")
st.title("\U0001f6e1\ufe0f Institutional Confluence & Narrative Options Engine")
st.write("FinBERT-powered financial sentiment \u2022 Live Finnhub news & social feeds \u2022 Multi-pillar technical confluence")

# ─────────────────────────────────────────────────────────────
# FINBERT MODEL
# ProsusAI/finbert is fine-tuned on 10-K filings, earnings calls,
# and financial news.  Score = P(positive) - P(negative) ∈ [-1, +1]
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model (first run only ~450 MB)...")
def load_finbert():
    tok = BertTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    mdl.eval()
    return tok, mdl

tokenizer, finbert_model = load_finbert()

def finbert_score(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = finbert_model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze()
        return round(float(probs[0]) - float(probs[1]), 4)
    except Exception:
        return 0.0

def batch_finbert_score(texts: list) -> float:
    scores = [finbert_score(t) for t in texts if t and t.strip()]
    return round(sum(scores) / len(scores), 4) if scores else 0.0

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for key, default in [("cached_watchlist", []), ("headline_map", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────
# TICKER VALIDATION
# Strategy: trust Finnhub's own symbol field first (already vetted).
# For regex candidates only, apply a fast blocklist filter then
# a single yfinance price check.  Module-level dict avoids
# repeated network calls for the same ticker.
# ─────────────────────────────────────────────────────────────
FORBIDDEN = {
    "NATO","UAE","WTI","LNG","USD","EUR","FED","SEC","CEO","USA","GDP",
    "CPI","THE","IPO","ETF","SPAC","CNBC","OPEC","FDIC","FOMC","WHO",
    "IMF","ESG","DOJ","IRS","CBO","NFT","EPS","FDA","CDC","EPA","NYSE",
    "AMEX","CBOE","CME","CFTC","EV","AI","ML","ICE","PE","UK","US","EU",
    "UN","PM","VP","SP","QE","VC","IV","FX","IR","IT","HR","PR","LLC",
    "INC","LTD","PLC","AGM","EOD","EOW","YOY","QOQ","MOM","APR","JAN",
    "FEB","MAR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC",
}

_ticker_cache: dict = {}

def quick_validate(ticker: str) -> bool:
    """Blocklist check only — no network call.  Used for regex candidates."""
    return (
        ticker not in FORBIDDEN
        and ticker.isalpha()
        and 2 <= len(ticker) <= 5
    )

def full_validate(ticker: str) -> bool:
    """Blocklist + live yfinance price check.  Cached per session."""
    if ticker in _ticker_cache:
        return _ticker_cache[ticker]
    if not quick_validate(ticker):
        _ticker_cache[ticker] = False
        return False
    try:
        price = yf.Ticker(ticker).fast_info.get("last_price", None)
        valid = bool(price and float(price) > 1.0)
    except Exception:
        valid = False
    _ticker_cache[ticker] = valid
    return valid

# ─────────────────────────────────────────────────────────────
# LIVE NARRATIVE SCRAPER
# Priority 1: use Finnhub's own 'related' ticker list per article
#   — these are pre-vetted equity symbols, no yfinance call needed.
# Priority 2: company-news endpoint per market mover symbol.
# Priority 3: regex on headline text (slow, last resort).
# Falls back to FALLBACK_TICKERS if < 5 valid tickers found.
# ─────────────────────────────────────────────────────────────
FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","TSLA","AMZN",
    "GOOGL","META","AMD","JPM","BAC",
    "XOM","UNH","JNJ","V","MA",
]

def fetch_and_expand_narrative():
    extracted: dict = {}   # ticker -> headline
    TARGET = 20
    api_key = st.secrets.get("FINNHUB_API_KEY", "")

    if api_key:
        # ── Approach A: market status endpoint gives top movers ──
        # Actually use Finnhub general news — but extract 'related'
        # tickers list which Finnhub attaches to many articles.
        for cat in ["general", "merger"]:
            if len(extracted) >= TARGET:
                break
            try:
                url = f"https://finnhub.io/api/v1/news?category={cat}&token={api_key}"
                items = requests.get(url, timeout=12).json()
                if not isinstance(items, list):
                    continue
            except Exception:
                continue

            for item in items:
                if len(extracted) >= TARGET:
                    break
                headline = item.get("headline", "")

                # ── Priority 1: Finnhub's 'related' field (list of tickers) ──
                related = item.get("related", "") or ""
                for sym in re.split(r"[,\s]+", related):
                    sym = sym.upper().strip().split(".")[0]
                    if sym and sym not in extracted and quick_validate(sym):
                        extracted[sym] = headline
                    if len(extracted) >= TARGET:
                        break

                # ── Priority 2: symbol field ──────────────────────────────
                raw_sym = item.get("symbol", "")
                if raw_sym:
                    sym = raw_sym.upper().split(".")[0].strip()
                    if sym and sym not in extracted and quick_validate(sym):
                        extracted[sym] = headline

                # ── Priority 3: regex on text (only if still short) ───────
                if len(extracted) < TARGET // 2:
                    candidates = re.findall(r"\b[A-Z]{2,5}\b",
                                            f"{headline} {item.get('summary','')}")
                    for sym in candidates[:20]:   # cap to avoid slow loops
                        if sym not in extracted and full_validate(sym):
                            extracted[sym] = headline
                        if len(extracted) >= TARGET:
                            break

        # ── Approach B: Finnhub stock symbols endpoint for popular names ──
        if len(extracted) < TARGET // 2:
            try:
                url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key}"
                syms = requests.get(url, timeout=12).json()
                # Pick first N symbols with displaySymbol <= 5 chars
                count = 0
                for s in syms:
                    ds = s.get("displaySymbol", "")
                    if ds and quick_validate(ds) and ds not in extracted:
                        extracted[ds] = "High-volume US equity from Finnhub symbol list."
                        count += 1
                    if len(extracted) >= TARGET or count >= 20:
                        break
            except Exception:
                pass

    used_fallback = len(extracted) < 5
    if used_fallback:
        for t in FALLBACK_TICKERS:
            if t not in extracted:
                extracted[t] = "Liquid large-cap \u2014 fallback watchlist (check FINNHUB_API_KEY)."
            if len(extracted) >= TARGET:
                break

    final = list(extracted.keys())[:TARGET]
    st.session_state.cached_watchlist = final
    st.session_state.headline_map = {t: extracted[t] for t in final}
    return used_fallback

# ─────────────────────────────────────────────────────────────
# SENTIMENT ENGINE
# ─────────────────────────────────────────────────────────────
def get_cross_channel_sentiment(sym: str):
    """Returns (blended, news_score, social_score, headlines_used)"""
    api_key = st.secrets.get("FINNHUB_API_KEY", "")
    sym = sym.strip().upper()
    headlines_used = []

    # ── FinBERT on corporate news headlines ──────────────────
    news_score = 0.0
    if api_key:
        try:
            today = datetime.today().strftime("%Y-%m-%d")
            past  = (datetime.today() - timedelta(days=20)).strftime("%Y-%m-%d")
            url   = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={past}&to={today}&token={api_key}"
            res   = requests.get(url, timeout=10).json()
            if isinstance(res, list) and res:
                texts = []
                for item in res[:15]:
                    h = item.get("headline", "")
                    s = item.get("summary",  "")
                    combined = f"{h}. {s}".strip(". ")
                    if combined:
                        texts.append(combined)
                        headlines_used.append(h)
                news_score = batch_finbert_score(texts)
        except Exception:
            pass

    # ── Social sentiment (bullishPercent - bearishPercent) ───
    social_score = 0.0
    if api_key:
        try:
            url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={sym}&token={api_key}"
            data = requests.get(url, timeout=10).json()
            scores = []
            for platform in ("reddit", "twitter"):
                for entry in data.get(platform, [])[:5]:
                    bull = entry.get("bullishPercent")
                    bear = entry.get("bearishPercent")
                    if bull is not None and bear is not None:
                        scores.append((bull - bear) / 100.0)
                    else:
                        pos   = entry.get("positiveCount", 0)
                        neg   = entry.get("negativeCount", 0)
                        total = pos + neg
                        if total > 0:
                            scores.append((pos - neg) / total)
            if scores:
                social_score = round(max(min(sum(scores)/len(scores), 1.0), -1.0), 4)
        except Exception:
            pass

    blended = round((news_score * 0.60) + (social_score * 0.40), 4)
    return blended, news_score, social_score, headlines_used

# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def get_confluence_data(ticker: str):
    try:
        df = yf.download(ticker.strip(), period="1y", interval="1d",
                         group_by="column", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()

        today_c = float(close.iloc[-1])
        prev_c  = float(close.iloc[-2])
        c10     = float(close.iloc[-10])

        trend_10d   = round(((today_c - c10) / c10) * 100, 2)
        is_up_today = today_c > prev_c

        ma200 = close.rolling(200).mean()
        above_200 = (today_c > float(ma200.iloc[-1])
                     if len(close) >= 200 and not pd.isna(ma200.iloc[-1])
                     else None)

        ma50 = close.rolling(50).mean()
        above_50 = (today_c > float(ma50.iloc[-1])
                    if len(close) >= 50 and not pd.isna(ma50.iloc[-1])
                    else None)

        avg_vol = float(volume.iloc[-11:-1].mean())
        rvol    = round(float(volume.iloc[-1]) / (avg_vol + 1e-9), 2)
        rsi     = round(float(calculate_rsi(close, 14).iloc[-1]), 2)
        atr     = round(float((high - low).rolling(14).mean().iloc[-1]), 2)

        return {
            "price": today_c, "trend_10d": trend_10d,
            "is_up_today": is_up_today, "above_200": above_200,
            "above_50": above_50, "rvol": rvol, "rsi": rsi, "atr": atr,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# CONFLUENCE SCORING (6 criteria)
# ─────────────────────────────────────────────────────────────
def score_confluence(tech: dict, sentiment: float):
    checks = {
        "Positive FinBERT Sentiment":  sentiment >= 0.10,
        "10d Price Uptrend":           tech["trend_10d"] > 0,
        "Up Today":                    tech["is_up_today"],
        "Above 200-day MA":            bool(tech["above_200"]) if tech["above_200"] is not None else False,
        "Above 50-day MA":             bool(tech["above_50"])  if tech["above_50"]  is not None else False,
        "Volume Spike (RVOL ≥ 1.5)":  tech["rvol"] >= 1.5,
    }
    return sum(checks.values()), checks

def recommend_strategy(score: int, rsi: float, sentiment: float) -> str:
    if score >= 5:
        return ("\U0001f7e1 CALL HOLD \u2014 Overbought RSI>75, wait for pullback."
                if rsi > 75 else
                "\U0001f7e2 BUY LONG CALLS \u2014 Strong FinBERT + Technical Confluence.")
    if score <= 1 and sentiment <= -0.10:
        return ("\U0001f7e1 PUT HOLD \u2014 Oversold RSI<25, wait for bounce."
                if rsi < 25 else
                "\U0001f534 BUY LONG PUTS \u2014 High Confluence Breakdown Signal.")
    if score == 4:
        return "\U0001f535 MODERATE BULLISH \u2014 Missing 1-2 confirmations."
    if score == 2:
        return "\U0001f7e0 MODERATE BEARISH \u2014 Limited bullish confirmation."
    return "\U0001f6ab NO SETUP \u2014 Conflicting signals, avoid options here."

# ─────────────────────────────────────────────────────────────
# STYLER HELPERS  (pandas 2.1+ uses .map not .applymap)
# ─────────────────────────────────────────────────────────────
def _style_strategy(val):
    v = str(val)
    if "CALLS" in v and "HOLD" not in v:
        return "background-color:#c8f7c5;color:#1a7a1a;font-weight:bold"
    if "PUTS" in v and "HOLD" not in v:
        return "background-color:#f7c5c5;color:#7a1a1a;font-weight:bold"
    if "HOLD" in v:
        return "background-color:#fff4cc;color:#7a6000"
    if "MODERATE" in v:
        return "background-color:#d0e8ff;color:#003a7a"
    return "color:#888"

def _style_score(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v >= 5:   return "background-color:#c8f7c5;font-weight:bold"
    if v >= 4:   return "background-color:#d0e8ff"
    if v <= 1:   return "background-color:#f7c5c5"
    return ""

def _style_sentiment(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v >= 0.10:  return "color:#1a7a1a;font-weight:bold"
    if v <= -0.10: return "color:#7a1a1a;font-weight:bold"
    return "color:#888"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.header("\u2699\ufe0f Target Matrix Controls")
scan_mode = st.sidebar.radio(
    "Discovery Feed:",
    ("\U0001f4f0 Live Dynamic News & Social Stream", "\u270d\ufe0f Manual Ticker Entry")
)

if scan_mode == "\u270d\ufe0f Manual Ticker Entry":
    user_input = st.sidebar.text_input(
        "Ticker Symbols (comma separated):", value="AAPL, NVDA, TSLA, MSFT, AMZN"
    )
    active_watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    if not st.session_state.cached_watchlist:
        with st.spinner("Scanning Finnhub narrative wires..."):
            used_fb = fetch_and_expand_narrative()
        if used_fb:
            st.sidebar.warning(
                "\u26a0\ufe0f Finnhub returned few/no tickers.\n\n"
                "Check **FINNHUB_API_KEY** is set in Streamlit Secrets.\n\n"
                "Showing fallback liquid large-caps."
            )

    if st.sidebar.button("\U0001f504 Refresh \u2014 Scrape Fresh Narrative Wires"):
        _ticker_cache.clear()
        st.session_state.cached_watchlist = []
        st.session_state.headline_map = {}
        with st.spinner("Reloading Finnhub news cycle..."):
            used_fb = fetch_and_expand_narrative()
        if used_fb:
            st.sidebar.warning("\u26a0\ufe0f Finnhub returned few tickers \u2014 using fallback watchlist.")
        st.rerun()

    active_watchlist = st.session_state.cached_watchlist

if active_watchlist:
    st.sidebar.success(f"\u2705 {len(active_watchlist)} tickers loaded")
    st.sidebar.caption(", ".join(active_watchlist))

with st.sidebar.expander("\u2139\ufe0f About FinBERT"):
    st.write(
        "**FinBERT** (ProsusAI/finbert) is BERT fine-tuned on 10-K filings, "
        "earnings calls and financial news.\n\n"
        "Score = P(positive) \u2212 P(negative) \u2192 range \u22121.0 to +1.0.\n\n"
        "Replaces VADER, which has no financial domain knowledge."
    )

run_scan = st.sidebar.button("\U0001f6e1\ufe0f Run Scan Matrix", type="primary")

# ─────────────────────────────────────────────────────────────
# MAIN SCAN PIPELINE
# ─────────────────────────────────────────────────────────────
if run_scan:
    if not active_watchlist:
        st.warning("No tickers loaded. Add tickers manually or run the live scraper first.")
        st.stop()

    results = []
    bar = st.progress(0, text="Initialising FinBERT pipeline...")
    total = len(active_watchlist)

    for i, ticker in enumerate(active_watchlist):
        bar.progress((i + 1) / total, text=f"Analysing {ticker} ({i+1}/{total})...")
        tech = get_confluence_data(ticker)
        if tech is None:
            continue
        blended, news_sc, social_sc, headlines = get_cross_channel_sentiment(ticker)
        score, checks = score_confluence(tech, blended)
        strategy = recommend_strategy(score, tech["rsi"], blended)

        results.append({
            "Ticker":             ticker,
            "Price ($)":          round(tech["price"], 2),
            "Confluence (/6)":    score,
            "Options Strategy":   strategy,
            "FinBERT Blended":    blended,
            "FinBERT News":       round(news_sc, 4),
            "Social Score":       round(social_sc, 4),
            "RSI (14)":           tech["rsi"],
            "RVOL":               tech["rvol"],
            "ATR (14d)":          tech["atr"],
            "10d Trend (%)":      tech["trend_10d"],
            "Above 200MA":        "\u2705" if tech["above_200"] else ("\u2796" if tech["above_200"] is None else "\u274c"),
            "Above 50MA":         "\u2705" if tech["above_50"]  else ("\u2796" if tech["above_50"]  is None else "\u274c"),
            "Up Today":           "\u2705" if tech["is_up_today"] else "\u274c",
            "Narrative Catalyst": st.session_state.headline_map.get(ticker, "Active float/volume tracking."),
            "_checks":            checks,
            "_headlines":         headlines[:3],
        })

    bar.empty()

    if not results:
        st.error("No valid tickers matched. Try adding tickers manually via the sidebar.")
        st.stop()

    df = pd.DataFrame(results)
    display_cols = [c for c in df.columns if not c.startswith("_")]

    # ── Results Table ──────────────────────────────────────────
    st.write(f"### \U0001f3af FinBERT Confluence Matrix \u2014 {len(df)} Stocks")
    styled = (
        df[display_cols]
        .style
        .map(_style_strategy,  subset=["Options Strategy"])
        .map(_style_score,     subset=["Confluence (/6)"])
        .map(_style_sentiment, subset=["FinBERT Blended", "FinBERT News", "Social Score"])
    )
    st.dataframe(styled, use_container_width=True)

    # ── Scatter plot ───────────────────────────────────────────
    st.write("### \U0001f5fa\ufe0f FinBERT Sentiment vs Volume Spike Cluster Map")
    fig = px.scatter(
        df, x="FinBERT Blended", y="RVOL", text="Ticker",
        color="Options Strategy", size="RSI (14)",
        hover_data=["Price ($)", "10d Trend (%)", "ATR (14d)", "Confluence (/6)"],
        range_x=[-1, 1],
        title="Options Signal Clusters (FinBERT News + Social Sentiment vs RVOL)",
        labels={"FinBERT Blended": "FinBERT Score (News 60% + Social 40%)",
                "RVOL": "Relative Volume (RVOL)"},
    )
    fig.update_traces(textposition="top center",
                      textfont=dict(size=12, family="Arial Black"))
    fig.add_hline(y=1.5,   line_dash="dash", line_color="black",
                  annotation_text="Institutional Volume Baseline (1.5x)")
    fig.add_vline(x=0,     line_dash="dash", line_color="gray",
                  annotation_text="Neutral")
    fig.add_vline(x=0.10,  line_dash="dot",  line_color="green",
                  annotation_text="Bullish Threshold")
    fig.add_vline(x=-0.10, line_dash="dot",  line_color="red",
                  annotation_text="Bearish Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # ── Confluence heatmap ─────────────────────────────────────
    st.write("### \U0001f52c Confluence Criteria Breakdown")
    checks_df = pd.DataFrame(
        {row["Ticker"]: row["_checks"] for row in results}
    ).T.astype(int)
    fig2 = px.imshow(
        checks_df,
        color_continuous_scale=[[0, "#f7c5c5"], [1, "#c8f7c5"]],
        aspect="auto",
        title="Per-Ticker Criteria (Green = Met, Red = Not Met)",
        text_auto=True,
    )
    fig2.update_xaxes(tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

    # ── FinBERT headline detail ────────────────────────────────
    st.write("### \U0001f4f0 FinBERT Input Headlines (sample per ticker)")
    for row in results:
        if row["_headlines"]:
            with st.expander(f"{row['Ticker']} \u2014 Blended: {row['FinBERT Blended']:+.3f}"):
                for h in row["_headlines"]:
                    s = finbert_score(h)
                    icon = "\U0001f7e2" if s > 0.10 else ("\U0001f534" if s < -0.10 else "\u26aa")
                    st.write(f"{icon} `{s:+.3f}` \u2014 {h}")

    st.info(
        "\u26a0\ufe0f **Disclaimer:** For research and educational purposes only. "
        "FinBERT scores are not financial advice. Options trading involves "
        "significant risk of loss. Always conduct your own due diligence."
    )

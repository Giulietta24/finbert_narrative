import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# FinBERT via HuggingFace transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(page_title="FinBERT Narrative Options Matrix", layout="wide")

st.title("🛡️ Institutional Confluence & Narrative Options Engine")
st.write("FinBERT-powered financial sentiment • Live Finnhub news & social feeds • Multi-pillar technical confluence")

# ─────────────────────────────────────────────────────────────
# FINBERT MODEL LOADER
# Uses ProsusAI/finbert — trained specifically on financial text
# (10-K filings, earnings calls, financial news)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model (first run only)...")
def load_finbert():
    """
    ProsusAI/finbert outputs 3 classes: positive, negative, neutral.
    We convert to a [-1, +1] score: positive_prob - negative_prob
    """
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

tokenizer, finbert_model = load_finbert()

def finbert_score(text: str) -> float:
    """
    Returns a single float in [-1.0, +1.0]:
      +1.0 = strongly positive financial sentiment
      -1.0 = strongly negative financial sentiment
       0.0 = neutral
    FinBERT label order: positive=0, negative=1, neutral=2
    """
    if not text or not text.strip():
        return 0.0
    try:
        # Truncate to 512 tokens (BERT hard limit)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            logits = finbert_model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze()
        # score = P(positive) - P(negative)
        score = float(probs[0]) - float(probs[1])
        return round(score, 4)
    except Exception:
        return 0.0

def batch_finbert_score(texts: list[str]) -> float:
    """Average FinBERT score across a list of headlines/summaries."""
    scores = [finbert_score(t) for t in texts if t and t.strip()]
    return round(sum(scores) / len(scores), 4) if scores else 0.0

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────
if "cached_watchlist" not in st.session_state:
    st.session_state.cached_watchlist = []
if "headline_map" not in st.session_state:
    st.session_state.headline_map = {}

# ─────────────────────────────────────────────────────────────
# TICKER VALIDATION
# ─────────────────────────────────────────────────────────────
# Expanded forbidden-word list — catches common acronyms that slip
# through regex and waste yfinance calls
FORBIDDEN = {
    "NATO", "UAE", "WTI", "LNG", "USD", "EUR", "FED", "SEC", "CEO",
    "USA", "UK", "GDP", "CPI", "THE", "IPO", "ETF", "SPAC", "CNBC",
    "OPEC", "FDIC", "FOMC", "WHO", "IMF", "ESG", "DOJ", "IRS", "CBO",
    "NFT", "DeFi", "DEFI", "EPS", "PE", "AI", "ML", "EV", "ICE",
    "FDA", "CDC", "EPA", "NYSE", "AMEX", "CBOE", "CME", "CFTC",
}

@st.cache_data(ttl=300, show_spinner=False)
def is_real_optionable_stock(ticker: str) -> bool:
    """
    Two-stage filter:
    1. Rejects known non-equity acronyms.
    2. Confirms yfinance can return a live price > $1.
    Cached for 5 minutes to avoid hammering yfinance on repeat calls.
    """
    if (
        ticker in FORBIDDEN
        or not ticker.isalpha()
        or len(ticker) < 2
        or len(ticker) > 5          # Allow 5-char tickers (e.g. GOOGL, AMZN)
    ):
        return False
    try:
        info = yf.Ticker(ticker).fast_info
        price = info.get("last_price", None)
        return bool(price and price > 1.0)
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# LIVE NARRATIVE SCRAPER (Finnhub → ticker discovery)
# ─────────────────────────────────────────────────────────────
def fetch_and_expand_narrative():
    """
    Pulls live Finnhub news across multiple categories.
    Prefers explicit symbol tags in the JSON; falls back to regex
    only when needed. Stops at 25 validated tickers.
    """
    try:
        if "FINNHUB_API_KEY" not in st.secrets:
            st.error("❌ Missing 'FINNHUB_API_KEY' in Streamlit Secrets.")
            return

        api_key = st.secrets["FINNHUB_API_KEY"]
        extracted_tickers: set[str] = set()
        temp_headline_map: dict[str, str] = {}
        TARGET = 25

        categories = ["general", "merger", "forex", "crypto"]

        for cat in categories:
            if len(extracted_tickers) >= TARGET:
                break
            try:
                url = f"https://finnhub.io/api/v1/news?category={cat}&token={api_key}"
                news_items = requests.get(url, timeout=10).json()
            except Exception:
                continue

            if not isinstance(news_items, list):
                continue

            for item in news_items:
                if len(extracted_tickers) >= TARGET:
                    break

                headline = item.get("headline", "")
                summary  = item.get("summary", "")

                # Priority 1: use the symbol field in the JSON (more reliable than regex)
                raw_sym = item.get("symbol", "")
                if raw_sym:
                    ticker = raw_sym.upper().split(".")[0].strip()
                    if ticker not in extracted_tickers and is_real_optionable_stock(ticker):
                        extracted_tickers.add(ticker)
                        temp_headline_map.setdefault(ticker, headline)

                # Priority 2: regex scan — only if still under target
                if len(extracted_tickers) < TARGET:
                    candidates = re.findall(r'\b[A-Z]{2,5}\b', f"{headline} {summary}")
                    for ticker in candidates:
                        if len(extracted_tickers) >= TARGET:
                            break
                        if ticker not in extracted_tickers and is_real_optionable_stock(ticker):
                            extracted_tickers.add(ticker)
                            temp_headline_map.setdefault(ticker, headline)

        final_list = list(extracted_tickers)[:22]
        st.session_state.cached_watchlist = final_list
        st.session_state.headline_map = {
            t: temp_headline_map.get(t, "Trending via active corporate events desk narrative.")
            for t in final_list
        }

    except Exception as e:
        st.error(f"Error in narrative expansion pipeline: {e}")

# ─────────────────────────────────────────────────────────────
# SENTIMENT ENGINE  (FinBERT + Finnhub social)
# ─────────────────────────────────────────────────────────────
def get_cross_channel_sentiment(ticker_symbol: str) -> tuple[float, float, float, list[str]]:
    """
    Returns (blended_score, news_score, social_score, headlines_used)

    news_score  : FinBERT average over last 20 days of corporate headlines
    social_score: Finnhub social-sentiment normalised to [-1, +1]
                  using (bullish_score - bearish_score) / total_score
                  — NOT raw clip, which was the original bug.
    Blend: 60% news (institutional) / 40% social (retail momentum)
    """
    api_key = st.secrets.get("FINNHUB_API_KEY", "")
    sym = ticker_symbol.strip().upper()
    headlines_used: list[str] = []

    # ── 1. FinBERT on recent corporate news ──────────────────
    news_score = 0.0
    try:
        today = datetime.today().strftime("%Y-%m-%d")
        past  = (datetime.today() - timedelta(days=20)).strftime("%Y-%m-%d")
        url   = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={past}&to={today}&token={api_key}"
        res   = requests.get(url, timeout=10).json()

        if isinstance(res, list) and res:
            texts = []
            for item in res[:15]:          # Cap at 15 to stay within model throughput
                h = item.get("headline", "")
                s = item.get("summary", "")
                combined = f"{h}. {s}".strip(". ")
                if combined:
                    texts.append(combined)
                    headlines_used.append(h)
            news_score = batch_finbert_score(texts)
    except Exception:
        pass

    # ── 2. Social sentiment (Reddit + Twitter via Finnhub) ───
    # FIX vs original: Finnhub social sentiment exposes bullishPercent & bearishPercent.
    # Correct normalisation: bullish_pct - bearish_pct → already in [0,1] space → map to [-1,+1]
    social_score = 0.0
    try:
        social_url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={sym}&token={api_key}"
        social_data = requests.get(social_url, timeout=10).json()

        scores: list[float] = []
        for platform in ("reddit", "twitter"):
            for entry in social_data.get(platform, [])[:5]:
                bull = entry.get("bullishPercent", None)
                bear = entry.get("bearishPercent", None)
                if bull is not None and bear is not None:
                    # Map [0%,100%] bullish/bearish to [-1,+1]
                    scores.append((bull - bear) / 100.0)
                elif entry.get("score", 0) != 0:
                    # Fallback: use raw score but normalise by positiveMention totals if available
                    pos = entry.get("positiveCount", 1)
                    neg = entry.get("negativeCount", 1)
                    total = pos + neg
                    if total > 0:
                        scores.append((pos - neg) / total)

        if scores:
            social_score = round(max(min(sum(scores) / len(scores), 1.0), -1.0), 4)
    except Exception:
        pass

    # ── 3. Blend ─────────────────────────────────────────────
    blended = round((news_score * 0.60) + (social_score * 0.40), 4)
    return blended, news_score, social_score, headlines_used

# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_confluence_data(ticker_symbol: str) -> dict | None:
    try:
        df = yf.download(
            ticker_symbol.strip(),
            period="1y", interval="1d",
            group_by="column", progress=False, auto_adjust=True
        )
        if df is None or df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        today_close     = float(close.iloc[-1])
        yesterday_close = float(close.iloc[-2])
        ten_days_ago    = float(close.iloc[-10])

        trend_10d      = round(((today_close - ten_days_ago) / ten_days_ago) * 100, 2)
        is_up_today    = today_close > yesterday_close

        ma200 = close.rolling(200).mean()
        above_macro_trend = (
            today_close > float(ma200.iloc[-1])
            if len(close) >= 200 and not pd.isna(ma200.iloc[-1])
            else None          # Not enough history — don't penalise
        )

        avg_vol_10d = float(volume.iloc[-11:-1].mean())
        rvol        = round(float(volume.iloc[-1]) / (avg_vol_10d + 1e-9), 2)
        rsi         = round(float(calculate_rsi(close, 14).iloc[-1]), 2)

        # Additional: 50-day MA cross signal
        ma50 = close.rolling(50).mean()
        above_50ma = (
            today_close > float(ma50.iloc[-1])
            if len(close) >= 50 and not pd.isna(ma50.iloc[-1])
            else None
        )

        # Average True Range (ATR) as a volatility proxy
        high = df["High"].squeeze()
        low  = df["Low"].squeeze()
        atr  = round(float((high - low).rolling(14).mean().iloc[-1]), 2)

        return {
            "price":             today_close,
            "trend_10d":         trend_10d,
            "is_up_today":       is_up_today,
            "above_macro_trend": above_macro_trend,
            "above_50ma":        above_50ma,
            "rvol":              rvol,
            "rsi":               rsi,
            "atr":               atr,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# CONFLUENCE SCORING  (now 6 criteria, each explained)
# ─────────────────────────────────────────────────────────────
def score_confluence(tech: dict, sentiment: float) -> tuple[int, dict[str, bool]]:
    """
    6-point confluence checklist — returns (score, breakdown dict).
    None values (insufficient history) are treated as non-contributing
    rather than penalising stocks with short histories.
    """
    checks = {
        "Positive FinBERT Sentiment":  sentiment >= 0.10,
        "10d Price Uptrend":           tech["trend_10d"] > 0,
        "Up Today":                    tech["is_up_today"],
        "Above 200-day MA":            bool(tech["above_macro_trend"]) if tech["above_macro_trend"] is not None else False,
        "Above 50-day MA":             bool(tech["above_50ma"])        if tech["above_50ma"] is not None else False,
        "Volume Spike (RVOL ≥ 1.5)":  tech["rvol"] >= 1.5,
    }
    score = sum(checks.values())
    return score, checks

def recommend_strategy(score: int, rsi: float, sentiment: float) -> str:
    if score >= 5:
        if rsi > 75:
            return "🟡 CALL HOLD — Overbought (RSI>75). Wait for pullback."
        return "🟢 BUY LONG CALLS — Strong FinBERT + Technical Confluence."
    elif score <= 1 and sentiment <= -0.10:
        if rsi < 25:
            return "🟡 PUT HOLD — Oversold (RSI<25). Wait for dead-cat bounce."
        return "🔴 BUY LONG PUTS — High Confluence Breakdown Signal."
    elif score == 4:
        return "🔵 MODERATE BULLISH — Missing 1-2 confirmations."
    elif score == 2:
        return "🟠 MODERATE BEARISH — Limited bullish confirmation."
    else:
        return "🚫 NO SETUP — Conflicting signals, avoid options here."

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Target Matrix Controls")

scan_mode = st.sidebar.radio(
    "Discovery Feed:",
    ("📰 Live Dynamic News & Social Stream", "✍️ Manual Ticker Entry")
)

if scan_mode == "✍️ Manual Ticker Entry":
    user_input = st.sidebar.text_input(
        "Enter Ticker Symbols (comma separated):",
        value="AAPL, NVDA, TSLA, MSFT, AMZN"
    )
    active_watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    if not st.session_state.cached_watchlist:
        with st.spinner("Scanning live Finnhub narrative wires..."):
            fetch_and_expand_narrative()

    if st.sidebar.button("🔄 Refresh — Scrape Fresh Narrative Wires"):
        with st.spinner("Flushing cache and reloading next news cycle..."):
            st.cache_data.clear()
            fetch_and_expand_narrative()

    active_watchlist = st.session_state.cached_watchlist

if active_watchlist:
    st.sidebar.success(f"✅ {len(active_watchlist)} tickers loaded")
    st.sidebar.caption(", ".join(active_watchlist))

# FinBERT info in sidebar
with st.sidebar.expander("ℹ️ About the Sentiment Model"):
    st.write(
        "**FinBERT** (ProsusAI/finbert) is a BERT model fine-tuned on "
        "10-K filings, earnings call transcripts, and financial news. "
        "It outputs Positive / Negative / Neutral probabilities. "
        "Score = P(positive) − P(negative) → range −1.0 to +1.0.\n\n"
        "This replaces the original VADER model, which was built for "
        "social media and has no financial domain knowledge."
    )

run_scan = st.sidebar.button("🛡️ Run Scan Matrix", type="primary")

# ─────────────────────────────────────────────────────────────
# MAIN SCAN PIPELINE
# ─────────────────────────────────────────────────────────────
if run_scan:
    if not active_watchlist:
        st.warning("No tickers loaded. Add tickers or run the live scraper first.")
        st.stop()

    results = []
    progress_bar = st.progress(0, text="Initialising FinBERT pipeline...")
    total = len(active_watchlist)

    for i, ticker in enumerate(active_watchlist):
        progress_bar.progress((i + 1) / total, text=f"Analysing {ticker} ({i+1}/{total})...")

        tech = get_confluence_data(ticker)
        if tech is None:
            continue

        blended, news_score, social_score, headlines = get_cross_channel_sentiment(ticker)
        score, checks = score_confluence(tech, blended)
        strategy = recommend_strategy(score, tech["rsi"], blended)

        results.append({
            "Ticker":                    ticker,
            "Price ($)":                 round(tech["price"], 2),
            "Confluence (/6)":           score,
            "Options Strategy":          strategy,
            "FinBERT Blended Score":     blended,
            "FinBERT News Score":        round(news_score, 4),
            "Social Score":              round(social_score, 4),
            "RSI (14)":                  tech["rsi"],
            "RVOL":                      tech["rvol"],
            "ATR (14d)":                 tech["atr"],
            "10d Trend (%)":             tech["trend_10d"],
            "Above 200MA":               "✅" if tech["above_macro_trend"] else ("➖" if tech["above_macro_trend"] is None else "❌"),
            "Above 50MA":                "✅" if tech["above_50ma"] else ("➖" if tech["above_50ma"] is None else "❌"),
            "Up Today":                  "✅" if tech["is_up_today"] else "❌",
            "Narrative Catalyst":        st.session_state.headline_map.get(ticker, "Active float/volume tracking."),
            "_checks":                   checks,
            "_headlines":                headlines[:3],
        })

    progress_bar.empty()

    if not results:
        st.error("No valid tickers matched during this scan window. Try refreshing or adding manual tickers.")
        st.stop()

    df = pd.DataFrame(results)

    # ── Results Table (hide internal cols) ─────────────────
    display_cols = [c for c in df.columns if not c.startswith("_")]
    st.write(f"### 🎯 FinBERT Confluence Matrix — {len(df)} Stocks")
    st.dataframe(
        df[display_cols].style.background_gradient(
            subset=["Confluence (/6)"], cmap="RdYlGn"
        ).background_gradient(
            subset=["FinBERT Blended Score"], cmap="RdYlGn"
        ),
        use_container_width=True
    )

    # ── Scatter: Sentiment vs Volume ────────────────────────
    st.write("### 🗺️ FinBERT Sentiment vs Volume Spike Cluster Map")
    fig = px.scatter(
        df,
        x="FinBERT Blended Score",
        y="RVOL",
        text="Ticker",
        color="Options Strategy",
        size="RSI (14)",
        hover_data=["Price ($)", "10d Trend (%)", "ATR (14d)", "Confluence (/6)"],
        range_x=[-1, 1],
        title="Options Signal Clusters (FinBERT News + Social Sentiment vs Relative Volume)",
        labels={
            "FinBERT Blended Score": "FinBERT Score (News 60% + Social 40%)",
            "RVOL": "Relative Volume (RVOL)"
        },
        color_discrete_map={
            "🟢 BUY LONG CALLS — Strong FinBERT + Technical Confluence.":    "#00c853",
            "🔴 BUY LONG PUTS — High Confluence Breakdown Signal.":           "#d50000",
            "🟡 CALL HOLD — Overbought (RSI>75). Wait for pullback.":         "#ffd600",
            "🟡 PUT HOLD — Oversold (RSI<25). Wait for dead-cat bounce.":     "#ff6d00",
            "🔵 MODERATE BULLISH — Missing 1-2 confirmations.":              "#2979ff",
            "🟠 MODERATE BEARISH — Limited bullish confirmation.":            "#ff6d00",
            "🚫 NO SETUP — Conflicting signals, avoid options here.":        "#9e9e9e",
        }
    )
    fig.update_traces(textposition="top center", textfont=dict(size=12, family="Arial Black"))
    fig.add_hline(y=1.5, line_dash="dash", line_color="black",
                  annotation_text="Institutional Volume Baseline (RVOL 1.5x)")
    fig.add_vline(x=0,    line_dash="dash", line_color="gray",
                  annotation_text="Sentiment Neutral")
    fig.add_vline(x=0.10, line_dash="dot",  line_color="green",
                  annotation_text="FinBERT Bullish Threshold")
    fig.add_vline(x=-0.10,line_dash="dot",  line_color="red",
                  annotation_text="FinBERT Bearish Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # ── Confluence Breakdown Heatmap ─────────────────────────
    st.write("### 🔬 Confluence Criteria Breakdown")
    checks_data = {}
    for row in results:
        checks_data[row["Ticker"]] = row["_checks"]

    checks_df = pd.DataFrame(checks_data).T.astype(int)
    fig2 = px.imshow(
        checks_df,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Per-Ticker Confluence Criteria (Green = Met, Red = Not Met)",
        labels=dict(color="Met")
    )
    fig2.update_xaxes(tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

    # ── FinBERT Headlines Used (expandable) ─────────────────
    st.write("### 📰 FinBERT Input Headlines (sample per ticker)")
    for row in results:
        if row["_headlines"]:
            with st.expander(f"{row['Ticker']} — Score: {row['FinBERT Blended Score']}"):
                for h in row["_headlines"]:
                    sentiment_val = finbert_score(h)
                    colour = "🟢" if sentiment_val > 0.10 else ("🔴" if sentiment_val < -0.10 else "⚪")
                    st.write(f"{colour} `{sentiment_val:+.3f}` — {h}")

    # ── Disclaimer ───────────────────────────────────────────
    st.info(
        "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
        "FinBERT sentiment scores are not financial advice. Options trading involves "
        "significant risk of loss. Always conduct your own due diligence."
    )

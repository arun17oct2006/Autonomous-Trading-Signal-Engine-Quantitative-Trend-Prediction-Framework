import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Quant Signal Engine", layout="wide", initial_sidebar_state="expanded")

# Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .signal-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# App Header with a clean layout
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🏹 Autonomous Trading Signal Engine")
    st.caption("Quantitative Trend Prediction Framework | v2.0")

# ==========================================
# TICKER RESOLVER (LOGIC UNCHANGED)
# ==========================================
def get_ticker_from_search(user_input):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={user_input}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data.get('quotes'):
            stocks = [q for q in data['quotes'] if q.get('quoteType') == 'EQUITY']
            return stocks[0]['symbol'] if stocks else data['quotes'][0]['symbol']
    except Exception:
        pass
    return None

def resolve_ticker(user_input):
    clean_input = user_input.strip().upper()
    common_map = {"APPL": "AAPL", "MSTF": "MSFT", "GOGL": "GOOGL", "AMZ": "AMZN", "TSL": "TSLA"}
    if clean_input in common_map: return common_map[clean_input]
    try:
        t = yf.Ticker(clean_input)
        if t.fast_info.get('lastPrice'): return clean_input
    except Exception: pass
    found = get_ticker_from_search(user_input)
    return found if found else clean_input

# --- SIDEBAR ---
with st.sidebar:
    # Add a professional branding image for the app
    st.image("https://img.icons8.com/fluency/144/stock-share.png", width=80) 
    st.header("Control Panel")
    user_input = st.text_input("Enter Ticker or Name", "AAPL")
    ticker = resolve_ticker(user_input)
    
    st.success(f"Target: **{ticker}**")
    run_btn = st.button("Generate Signals", use_container_width=True)
    
    st.divider()
    st.markdown("### Model Parameters")
    st.info("Ensemble: XGBoost + RF\n\nTarget: Excess Return (5D-MA)")

# --- MAIN LOGIC ---
if run_btn:
    with st.spinner("Processing Market Intelligence..."):
        stock = yf.Ticker(ticker)
        try:
            df = stock.history(period="5y", auto_adjust=True)
        except:
            st.error("Data fetch failed")
            st.stop()

        if df.empty:
            st.error(f"Invalid Ticker: {ticker}")
            st.stop()

        df.columns = [str(col).lower() for col in df.columns]

        # Features (Logic Kept Exactly)
        df["return"] = df["close"].pct_change()
        def compute_rsi(series, period=14):
                    delta = series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                    rs = gain / (loss + 1e-9)
                    return 100 - (100 / (1 + rs))

        df["rsi"] = compute_rsi(df["close"])
        df["rsi_change"] = df["rsi"].diff()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["price_dist_sma"] = (df["close"] - df["sma_20"]) / (df["sma_20"] + 1e-9)
        df["vol_surge"] = df["volume"] / (df["volume"].rolling(5).mean() + 1e-9)
        df["z_score"] = (df["return"] - df["return"].rolling(20).mean()) / (df["return"].rolling(20).std() + 1e-9)
        df["lag_ret_1"] = df["return"].shift(1)
        df["target_day"] = df["return"].shift(-1) - df["return"].rolling(5).mean()

        features = ["rsi_change", "price_dist_sma", "vol_surge", "z_score", "lag_ret_1"]
        data = df.dropna()
        split = int(len(data) * 0.8)
        train, test = data[:split], data[split:]
        X_train, y_train = train[features], train["target_day"]
        X_test, y_test = test[features], test["target_day"]

        # Models
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_train, y_train)
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=5).fit(X_train, y_train)

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))

        if xgb_rmse < rf_rmse:
            model, model_name, rmse = xgb, "XGBoost", xgb_rmse
        else:
            model, model_name, rmse = rf, "Random Forest", rf_rmse

        # Stats Processing
        preds = model.predict(X_test)
        rolling_mean = data["return"].rolling(5).mean().iloc[split:]
        pred_returns = preds + rolling_mean
        actual_returns = y_test + rolling_mean
        hit_ratio = (np.sign(pred_returns) == np.sign(actual_returns)).mean()

        # Latest Prediction
        latest = data[features].iloc[-1:]
        next_pred = model.predict(latest)[0]
        current_ma = data["return"].rolling(5).mean().iloc[-1]
        final_pred = next_pred + current_ma
        price = df["close"].iloc[-1]
        target_price = price * (1 + final_pred)
        rsi_now = df["rsi"].iloc[-1]

        # --- REFINED DASHBOARD ---
        
        # Row 1: Key Executive Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Live Price", f"${price:.2f}")
        with c2:
            st.metric("Predicted Target", f"${target_price:.2f}", f"{(final_pred*100):+.2f}%")
        with c3:
            st.metric("Confidence (Hit Ratio)", f"{hit_ratio:.2%}")

        st.divider()

        # Row 2: Prediction Result Card
        if final_pred > 0.005 and rsi_now < 70:
            st.markdown('<div style="background-color:rgba(0,255,0,0.1); border:1px solid green; border-radius:10px; padding:20px; text-align:center;">'
                        '<h2 style="color:green; margin:0;">🚀 SIGNAL: STRONG BUY</h2></div>', unsafe_allow_html=True)
        elif final_pred < -0.005 and rsi_now > 30:
            st.markdown('<div style="background-color:rgba(255,0,0,0.1); border:1px solid red; border-radius:10px; padding:20px; text-align:center;">'
                        '<h2 style="color:red; margin:0;">📉 SIGNAL: STRONG SELL</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color:rgba(255,255,0,0.1); border:1px solid orange; border-radius:10px; padding:20px; text-align:center;">'
                        '<h2 style="color:orange; margin:0;">⚖️ SIGNAL: NEUTRAL / HOLD</h2></div>', unsafe_allow_html=True)

        st.write("")

        # Row 3: Visual Analytics
        tab_chart, tab_info = st.tabs(["📊 Market Analytics", "🧬 Model Insights"])

        with tab_chart:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index[-120:], df["close"][-120:], label="Market Price", color='#00d4ff', linewidth=2)
            ax.fill_between(df.index[-120:], df["close"][-120:], alpha=0.1, color='#00d4ff')
            ax.plot(df.index[-120:], df["sma_20"][-120:], label="SMA 20", color='#ffaa00', linestyle='--')
            ax.set_title(f"{ticker} Recent Performance", color="white")
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e2227')
            st.pyplot(fig)

        with tab_info:
            col_l, col_r = st.columns(2)
            with col_l:
                st.write(f"**Optimization:** {model_name}")
                st.write(f"**Current RSI:** {rsi_now:.2f}")
            with col_r:
                # Adding an illustrative image for AI Training
                st.image("https://img.icons8.com/fluency/100/artificial-intelligence.png", width=50)
                st.caption("Heuristic Scoring & XGBoost regression active.")

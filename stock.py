import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# 1. USER INPUT & SETUP
ticker_symbol = input("Enter Stock Ticker (e.g., TSLA, NVDA, RELIANCE.NS): ").upper()
today_dt = datetime.now() - timedelta(days=2)  # Use yesterday's date for prediction context

print(f"\n--- Running Anti-Lag Precision Model for {ticker_symbol} ---")

# 2. DATA ACQUISITION (5 Years for Deep Context)
df = yf.download(ticker_symbol, period="5y", auto_adjust=True)
if df.empty:
    print("Error: Ticker not found.")
    exit()

# Flatten MultiIndex and Lowercase
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(col).lower() for col in df.columns]

# 3. ADVANCED FEATURE ENGINEERING (The "Velocity" Logic)
df["return"] = df["close"].pct_change()

# Feature 1: RSI Velocity (Is momentum speeding up or slowing down?)
df["rsi"] = ta.rsi(df["close"], length=14)
df["rsi_change"] = df["rsi"].diff()

# Feature 2: Price Stretch (How far is price from its 'normal' 20-day average?)
df["sma_20"] = df["close"].rolling(20).mean()
df["price_dist_sma"] = (df["close"] - df["sma_20"]) / df["sma_20"]

# Feature 3: Volume Surge (Is big money entering?)
df["vol_surge"] = df["volume"] / df["volume"].rolling(5).mean()

# Feature 4: Z-Score (Statistical Volatility)
df["z_score"] = (df["return"] - df["return"].rolling(20).mean()) / (df["return"].rolling(20).std() + 1e-9)

# Feature 5: Lags (Short-term memory)
df["lag_ret_1"] = df["return"].shift(1)
df["lag_ret_2"] = df["return"].shift(2)

# 4. TARGETS (Predicting the 'Breakout', not the 'Price')
# We subtract the 5-day mean to force the model to predict 'excess' movement
df["target_day"] = df["return"].shift(-1) - df["return"].rolling(5).mean()
df["target_week"] = df["return"].shift(-5).rolling(5).sum()

# Clean up
features = ["rsi_change", "price_dist_sma", "vol_surge", "z_score", "lag_ret_1", "lag_ret_2"]
full_data = df.dropna().copy()

# 5. HUBER REGRESSION ENGINE (Ignores market noise)
def get_prediction(target_col):
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.005,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:pseudohubererror', # Crucial: Treats $5 jumps as signals, not errors
        random_state=42
    )
    model.fit(full_data[features], full_data[target_col])
    
    # Use the absolute last completed row for prediction
    latest_row = df[features].tail(1)
    if latest_row.isnull().values.any():
        latest_row = full_data[features].tail(1)
        
    return model.predict(latest_row)[0]

# 6. EXECUTION
print("Calculating market breakouts...")
pred_day_raw = get_prediction("target_day")
pred_week_raw = get_prediction("target_week")

# Re-adjusting the prediction back to a readable percentage
# (Adding back the recent mean return)
current_mean_ret = df["return"].rolling(5).mean().iloc[-1]
final_day_pred = pred_day_raw + current_mean_ret
final_week_pred = pred_week_raw 

curr_price = df['close'].iloc[-1]

# 7. FINAL REPORT
target_date = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")

print("\n" + "="*55)
print(f"PREDICTION REPORT: {ticker_symbol} | Generated: {datetime.now().strftime('%H:%M')}")
print(f"Current Price: ${curr_price:.2f}")
print("-" * 55)
print(f"Forecast for {target_date}:")
print(f"  -> Predicted Return: {final_day_pred:+.2%}")
print(f"  -> Estimated Price:  ${curr_price * (1 + final_day_pred):.2f}")
print("-" * 55)
print(f"1-Week Outlook:")
print(f"  -> Predicted Return: {final_week_pred:+.2%}")
print(f"  -> Estimated Price:  ${curr_price * (1 + final_week_pred):.2f}")
print("="*55)

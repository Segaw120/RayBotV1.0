# ------------------- CHUNK 1 -------------------
import os
import logging
import time
import threading
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sodapy import Socrata
from yahooquery import Ticker
import streamlit as st

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Assets List (COT names with tickers) ---
assets_list = [
    {"name": "GOLD - COMMODITY EXCHANGE INC.", "symbol": "GC=F"},
    {"name": "SILVER - COMMODITY EXCHANGE INC.", "symbol": "SI=F"},
    {"name": "EURO FX - CHICAGO MERCANTILE EXCHANGE", "symbol": "6E=F"},
    {"name": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "symbol": "6J=F"},
    {"name": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", "symbol": "6B=F"},
    {"name": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "symbol": "6C=F"},
    {"name": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "symbol": "6A=F"},
    {"name": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", "symbol": "6S=F"},
    {"name": "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "symbol": "ES=F"},
    {"name": "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "symbol": "NQ=F"},
    {"name": "DOW JONES INDUSTRIAL AVERAGE - CHICAGO MERCANTILE EXCHANGE", "symbol": "YM=F"},
    {"name": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE", "symbol": "CL=F"},
    {"name": "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE", "symbol": "NG=F"},
    {"name": "COPPER - COMMODITY EXCHANGE INC.", "symbol": "HG=F"},
    {"name": "PLATINUM - NEW YORK MERCANTILE EXCHANGE", "symbol": "PL=F"},
    {"name": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE", "symbol": "PA=F"},
]

# ------------------------------
# Fetching Functions
# ------------------------------

def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching COT data for {market_name}")
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500,
            )
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", 0), errors="coerce")
                # Commercial and non-commercial positions
                df["commercial_long"] = pd.to_numeric(df.get("commercial_long_all", 0), errors="coerce")
                df["commercial_short"] = pd.to_numeric(df.get("commercial_short_all", 0), errors="coerce")
                df["non_commercial_long"] = pd.to_numeric(df.get("non_commercial_long_all", 0), errors="coerce")
                df["non_commercial_short"] = pd.to_numeric(df.get("non_commercial_short_all", 0), errors="coerce")
                df["commercial_net"] = df["commercial_long"] - df["commercial_short"]
                df["non_commercial_net"] = df["non_commercial_long"] - df["non_commercial_short"]

                # Position percentages
                df["commercial_position_pct"] = (df["commercial_long"] / (df["commercial_long"] + df["commercial_short"])) * 100
                df["non_commercial_position_pct"] = (df["non_commercial_long"] / (df["non_commercial_long"] + df["non_commercial_short"])) * 100

                # Z-scores (52-week rolling)
                df["commercial_net_zscore"] = (df["commercial_net"] - df["commercial_net"].rolling(52).mean()) / df["commercial_net"].rolling(52).std()
                df["non_commercial_net_zscore"] = (df["non_commercial_net"] - df["non_commercial_net"].rolling(52).mean()) / df["non_commercial_net"].rolling(52).std()

                return df.sort_values("report_date").reset_index(drop=True)
            else:
                logger.warning(f"No COT data for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
            attempt += 1
            time.sleep(2)
    logger.error(f"Failed fetching COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

def fetch_yahooquery_data(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker]
            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["date"])
            # Add technical indicators
            hist = calculate_technical_indicators(hist)
            return hist.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
            attempt += 1
            time.sleep(2)
    logger.error(f"Failed fetching Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close_col = "close" if "close" in df.columns else "Close"
    high_col = "high" if "high" in df.columns else "High"
    low_col = "low" if "low" in df.columns else "Low"
    vol_col = "volume" if "volume" in df.columns else "Volume"

    # RVOL (20-day)
    if vol_col in df.columns:
        df["rvol"] = df[vol_col] / df[vol_col].rolling(20).mean()
    # RSI
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    # SMAs
    df["sma20"] = df[close_col].rolling(20).mean()
    df["sma50"] = df[close_col].rolling(50).mean()
    df["sma200"] = df[close_col].rolling(200).mean()
    # Bollinger Bands
    df["bb_middle"] = df[close_col].rolling(20).mean()
    df["bb_std"] = df[close_col].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    # ATR14
    tr1 = df[high_col] - df[low_col]
    tr2 = abs(df[high_col] - df[close_col].shift())
    tr3 = abs(df[low_col] - df[close_col].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    # Volatility
    df["volatility"] = df[close_col].pct_change().rolling(20).std() * np.sqrt(252) * 100
    # 52-week high/low
    df["52w_high"] = df[close_col].rolling(252).max()
    df["52w_low"] = df[close_col].rolling(252).min()
    df["pct_from_52w_high"] = (df[close_col] / df["52w_high"] - 1) * 100
    df["pct_from_52w_low"] = (df[close_col] / df["52w_low"] - 1) * 100
    return df




# ------------------- CHUNK 2 -------------------

# --- Merge COT and Price Data ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()

    cot_columns = ["report_date", "open_interest_all", "commercial_net", "non_commercial_net", 
                   "commercial_position_pct", "non_commercial_position_pct", 
                   "commercial_net_zscore", "non_commercial_net_zscore"]

    for col in cot_columns:
        if col not in cot_df.columns:
            cot_df[col] = np.nan

    cot_df_small = cot_df[cot_columns].copy()
    cot_df_small.rename(columns={"report_date": "date"}, inplace=True)
    cot_df_small["date"] = pd.to_datetime(cot_df_small["date"])

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date")
    cot_df_small = cot_df_small.sort_values("date")

    full_dates = pd.DataFrame({"date": pd.date_range(price_df["date"].min(), price_df["date"].max())})
    cot_df_filled = pd.merge_asof(full_dates, cot_df_small, on="date", direction="backward")

    merged = pd.merge(price_df, cot_df_filled, on="date", how="left")

    # Forward fill COT data until next report
    for col in cot_columns[1:]:
        merged[col] = merged[col].ffill()

    return merged

# --- Calculate Health Gauge (365-day rolling) ---
def calculate_health_gauge(merged_df: pd.DataFrame) -> float:
    if merged_df.empty:
        return np.nan

    latest = merged_df.tail(1).iloc[0]
    recent = merged_df.tail(365).copy()

    close_col = "close" if "close" in recent.columns else "Close"
    if close_col is None:
        return np.nan

    scores = []

    # 1. Commercial net z-score (25%)
    if "commercial_net_zscore" in latest and not pd.isna(latest["commercial_net_zscore"]):
        comm_score = max(0, min(1, 0.5 - latest["commercial_net_zscore"]/4))
        scores.append((comm_score, 0.25))

    # 2. Trend alignment (20%)
    if all(x in latest for x in ["sma20","sma50","sma200"]):
        trend_signals = [
            latest[close_col] > latest["sma20"],
            latest["sma20"] > latest["sma50"],
            latest["sma50"] > latest["sma200"]
        ]
        trend_score = sum(trend_signals) / len(trend_signals)
        scores.append((trend_score, 0.20))

    # 3. Momentum (RSI) (15%)
    if "rsi" in latest and not pd.isna(latest["rsi"]):
        rsi = latest["rsi"]
        if rsi < 30: rsi_score=0.3
        elif rsi > 70: rsi_score=0.7
        else: rsi_score=0.5 + (rsi - 50)/100
        scores.append((rsi_score, 0.15))

    # 4. Volatility & volume (15%)
    vol_score = 0.5
    if all(x in latest for x in ["bb_width","rvol"]):
        bb_percentile = stats.percentileofscore(recent["bb_width"].dropna(), latest["bb_width"])/100
        bb_score = 1 - bb_percentile
        rvol_score = min(1.0, latest["rvol"]/2.0) if not pd.isna(latest["rvol"]) else 0.5
        vol_score = 0.7*bb_score + 0.3*rvol_score
        scores.append((vol_score,0.15))

    # 5. Distance from 52-week high/low (15%)
    if all(x in latest for x in ["pct_from_52w_high","pct_from_52w_low"]):
        high_score = max(0,min(1,1-abs(latest["pct_from_52w_high"])/100))
        low_score = max(0,min(1,latest["pct_from_52w_low"]/100))
        dist_score = 0.7*high_score + 0.3*low_score
        scores.append((dist_score,0.15))

    # 6. Open interest (10%)
    if "open_interest_all" in latest and not pd.isna(latest["open_interest_all"]):
        oi = recent["open_interest_all"].dropna()
        if not oi.empty:
            oi_pctile = stats.percentileofscore(oi, latest["open_interest_all"])/100
            scores.append((oi_pctile,0.10))

    weighted_sum = sum(score*weight for score,weight in scores)
    total_weight = sum(weight for _,weight in scores)
    health_score = (weighted_sum/total_weight)*10 if total_weight>0 else 5.0
    return float(health_score)

# --- Generate Signals based on Health Gauge ---
def generate_signals(merged_df: pd.DataFrame) -> Dict:
    if merged_df.empty:
        return {"signal":"NEUTRAL","strength":0,"reasoning":"Insufficient data"}

    latest = merged_df.iloc[-1]
    health_score = latest.get("health_score",5.0)

    reasoning = f"Health gauge: {health_score:.2f}/10"
    if health_score > 6:
        signal = "BUY"
        strength = 3
    elif health_score < 4:
        signal = "SELL"
        strength = 3
    else:
        signal = "NEUTRAL"
        strength = 0

    return {"signal":signal,"strength":strength,"reasoning":reasoning}

# --- Backtest Example ---
def backtest_asset(asset: Dict, start_date: str, end_date: str) -> pd.DataFrame:
    cot_df = fetch_cot_data(asset["name"])
    price_df = fetch_yahooquery_data(asset["symbol"], start_date, end_date)
    merged_df = merge_cot_price(cot_df, price_df)
    if merged_df.empty:
        return pd.DataFrame()
    merged_df["health_score"] = merged_df.apply(lambda row: calculate_health_gauge(merged_df.loc[:row.name]), axis=1)
    merged_df["signal_info"] = merged_df.apply(lambda row: generate_signals(merged_df.loc[:row.name]), axis=1)
    return merged_df

# --- Example usage ---
if __name__ == "__main__":
    start_date = (datetime.today() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    for asset in assets_list[:2]:  # Two at a time
        df = backtest_asset(asset, start_date, end_date)
        print(f"{asset['name']} backtest completed. Latest signal: {df['signal_info'].iloc[-1] if not df.empty else 'No Data'}")




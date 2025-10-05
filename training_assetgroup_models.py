import os
import uuid
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# optional libraries
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# Hugging Face
try:
    from huggingface_hub import HfApi, upload_file
except Exception:
    HfApi = None
    upload_file = None

# Supabase
try:
    from supabase import create_client
except Exception:
    create_client = None

# CascadeTrader placeholder
try:
    from cascade_trader import CascadeTrader  # Adjust to your actual module path
except Exception:
    CascadeTrader = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cascade_trader_app")

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Cascade Trader + HF + Supabase", layout="wide")
st.title("Cascade Trader — Asset Group Specialized Training")

# ---------------- Asset Group ----------------
COTTickerMap = {
    "FX": ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"],
    "Metals": ["GC=F","SI=F","PL=F","PA=F","HG=F","ALI=F","NI=F","ZNC=F","PB=F","TIN=F"],
    "Energy": ["CL=F","BZ=F","NG=F","HO=F","RB=F","LPG=F"],
    "Indices": ["^GSPC","^DJI","^IXIC","^RUT","^FTSE","^GDAXI","^FCHI","^N225","^HSI","000001.SS","^VIX"],
    "Crypto": ["BTC-USD","ETH-USD","XRP-USD","LTC-USD","ADA-USD","DOT-USD","SOL-USD","DOGE-USD"]
}

asset_group = st.selectbox("Select Asset Group to Train", list(COTTickerMap.keys()))
tickers = COTTickerMap[asset_group]

# ---------------- Training Settings ----------------
st.sidebar.header("Training Settings")
seq_len = int(st.sidebar.number_input("L1 sequence length", min_value=8, max_value=256, value=64))
epochs_l1 = int(st.sidebar.number_input("L1 epochs", min_value=1, value=8))
epochs_l23 = int(st.sidebar.number_input("L2/L3 epochs", min_value=1, value=8))
device_choice = st.sidebar.selectbox("Device", ["auto","cpu","cuda"], index=0)

st.sidebar.header("Breadth & Sweep")
run_breadth = st.sidebar.checkbox("Enable breadth backtest", value=True)
run_sweep = st.sidebar.checkbox("Enable grid sweep", value=False)

st.sidebar.header("Hugging Face Upload")
hf_token = st.text_input("HF Token", type="password")
repo_name = st.text_input("HF repo name", value=f"cascade_{asset_group.lower()}")

st.sidebar.header("Supabase")
SUPABASE_URL = st.text_input("Supabase URL", type="default")
SUPABASE_KEY = st.text_input("Supabase Service Key", type="password")

# ---------------- Feature Engineering ----------------
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df.get('volume', pd.Series(0.0, index=df.index)).astype(float)
    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1,-0.999999))
    tr = (h-l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min)/denom).fillna(0.5)
    return f.replace([np.inf,-np.inf],0.0).fillna(0.0)

# ---------------- Hugging Face Upload ----------------
def hf_upload(pt_path: str, repo_name: str, hf_token: str) -> str:
    if HfApi is None or upload_file is None:
        st.warning("HF libraries not installed")
        return ""
    api = HfApi()
    username = api.whoami(token=hf_token)["name"]
    repo_id = f"{username}/{repo_name}"
    api.create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
    upload_file(pt_path, pt_path, repo_id=repo_id, token=hf_token)
    return f"https://huggingface.co/{repo_id}/blob/main/{os.path.basename(pt_path)}"

# ---------------- Supabase Logging ----------------
def log_model_metrics_supabase(asset_group, ticker_stats, supabase_url, supabase_key, correlation_id=None):
    if create_client is None:
        st.warning("Supabase client not installed")
        return
    supabase = create_client(supabase_url, supabase_key)
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    # Aggregate group-level metrics
    group_metrics = {}
    for k in ticker_stats[next(iter(ticker_stats))].keys():
        group_metrics[k] = float(np.mean([v[k] for v in ticker_stats.values()]))

    # Insert per-ticker rows
    for ticker, stats in ticker_stats.items():
        row = {
            "asset_group": asset_group,
            "ticker": ticker,
            "metrics_json": json.dumps(stats),
            "num_bars": stats.get("num_bars",0),
            "num_candidates": stats.get("num_candidates",0),
            "trained_at": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id
        }
        supabase.table("model_metrics").insert(row).execute()

    # Insert aggregated group row (ticker=None)
    group_row = {
        "asset_group": asset_group,
        "ticker": None,
        "metrics_json": json.dumps(group_metrics),
        "num_bars": int(np.mean([v.get("num_bars",0) for v in ticker_stats.values()])),
        "num_candidates": int(np.mean([v.get("num_candidates",0) for v in ticker_stats.values()])),
        "trained_at": datetime.utcnow().isoformat(),
        "correlation_id": correlation_id
    }
    supabase.table("model_metrics").insert(group_row).execute()

# ---------------- Asset Group Pipeline ----------------
def run_asset_group_pipeline(asset_group, tickers):
    st.info(f"Starting pipeline for asset group: {asset_group}")
    all_bars = []
    ticker_stats = {}

    for sym in tickers:
        st.subheader(f"Fetching data for {sym}")
        try:
            tq = YahooTicker(sym)
            raw = tq.history(start=(datetime.today()-timedelta(days=90)).date().isoformat(),
                             end=datetime.today().date().isoformat(),
                             interval="1d")
            if isinstance(raw, dict): raw = pd.DataFrame(raw)
            if isinstance(raw.index, pd.MultiIndex): raw = raw.reset_index(level=0, drop=True)
            # TZ fix: unify all datetime index to tz-naive UTC
            raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)
            raw.columns = [c.lower() for c in raw.columns]
            if "close" not in raw.columns and "adjclose" in raw.columns:
                raw["close"] = raw["adjclose"]
            raw = raw[~raw.index.duplicated(keep="first")]
            raw['symbol'] = sym
            all_bars.append(raw)
        except Exception as e:
            logger.exception("Failed to fetch %s: %s", sym, e)
            st.error(f"Failed to fetch {sym}: {e}")

    if not all_bars:
        st.warning("No data fetched for this group.")
        return None

    combined_bars = pd.concat(all_bars).sort_index()

    # Generate candidates per ticker
    events = []
    for sym in tickers:
        df = combined_bars[combined_bars['symbol']==sym]
        df.loc[:, 'rvol'] = (df['volume']/df['volume'].rolling(20,min_periods=1).mean()).fillna(1.0)
        df_cands = pd.DataFrame({
            "candidate_time": df.index,
            "label": np.random.randint(0,2,len(df))
        })
        df_cands['symbol'] = sym
        events.append(df_cands)
        ticker_stats[sym] = {
            "num_bars": len(df),
            "num_candidates": len(df_cands),
            "val_acc": float(np.random.rand()),  # placeholder for real metric
            "breadth_score": float(np.random.rand())
        }

    events = pd.concat(events)
    bar_idx_map = {t:i for i,t in enumerate(combined_bars.index)}
    events['t'] = events['candidate_time'].map(lambda t: bar_idx_map.get(t,0))

    # Train cascade
    if CascadeTrader is None:
        st.error("CascadeTrader class not defined")
        return
    trader = CascadeTrader(seq_len=seq_len, feat_windows=(5,10,20), device=device_choice)
    trader.fit(combined_bars, events, l2_use_xgb=(xgb is not None), epochs_l1=epochs_l1, epochs_l23=epochs_l23)

    # Export
    out_dir = f"artifacts_{asset_group}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, f"{asset_group}_cascade.pt")
    torch.save(trader, pt_path)

    # HF upload
    hf_url = None
    if hf_token and repo_name:
        try:
            hf_url = hf_upload(pt_path, repo_name, hf_token)
        except Exception as e:
            logger.exception("HF upload failed: %s", e)
            st.error(f"Hugging Face upload failed: {e}")

    # Supabase logging
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            log_model_metrics_supabase(asset_group, ticker_stats, SUPABASE_URL, SUPABASE_KEY)
            st.success("Logged metrics to Supabase")
        except Exception as e:
            logger.exception("Supabase logging failed: %s", e)
            st.error(f"Supabase logging failed: {e}")

    st.success(f"Training & export finished for {asset_group}")
    return {"trader": trader, "pt_path": pt_path, "hf_url": hf_url, "ticker_stats": ticker_stats}

# ---------------- Run button ----------------
if st.sidebar.button("Run Asset Group Training"):
    result = run_asset_group_pipeline(asset_group, tickers)
    if result:
        st.write("Model export path:", result["pt_path"])
        if result["hf_url"]:
            st.write("Hugging Face URL:", result["hf_url"])
        st.subheader("Ticker Metrics")
        st.dataframe(pd.DataFrame(result["ticker_stats"]).T)

# training_assetgroup_models.py
"""
Full integrated asset-group training pipeline (Finnage primary / Yahoo fallback).
Includes: fetch -> features -> candidates -> events -> cascade train -> breadth -> export -> HF upload -> Supabase log
"""

import os
import uuid
import json
import math
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid

# optional libs (import guards)
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
except Exception:
    torch = None
    nn = None

# HF
try:
    from huggingface_hub import HfApi, upload_file
except Exception:
    HfApi = None
    upload_file = None

# Supabase client
try:
    from supabase import create_client
except Exception:
    create_client = None

# ---------------- Config (hardcoded per request) ----------------
SUPABASE_URL = "https://jubcotqsbvguwzklngzd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1YmNvdHFzYnZndXd6a2xuZ3pkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTU0MjA3MCwiZXhwIjoyMDc1MTE4MDcwfQ.1HV-o9JFa_nCZGXcoap2OgOCKjRSlyFSRvKmYk70eDk"

HF_TOKEN = os.getenv("HF_TOKEN")
FINNAGE_API_KEY = os.getenv("FINNAGE_API_KEY")

# ---------------- Mapping Yahoo -> Finnage symbols ----------------
yahoo_to_finage_mapping = {
    "FX": {
        "EURUSD=X": "EURUSD",
        "JPY=X": "USDJPY",
        "GBPUSD=X": "GBPUSD",
        "AUDUSD=X": "AUDUSD",
        "USDCAD=X": "USDCAD",
        "USDCHF=X": "USDCHF",
        "NZDUSD=X": "NZDUSD",
        "DX-Y.NYB": "DXY",
    },
    "Metals": {
        "GC=F": "XAUUSD",
        "SI=F": "XAGUSD",
        "PL=F": "XPTUSD",
        "PA=F": "XPDUSD",
        "HG=F": "XCUUSD",
    },
    "Energy": {
        "CL=F": "USOIL",
        "BZ=F": "UKOIL",
        "NG=F": "NGAS",
    },
    "Indices": {
        "^GSPC": "SPX500",
        "^DJI": "US30",
        "^IXIC": "NAS100",
        "^RUT": "US2000",
        "^FTSE": "UK100",
        "^GDAXI": "GER40",
        "^FCHI": "FRA40",
        "^N225": "JPN225",
        "^HSI": "HK50",
        "000001.SS": "CHINA50",
        "^VIX": "VIX",
    },
    "Crypto": {
        "BTC-USD": "BTCUSD",
        "ETH-USD": "ETHUSD",
        "XRP-USD": "XRPUSD",
        "LTC-USD": "LTCUSD",
        "ADA-USD": "ADAUSD",
        "DOT-USD": "DOTUSD",
        "SOL-USD": "SOLUSD",
        "DOGE-USD": "DOGEUSD",
    },
}

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cascade_trainer_assetgroup")
logger.setLevel(logging.DEBUG)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Cascade Trader — Finnage Primary", layout="wide")
st.title("Cascade Trader — Asset Group Specialized Training (Finnage primary)")

# asset groups
COTTickerMap = {
    "FX": ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"],
    "Metals": ["GC=F","SI=F","PL=F","PA=F","HG=F","ALI=F","NI=F","ZNC=F","PB=F","TIN=F"],
    "Energy": ["CL=F","BZ=F","NG=F","HO=F","RB=F","LPG=F"],
    "Indices": ["^GSPC","^DJI","^IXIC","^RUT","^FTSE","^GDAXI","^FCHI","^N225","^HSI","000001.SS","^VIX"],
    "Crypto": ["BTC-USD","ETH-USD","XRP-USD","LTC-USD","ADA-USD","DOT-USD","SOL-USD","DOGE-USD"]
}

asset_group = st.selectbox("Select Asset Group", list(COTTickerMap.keys()))
tickers = COTTickerMap[asset_group]

# dates: end = today, start = 1 year before
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=365)
st.sidebar.markdown(f"**Data interval:** 1d  \n**Start:** {start_date}  \n**End:** {end_date}")

# training settings
st.sidebar.header("Training settings")
seq_len = int(st.sidebar.number_input("L1 sequence length", min_value=8, max_value=512, value=64, step=8))
epochs_l1 = int(st.sidebar.number_input("L1 epochs", min_value=1, value=8))
epochs_l23 = int(st.sidebar.number_input("L2/L3 epochs", min_value=1, value=8))
device_choice = st.sidebar.selectbox("Device", ["auto","cpu","cuda"], index=0)
run_breadth = st.sidebar.checkbox("Run breadth backtest", value=True)
repo_name = st.sidebar.text_input("HF repo name", value=f"cascade_{asset_group.lower()}")

st.sidebar.text(f"HF token in env: {'present' if HF_TOKEN else 'missing'}")
st.sidebar.text("Supabase URL/key: hardcoded in script")

# ------------------------------
# Candidate generator (from candidate_generator.py)
# ------------------------------
def _true_range_series(df: pd.DataFrame) -> pd.Series:
    prev = df['close'].shift(1).fillna(df['close'].iloc[0])
    tr = pd.concat([(df['high'] - df['low']).abs(),
                    (df['high'] - prev).abs(),
                    (df['low'] - prev).abs()], axis=1).max(axis=1)
    return tr.fillna(0.0)

def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long",
    min_volume: Optional[float] = None,
    require_confluence: bool = False
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy().sort_index()
    df.index = pd.to_datetime(df.index)
    df.loc[:, 'tr'] = _true_range_series(df)
    df.loc[:, 'atr'] = df['tr'].rolling(window=atr_window, min_periods=1).mean().fillna(method='bfill')
    records: List[Dict[str,Any]] = []
    n = len(df)
    idxs = list(range(lookback, n))
    for i in idxs:
        if min_volume is not None and 'volume' in df.columns:
            if float(df['volume'].iat[i]) < min_volume:
                continue
        entry_time = df.index[i]
        entry_px = float(df['close'].iat[i])
        atr = float(df['atr'].iat[i]) if not math.isnan(df['atr'].iat[i]) else 0.0
        if atr <= 0:
            continue
        if direction == "long":
            sl_px = entry_px - k_sl * atr
            tp_px = entry_px + k_tp * atr
        else:
            sl_px = entry_px + k_sl * atr
            tp_px = entry_px - k_tp * atr
        end_i = min(i + max_bars, n - 1)
        label = 0
        hit_i = end_i
        hit_px = float(df['close'].iat[end_i])
        for j in range(i+1, end_i+1):
            hi = float(df['high'].iat[j])
            lo = float(df['low'].iat[j])
            if direction == "long":
                if hi >= tp_px:
                    label = 1; hit_i = j; hit_px = tp_px; break
                if lo <= sl_px:
                    label = 0; hit_i = j; hit_px = sl_px; break
            else:
                if lo <= tp_px:
                    label = 1; hit_i = j; hit_px = tp_px; break
                if hi >= sl_px:
                    label = 0; hit_i = j; hit_px = sl_px; break
        realized_return = (hit_px - entry_px) / entry_px if direction == "long" else (entry_px - hit_px) / entry_px
        duration_min = (df.index[hit_i] - entry_time).total_seconds() / 60.0
        rec = {
            "candidate_time": entry_time,
            "entry_price": entry_px,
            "tp_price": tp_px,
            "sl_price": sl_px,
            "label": int(label),
            "hit_idx": int(hit_i),
            "realized_return": float(realized_return),
            "duration_min": float(duration_min),
            "direction": direction
        }
        records.append(rec)
    return pd.DataFrame.from_records(records)

# ------------------------------
# Events mapping (from events.py)
# ------------------------------
def create_events_from_candidates(candidates: pd.DataFrame, bars: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Any,int]]:
    if candidates is None or candidates.empty or bars is None or bars.empty:
        return pd.DataFrame(), {}
    bars = bars.sort_index()
    ts_list = list(bars.index)
    idx_map = {ts: i for i, ts in enumerate(ts_list)}
    t_indices = []
    labels = []
    # iterate candidates preserving original order
    for _, row in candidates.iterrows():
        t = pd.to_datetime(row['candidate_time'])
        if t in idx_map:
            t_indices.append(idx_map[t])
        else:
            locs = bars.index[bars.index <= t]
            if len(locs) > 0:
                t_indices.append(idx_map[locs[-1]])
            else:
                t_indices.append(0)
        labels.append(int(row.get('label', 0)))
    events = pd.DataFrame({"t": np.array(t_indices, dtype=int), "y": np.array(labels, dtype=int)})
    return events, idx_map

# ------------------------------
# Features (from features.py)
# ------------------------------
def compute_engineered_features_advanced(df: pd.DataFrame, windows: Tuple[int,...]=(5,10,20)) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)
    ret = c.pct_change().fillna(0.0)
    f['ret1'] = ret
    f['logret1'] = np.log1p(ret.clip(lower=-0.999999).fillna(0.0))
    tr = (h - l).abs().fillna(0.0)
    f['tr'] = tr
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(w*3).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def normalize_volatility(features: pd.DataFrame, vol_column: str = "vol_20") -> pd.DataFrame:
    if features is None or features.empty:
        return features
    f = features.copy()
    if vol_column not in f.columns:
        return f
    denom = f[vol_column].replace(0, np.nan).fillna(f[vol_column].median())
    for col in f.columns:
        if col == vol_column: continue
        if pd.api.types.is_numeric_dtype(f[col]):
            f[col] = f[col] / denom
    return f.fillna(0.0)

def merge_candidate_features(candidates: pd.DataFrame, feature_frame: pd.DataFrame, seq_len:int=64) -> pd.DataFrame:
    if candidates is None or candidates.empty or feature_frame is None or feature_frame.empty:
        return pd.DataFrame()
    rows = []
    feat_vals = feature_frame.values
    n = feat_vals.shape[1]
    idx_map = {t:i for i,t in enumerate(feature_frame.index)}
    for _, r in candidates.iterrows():
        t = pd.to_datetime(r['candidate_time'])
        i = idx_map.get(t, None)
        if i is None:
            locs = feature_frame.index[feature_frame.index <= t]
            if len(locs) == 0:
                i = 0
            else:
                i = idx_map[locs[-1]]
        start = max(0, i - seq_len + 1)
        seq = feat_vals[start:i+1]
        if seq.shape[0] < seq_len:
            pad = np.repeat(seq[[0]], seq_len - seq.shape[0], axis=0)
            seq = np.vstack([pad, seq])
        flat = seq.reshape(-1)
        row = r.to_dict()
        row['flat_features'] = flat
        rows.append(row)
    return pd.DataFrame.from_records(rows)

# ------------------------------
# Breadth/backtest (from breadth.py)
# ------------------------------
def run_breadth_levels(preds: pd.DataFrame,
                       cands: pd.DataFrame,
                       bars: pd.DataFrame,
                       levels: List[Dict[str,Any]],
                       simulate_limits_fn) -> Dict[str,Any]:
    out = {"detailed": {}, "summary": []}
    if preds is None or preds.empty or cands is None or cands.empty:
        return out
    preds_map = preds.set_index('t')
    t_indices = preds['t'].values
    # scale p3 to 0..10
    sig = (preds_map.reindex(t_indices)['p3'].fillna(0.0).values * 10.0)
    cands = cands.reset_index(drop=True).copy()
    cands['signal'] = sig
    for level in levels:
        name = level.get('name','L')
        buy_min = level.get('buy_min', 0.0)
        buy_max = level.get('buy_max', 10.0)
        sl = level.get('sl', 0.02)
        rr = level.get('rr', 2.0)
        tp = rr * sl
        sel = cands[(cands['signal'] >= buy_min) & (cands['signal'] <= buy_max)].copy()
        if sel.empty:
            out['detailed'][name] = pd.DataFrame()
            continue
        sel['pred_label'] = 1
        trades = simulate_limits_fn(sel, bars, label_col='pred_label', sl=sl, tp=tp, max_holding=level.get('max_holding',60))
        out['detailed'][name] = trades
        if trades is None or trades.empty:
            continue
        total_trades = len(trades)
        win_rate = float((trades['pnl'] > 0).mean())
        avg_pnl = float(trades['pnl'].mean())
        total_pnl = float(trades['pnl'].sum())
        max_dd = float(trades['pnl'].cumsum().min())
        summary_row = {
            "mode": name,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "max_drawdown": max_dd
        }
        out['summary'].append(summary_row)
    return out

# Simple simulate_limits_fn used by breadth runner
def simulate_limits_fn(sel: pd.DataFrame, bars: pd.DataFrame, label_col='pred_label', sl=0.02, tp=0.04, max_holding=60) -> pd.DataFrame:
    """Simulate limit entry: entry at candidate close, then look forward up to max_holding bars"""
    trades = []
    bars = bars.sort_index()
    for _, r in sel.iterrows():
        t = pd.to_datetime(r['candidate_time'])
        # find bar index
        locs = bars.index[bars.index >= t]
        if len(locs) == 0:
            continue
        start_idx = bars.index.get_loc(locs[0])
        entry_px = float(bars['close'].iat[start_idx])
        sl_px = entry_px * (1 - sl)
        tp_px = entry_px * (1 + tp)
        hit = None
        for j in range(start_idx+1, min(start_idx+1+max_holding, len(bars))):
            hi = float(bars['high'].iat[j]); lo = float(bars['low'].iat[j])
            if hi >= tp_px:
                pnl = tp_px - entry_px
                hit = {'entry_time': bars.index[start_idx], 'exit_time': bars.index[j], 'pnl': pnl}
                break
            if lo <= sl_px:
                pnl = sl_px - entry_px
                hit = {'entry_time': bars.index[start_idx], 'exit_time': bars.index[j], 'pnl': pnl}
                break
        if hit is None:
            # close at last available
            exit_px = float(bars['close'].iat[min(start_idx+max_holding, len(bars)-1)])
            pnl = exit_px - entry_px
            hit = {'entry_time': bars.index[start_idx], 'exit_time': bars.index[min(start_idx+max_holding, len(bars)-1)], 'pnl': pnl}
        trades.append(hit)
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    # convert pnl to percentage if desired -- here absolute; convert to pct
    df['pnl'] = df['pnl'] / df['entry_time'].apply(lambda x: 1)  # placeholder; keep absolute
    return df

# ------------------------------
# Optimization helpers (from optimization.py) - minimal used here
# ------------------------------
def grid_search_parallel(param_grid: Dict[str,List[Any]], prepare_fn, eval_fn, max_workers: int = 4):
    pg = list(ParameterGrid(param_grid))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(lambda cfg: evaluate_config_for_group(cfg, prepare_fn, eval_fn)): cfg for cfg in pg}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
    return results

def evaluate_config_for_group(config, prepare_fn, eval_fn):
    try:
        train_df, bars = prepare_fn(config)
        metrics = eval_fn(train_df, bars, config)
        return {"config": config, "metrics": metrics}
    except Exception as e:
        logger.exception("Config eval failed: %s", e)
        return {"config": config, "metrics": {"error": str(e)}}

# ------------------------------
# Minimal CascadeTrainer (cleaned and corrected)
# ------------------------------
if torch is not None:
    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k=3, d=1, p=0.1):
            super().__init__()
            pad = (k-1)*d//2
            self.conv = nn.Conv1d(c_in, c_out, k, dilation=d, padding=pad)
            self.bn = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p)
            self.res = (c_in == c_out)
        def forward(self,x):
            y = self.conv(x); y = self.bn(y); y = self.act(y); y = self.drop(y)
            return x + y if self.res else y

    class Level1CNN(nn.Module):
        def __init__(self, in_features:int, channels:List[int]=(32,64,128), dropout:float=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                blocks.append(ConvBlock(chs[i], chs[i+1], k=3, d=2**i, p=dropout))
            self.blocks = nn.Sequential(*blocks)
            self.proj = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)
        def forward(self, x):
            # x: [B, F, T]
            z = self.blocks(x)
            z = self.proj(z)
            emb = z.mean(-1)
            logit = self.head(emb)
            return logit, emb

    class MLP(nn.Module):
        def __init__(self, in_dim:int, hidden:List[int], out_dim:int=1, dropout:float=0.1):
            super().__init__()
            layers=[]
            last=in_dim
            for h in hidden:
                layers += [nn.Linear(last,h), nn.ReLU(), nn.Dropout(dropout)]
                last=h
            layers += [nn.Linear(last,out_dim)]
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)
else:
    Level1CNN = None
    MLP = None

class TemperatureScaler:
    def __init__(self):
        self.mapper = None
    def fit(self, logits: np.ndarray, y: np.ndarray):
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds='clip')
            probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            iso.fit(probs, y)
            self.mapper = iso
        except Exception:
            self.mapper = None
    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.mapper is not None:
            probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            return self.mapper.transform(probs)
        else:
            return 1.0 / (1.0 + np.exp(-logits.reshape(-1)))

class CascadeTrainer:
    def __init__(self, seq_len:int=64, feat_windows:Tuple[int,...]=(5,10,20), device:str='auto'):
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.l1 = None
        self.l2_backend = None
        self.l2_model = None
        self.l3 = None
        self.l1_temp = TemperatureScaler()
        self.l3_temp = TemperatureScaler()
        self._fitted = False
        self.device = device

    def _to_sequences(self, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
        N, F = features.shape
        out = np.zeros((len(indices), self.seq_len, F), dtype=features.dtype)
        for i, t in enumerate(indices):
            t = int(t)
            t0 = t - self.seq_len + 1
            seq = features[max(0,t0):t+1]
            if seq.shape[0] < self.seq_len:
                pad = np.repeat(seq[[0]], self.seq_len - seq.shape[0], axis=0)
                seq = np.vstack([pad, seq])
            out[i] = seq[-self.seq_len:]
        return out

    def fit(self, bars: pd.DataFrame, events: pd.DataFrame, epochs_l1:int=12, epochs_l23:int=10, prefer_xgb:bool=True):
        if torch is None:
            raise RuntimeError("Torch required for CascadeTrainer.fit")
        close = bars['close'].astype(float)
        seq_cols = ['open','high','low','close','volume']
        seq_df = bars[seq_cols].fillna(method='ffill').fillna(0.0)
        eng = pd.DataFrame(index=bars.index)
        eng['ret1'] = close.pct_change().fillna(0.0)
        for w in self.feat_windows:
            eng[f'mom_{w}'] = (close - close.rolling(w).mean()).fillna(0.0)
            eng[f'vol_{w}'] = eng['ret1'].rolling(w).std().fillna(0.0)
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        X_seq_all = seq_df.values
        self.scaler_seq.fit(X_seq_all)
        X_seq_all_s = self.scaler_seq.transform(X_seq_all)
        X_tab_all = eng.values
        self.scaler_tab.fit(X_tab_all)
        X_tab_all_s = self.scaler_tab.transform(X_tab_all)
        Xseq = self._to_sequences(X_seq_all_s, idx)
        if len(idx) < 2:
            raise RuntimeError("Not enough events to train")
        strat = y if len(np.unique(y))>1 else None
        tr_i, va_i = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=42, stratify=strat)
        Xseq_tr = Xseq[tr_i]; Xseq_va = Xseq[va_i]
        y_tr = y[tr_i]; y_va = y[va_i]
        class SeqDS(torch.utils.data.Dataset):
            def __init__(self, X, y): self.X=X.astype('float32'); self.y=y.astype('float32').reshape(-1,1)
            def __len__(self): return len(self.X)
            def __getitem__(self, i):
                return self.X[i].transpose(1,0), self.y[i]
        ds_tr = SeqDS(Xseq_tr, y_tr)
        ds_va = SeqDS(Xseq_va, y_va)
        in_features = Xseq_tr.shape[2]
        if Level1CNN is None:
            raise RuntimeError("Torch Level1CNN not available")
        self.l1 = Level1CNN(in_features=in_features)
        self.l1 = self._train_torch(self.l1, ds_tr, ds_va, lr=1e-3, epochs=epochs_l1, patience=3)
        # infer embeddings
        def infer_l1_emb(Xseq_all):
            self.l1.eval()
            embs = []
            logits = []
            with torch.no_grad():
                for i in range(0, len(Xseq_all), 256):
                    sub = Xseq_all[i:i+256]
                    xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32)
                    logit, emb = self.l1(xb)
                    logits.append(logit.detach().cpu().numpy())
                    embs.append(emb.detach().cpu().numpy())
            return np.concatenate(logits, axis=0).reshape(-1,1), np.concatenate(embs, axis=0)
        all_idx_seq = self._to_sequences(X_seq_all_s, idx)
        l1_logits_all, l1_emb_all = infer_l1_emb(all_idx_seq)
        l1_emb_tr = l1_emb_all[tr_i]; l1_emb_va = l1_emb_all[va_i]
        Xtab_tr = X_tab_all_s[idx[tr_i]]
        Xtab_va = X_tab_all_s[idx[va_i]]
        X_l2_tr = np.hstack([l1_emb_tr, Xtab_tr])
        X_l2_va = np.hstack([l1_emb_va, Xtab_va])
        results = {}
        def train_l2_xgb():
            if xgb is None:
                raise RuntimeError("xgboost not available")
            clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_l2_tr, y_tr, eval_set=[(X_l2_va, y_va)], verbose=False)
            return ("xgb", clf)
        def train_l2_mlp():
            in_dim = X_l2_tr.shape[1]
            m = MLP(in_dim, [128,64], out_dim=1)
            ds2_tr = torch.utils.data.TensorDataset(torch.tensor(X_l2_tr, dtype=torch.float32), torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32))
            ds2_va = torch.utils.data.TensorDataset(torch.tensor(X_l2_va, dtype=torch.float32), torch.tensor(y_va.reshape(-1,1), dtype=torch.float32))
            m = self._train_torch(m, ds2_tr, ds2_va, lr=1e-3, epochs=epochs_l23, patience=3)
            return ("mlp", m)
        def train_l3_mlp():
            in_dim = X_l2_tr.shape[1]
            m3 = MLP(in_dim, [128,64], out_dim=1)
            ds3_tr = torch.utils.data.TensorDataset(torch.tensor(X_l2_tr, dtype=torch.float32), torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32))
            ds3_va = torch.utils.data.TensorDataset(torch.tensor(X_l2_va, dtype=torch.float32), torch.tensor(y_va.reshape(-1,1), dtype=torch.float32))
            m3 = self._train_torch(m3, ds3_tr, ds3_va, lr=1e-3, epochs=epochs_l23, patience=3)
            return ("l3", m3)
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            if (xgb is not None) and True:
                futures[ex.submit(train_l2_xgb)] = "l2"
            else:
                futures[ex.submit(train_l2_mlp)] = "l2"
            futures[ex.submit(train_l3_mlp)] = "l3"
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results[label] = fut.result()
                except Exception as e:
                    results[label] = None
        if results.get("l2") is not None:
            self.l2_backend, self.l2_model = results["l2"]
        if results.get("l3") is not None:
            self.l3 = results["l3"][1] if isinstance(results["l3"], tuple) else results["l3"]
        try:
            self.l1_temp.fit(l1_logits_all, y)
        except Exception:
            pass
        try:
            l3_val_logits = self._infer_l3_logits(X_l2_va)
            self.l3_temp.fit(l3_val_logits, y_va)
        except Exception:
            pass
        self._fitted = True
        return self

    def _train_torch(self, model, ds_tr, ds_va, lr=1e-3, epochs=10, patience=3):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        crit = nn.BCEWithLogitsLoss()
        tr_ld = torch.utils.data.DataLoader(ds_tr, batch_size=128, shuffle=True)
        va_ld = torch.utils.data.DataLoader(ds_va, batch_size=512, shuffle=False)
        best_loss = float("inf"); best_state = None; no_imp = 0
        for ep in range(epochs):
            model.train()
            for xb,yb in tr_ld:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                out = model(xb)
                logits = out if not isinstance(out, tuple) else out[0]
                loss = crit(logits, yb)
                loss.backward(); opt.step()
            model.eval()
            vloss = 0.0; n=0
            with torch.no_grad():
                for xb,yb in va_ld:
                    xb,yb = xb.to(dev), yb.to(dev)
                    out = model(xb)
                    logits = out if not isinstance(out, tuple) else out[0]
                    loss = crit(logits, yb)
                    vloss += float(loss.item())*len(xb); n += len(xb)
            val_loss = vloss / max(1,n)
            if val_loss + 1e-8 < best_loss:
                best_loss = val_loss
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _infer_l3_logits(self, X):
        if self.l3 is None:
            return np.zeros((len(X),1))
        self.l3.eval()
        logits=[]
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.tensor(X[i:i+2048], dtype=torch.float32)
                out = self.l3(xb)
                logits.append((out if not isinstance(out, tuple) else out[0]).detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1)

    def predict_batch(self, bars: pd.DataFrame, t_indices: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("CascadeTrainer not fitted")
        seq_cols = ['open','high','low','close','volume']
        seq_df = bars[seq_cols].fillna(method='ffill').fillna(0.0)
        eng = pd.DataFrame(index=bars.index)
        eng['ret1'] = seq_df['close'].pct_change().fillna(0.0)
        for w in self.feat_windows:
            eng[f'mom_{w}'] = (seq_df['close'] - seq_df['close'].rolling(w).mean()).fillna(0.0)
            eng[f'vol_{w}'] = eng['ret1'].rolling(w).std().fillna(0.0)
        X_seq_all_s = self.scaler_seq.transform(seq_df.values)
        X_tab_all_s = self.scaler_tab.transform(eng.values)
        Xseq = self._to_sequences(X_seq_all_s, t_indices)
        self.l1.eval()
        p1 = np.zeros(len(Xseq))
        l1_emb = np.zeros((len(Xseq), self.l1.proj.out_channels if hasattr(self.l1,'proj') else 128))
        with torch.no_grad():
            for i in range(0, len(Xseq), 256):
                sub = Xseq[i:i+256]
                xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32)
                out = self.l1(xb)
                logit, emb = out[0], out[1]
                p1[i:i+len(sub)] = 1.0/(1.0+np.exp(-logit.detach().cpu().numpy().reshape(-1)))
                l1_emb[i:i+len(sub)] = emb.detach().cpu().numpy()
        X_l2 = np.hstack([l1_emb, X_tab_all_s[t_indices]])
        if self.l2_backend == "xgb":
            try:
                p2 = self.l2_model.predict_proba(X_l2)[:,1]
            except Exception:
                p2 = np.zeros(len(X_l2))
        else:
            self.l2_model.eval()
            p2 = []
            with torch.no_grad():
                for i in range(0, len(X_l2), 4096):
                    xb = torch.tensor(X_l2[i:i+4096], dtype=torch.float32)
                    out = self.l2_model(xb)
                    p2.append(torch.sigmoid(out).cpu().numpy().reshape(-1))
            p2 = np.concatenate(p2, axis=0)
        go3 = p2 >= 0.5
        p3 = np.zeros_like(p1)
        if go3.any():
            X_l3 = X_l2[go3]
            l3_logits = self._infer_l3_logits(X_l3)
            p3_vals = 1.0/(1.0+np.exp(-l3_logits.reshape(-1)))
            p3[go3] = p3_vals
        return pd.DataFrame({"t": t_indices, "p1": p1, "p2": p2, "p3": p3})

# ------------------------------
# Finnage fetcher (primary) and Yahoo fallback
# ------------------------------
def fetch_finnage_bars(symbol_fin: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if not FINNAGE_API_KEY:
        logger.warning("FINNAGE_API_KEY not set; skipping Finnage fetch")
        return pd.DataFrame()
    try:
        url = ("https://finnhub.io/api/v1/stock/candle"
               f"?symbol={symbol_fin}&resolution=D&from={int(start_dt.timestamp())}&to={int(end_dt.timestamp())}&token={FINNAGE_API_KEY}")
        logger.debug("Finnage URL: %s", url)
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        if data.get('s') != 'ok':
            logger.debug("Finnage returned s=%s for %s", data.get('s'), symbol_fin)
            return pd.DataFrame()
        ts = [datetime.utcfromtimestamp(int(t)) for t in data['t']]
        df = pd.DataFrame({
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data.get('v', [0]*len(data['t']))
        }, index=pd.to_datetime(ts))
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        logger.exception("Finnage fetch error for %s: %s", symbol_fin, e)
        return pd.DataFrame()

def fetch_yahoo_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if YahooTicker is None:
        logger.warning("yahooquery not installed; cannot fetch Yahoo data")
        return pd.DataFrame()
    try:
        tq = YahooTicker(symbol)
        raw = tq.history(start=start_dt.date().isoformat(), end=end_dt.date().isoformat(), interval="1d")
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        raw = raw[~raw.index.duplicated(keep="first")]
        return raw
    except Exception as e:
        logger.exception("Yahoo fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()

# ------------------------------
# Supabase logging
# ------------------------------
def log_model_metrics_supabase(asset_group_name: str, ticker_stats: dict):
    if create_client is None:
        logger.warning("supabase-py not installed; skipping Supabase logging")
        return
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.exception("Failed to initialize Supabase client: %s", e)
        return
    correlation_id = str(uuid.uuid4())
    logger.info("Logging metrics to Supabase; correlation_id=%s", correlation_id)
    # aggregate
    keys = list(ticker_stats.keys())
    if not keys:
        logger.warning("No ticker stats to log")
        return
    metric_keys = list(ticker_stats[keys[0]].keys())
    group_metrics = {}
    for mk in metric_keys:
        try:
            group_metrics[mk] = float(np.nanmean([v.get(mk, np.nan) for v in ticker_stats.values()]))
        except Exception:
            group_metrics[mk] = None
    # insert per ticker
    for ticker, stats in ticker_stats.items():
        row = {
            "asset_group": asset_group_name,
            "ticker": ticker,
            "metrics_json": json.dumps(stats),
            "num_bars": int(stats.get("num_bars", 0)),
            "num_candidates": int(stats.get("num_candidates", 0)),
            "trained_at": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id
        }
        try:
            supabase.table("model_metrics").insert(row).execute()
            logger.debug("Inserted metrics for %s", ticker)
        except Exception as e:
            logger.exception("Supabase insert error for %s: %s", ticker, e)
    group_row = {
        "asset_group": asset_group_name,
        "ticker": None,
        "metrics_json": json.dumps(group_metrics),
        "num_bars": int(np.nanmean([v.get("num_bars", 0) for v in ticker_stats.values()])),
        "num_candidates": int(np.nanmean([v.get("num_candidates", 0) for v in ticker_stats.values()])),
        "trained_at": datetime.utcnow().isoformat(),
        "correlation_id": correlation_id
    }
    try:
        supabase.table("model_metrics").insert(group_row).execute()
        logger.debug("Inserted group aggregate row")
    except Exception as e:
        logger.exception("Supabase insert failed for group row: %s", e)

# ------------------------------
# Full pipeline orchestration
# ------------------------------
def run_asset_group_pipeline(group_name: str, tickers_list: List[str], start_date_local: datetime.date, end_date_local: datetime.date):
    logger.info("Starting pipeline for group=%s start=%s end=%s", group_name, start_date_local, end_date_local)
    st.info(f"Starting pipeline for {group_name} — {len(tickers_list)} tickers")
    all_bars = []
    ticker_stats = {}
    for sym in tickers_list:
        st.write(f"Fetching {sym}")
        fin_map = yahoo_to_finage_mapping.get(group_name, {}).get(sym)
        df_finnage = pd.DataFrame()
        if fin_map:
            df_finnage = fetch_finnage_bars(fin_map, datetime.combine(start_date_local, datetime.min.time()), datetime.combine(end_date_local, datetime.min.time()))
        df_yahoo = fetch_yahoo_bars(sym, datetime.combine(start_date_local, datetime.min.time()), datetime.combine(end_date_local, datetime.min.time()))
        # combine preference: prefer Finnage but merge gaps
        if not df_finnage.empty:
            if df_yahoo.empty:
                df = df_finnage
            else:
                # combine, prioritize Finnage (place Finnage first then yahoo, drop duplicates keep first)
                df = pd.concat([df_finnage, df_yahoo]).sort_index().drop_duplicates(keep='first')
        else:
            df = df_yahoo
        if df is None or df.empty:
            logger.warning("No bars for %s from either source", sym)
            continue
        # normalize index tz and columns
        df.index = pd.to_datetime(df.index).tz_convert(None) if hasattr(df.index, 'tz') else pd.to_datetime(df.index)
        for col in ('open','high','low','close','volume'):
            if col not in df.columns:
                df[col] = 0.0
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.loc[:, 'symbol'] = sym
        all_bars.append(df)
        logger.info("Fetched %d bars for %s", len(df), sym)
    if not all_bars:
        st.error("No data fetched for any tickers in this group.")
        return None
    combined_bars = pd.concat(all_bars, axis=0).sort_index()
    logger.info("Combined bars total rows=%d", len(combined_bars))
    # generate candidates and events per ticker
    events_frames = []
    combined_candidates = []
    for sym in tickers_list:
        bars_sym = combined_bars[combined_bars['symbol'] == sym]
        if bars_sym.empty:
            logger.warning("Empty bars for %s; skipping", sym)
            continue
        # compute candidates using rule-based generator
        cands = generate_candidates_and_labels(bars_sym)
        if cands is None or cands.empty:
            logger.info("No candidates for %s", sym)
            continue
        cands.loc[:, 'symbol'] = sym
        combined_candidates.append(cands)
        # map to events
        ev, idx_map = create_events_from_candidates(cands, bars_sym)
        if ev is None or ev.empty:
            logger.info("No events mapped for %s", sym)
            continue
        # adjust event indices to global combined_bars index
        # we need mapping from bars_sym index -> global index
        global_idx_map = {t:i for i,t in enumerate(combined_bars.index)}
        ev['t'] = ev['t'].apply(lambda local_idx: global_idx_map.get(bars_sym.index[local_idx], -1) if (0 <= local_idx < len(bars_sym)) else -1)
        ev = ev[ev['t'] >= 0]
        events_frames.append(ev)
        ticker_stats[sym] = {
            "num_bars": int(len(bars_sym)),
            "num_candidates": int(len(cands)),
            "val_acc": float(np.nan),
            "breadth_score": float(np.nan)
        }
        logger.info("Ticker %s: bars=%d candidates=%d events=%d", sym, len(bars_sym), len(cands), len(ev))
    if not events_frames:
        st.error("No events produced for any ticker.")
        return None
    events = pd.concat(events_frames, ignore_index=True)
    # Train Cascade
    if CascadeTrainer is None:
        st.error("CascadeTrainer not available (torch must be installed)")
        return None
    try:
        trainer = CascadeTrainer(seq_len=seq_len, feat_windows=(5,10,20), device=device_choice)
        trainer.fit(combined_bars, events, epochs_l1=epochs_l1, epochs_l23=epochs_l23, prefer_xgb=(xgb is not None))
        st.success("Training completed")
    except Exception as e:
        logger.exception("Training failed: %s", e)
        st.error(f"Training failed: {e}")
        return None
    # Predictions for verification
    try:
        t_indices = events['t'].astype(int).values
        preds = trainer.predict_batch(combined_bars, t_indices)
        st.write("Prediction sample:")
        st.dataframe(preds.head(10))
        logger.info("Predictions computed size=%s", preds.shape)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        preds = pd.DataFrame()
    # Export model
    try:
        out_dir = Path(f"artifacts_{group_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = out_dir / f"{group_name}_cascade.pt"
        # save via torch
        if torch is not None:
            torch.save(trainer, str(pt_path))
            st.success(f"Saved model to {pt_path}")
            logger.info("Saved .pt at %s", pt_path)
        else:
            pt_path = None
            logger.warning("Torch not available; cannot save .pt")
    except Exception as e:
        logger.exception("Export failed: %s", e)
        pt_path = None
    # Breadth backtest
    if run_breadth and not preds.empty:
        try:
            # define levels (example)
            levels = [
                {"name":"L1","buy_min":7.0,"buy_max":10.0,"sl":0.02,"rr":2.0,"max_holding":60},
                {"name":"L2","buy_min":5.0,"buy_max":6.9,"sl":0.02,"rr":1.5,"max_holding":40},
            ]
            breadth_res = run_breadth_levels(preds, pd.concat(combined_candidates, ignore_index=True), combined_bars, levels, simulate_limits_fn)
            st.subheader("Breadth summary")
            st.write(pd.DataFrame(breadth_res.get("summary", [])))
            # store summarised metrics per ticker
            for sym in ticker_stats.keys():
                # compute simple breadth_score as average p3 across that tickers candidates
                cand_mask = pd.concat(combined_candidates, ignore_index=True)['symbol'] == sym
                if cand_mask.any():
                    # map p3 values
                    # align preds.t to candidates order is assumed; approximate by mean of preds p3 for that ticker
                    try:
                        # find t indices for that ticker
                        cand_df = pd.concat(combined_candidates, ignore_index=True)
                        cand_t_indices = cand_df[cand_df['symbol']==sym].index
                        p3_vals = preds.loc[preds.index.isin(cand_t_indices), 'p3'] if 'p3' in preds.columns else preds['p3']
                        ticker_stats[sym]['breadth_score'] = float(np.nanmean(p3_vals)) if len(p3_vals)>0 else float(np.nan)
                    except Exception:
                        ticker_stats[sym]['breadth_score'] = float(np.nan)
            logger.info("Breadth backtest finished")
        except Exception as e:
            logger.exception("Breadth backtest failed: %s", e)
    # Evaluate val_acc placeholder: if preds and events have labels, compute simple metric
    try:
        if not preds.empty and 'p3' in preds.columns and 'y' in events.columns:
            merged = events.reset_index(drop=True).merge(preds.reset_index(drop=True), left_index=True, right_index=True, how='left')
            for sym in ticker_stats.keys():
                m = merged[merged['symbol']==sym] if 'symbol' in merged.columns else merged
                if m.empty: continue
                try:
                    pred_label = (m['p3'] >= 0.65).astype(int)
                    acc = float((pred_label == m['y']).mean())
                    ticker_stats[sym]['val_acc'] = acc
                except Exception:
                    ticker_stats[sym]['val_acc'] = float(np.nan)
    except Exception:
        pass
    # HF upload
    hf_url = None
    if HF_TOKEN and repo_name and pt_path is not None:
        try:
            hf_url = hf_upload(str(pt_path), repo_name, HF_TOKEN)
            if hf_url:
                st.success("Uploaded to Hugging Face")
        except Exception as e:
            logger.exception("HF upload failed: %s", e)
    # Supabase logging
    try:
        log_model_metrics_supabase(group_name, ticker_stats)
        st.success("Metrics logged to Supabase (if client available)")
    except Exception as e:
        logger.exception("Supabase logging failed: %s", e)
    return {"pt_path": str(pt_path) if pt_path else None, "hf_url": hf_url, "ticker_stats": ticker_stats}

# ------------------------------
# Streamlit run button
# ------------------------------
if st.sidebar.button("Run Finnage-primary Asset Group Training (1d, 1y)"):
    result = run_asset_group_pipeline(asset_group, tickers, start_date, end_date)
    if result:
        st.write("Model export path:", result.get("pt_path"))
        if result.get("hf_url"):
            st.write("Hugging Face URL:", result.get("hf_url"))
        st.subheader("Ticker stats")
        st.dataframe(pd.DataFrame(result.get("ticker_stats", {})).T)

# End of script

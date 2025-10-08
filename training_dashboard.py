import os
import sys
import gc
import logging
import joblib
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ML imports
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Data fetching
from yahooquery import Ticker

# Supabase wrapper
from supabase_client_wrapper import SupabaseClientWrapper

# -----------------------------------------------------
# Logging setup
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cascade_trader")

# -----------------------------------------------------
# Supabase client initialization
# -----------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dzddytphimhoxeccxqsw.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client = None
if SUPABASE_SERVICE_ROLE_KEY:
    supabase_client = SupabaseClientWrapper(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    logger.info("Supabase client initialized for stats logging")
else:
    logger.warning("Supabase service role key not set. Stats logging disabled.")

# -----------------------------------------------------
# Streamlit App UI - Sidebar Controls
# -----------------------------------------------------
st.set_page_config(page_title="Cascade Trader", layout="wide")

st.sidebar.header("Cascade Trader Controls")

# Date selection
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=180))

# Asset ticker
symbol = st.sidebar.text_input("Asset Ticker", "AAPL")

# Interval
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "30m", "15m", "5m", "1m"],
    index=0
)

# Buy/Sell thresholds per layer
st.sidebar.subheader("Layer Thresholds (Buy)")
lvl1_buy_min = st.sidebar.slider("L1 Buy Min", 0.0, 10.0, 3.0, 0.1)
lvl1_buy_max = st.sidebar.slider("L1 Buy Max", 0.0, 10.0, 10.0, 0.1)
lvl2_buy_min = st.sidebar.slider("L2 Buy Min", 0.0, 10.0, 5.5, 0.1)
lvl2_buy_max = st.sidebar.slider("L2 Buy Max", 0.0, 10.0, 10.0, 0.1)
lvl3_buy_min = st.sidebar.slider("L3 Buy Min", 0.0, 10.0, 6.5, 0.1)
lvl3_buy_max = st.sidebar.slider("L3 Buy Max", 0.0, 10.0, 10.0, 0.1)

st.sidebar.subheader("Layer Thresholds (Sell)")
lvl1_sell_min = st.sidebar.slider("L1 Sell Min", 0.0, 10.0, 0.0, 0.1)
lvl1_sell_max = st.sidebar.slider("L1 Sell Max", 0.0, 10.0, 7.0, 0.1)
lvl2_sell_min = st.sidebar.slider("L2 Sell Min", 0.0, 10.0, 0.0, 0.1)
lvl2_sell_max = st.sidebar.slider("L2 Sell Max", 0.0, 10.0, 5.0, 0.1)
lvl3_sell_min = st.sidebar.slider("L3 Sell Min", 0.0, 10.0, 0.0, 0.1)
lvl3_sell_max = st.sidebar.slider("L3 Sell Max", 0.0, 10.0, 4.0, 0.1)

# Pipeline control buttons
st.sidebar.subheader("Pipeline Execution")
run_full_pipeline_btn = st.sidebar.button("Run Full Pipeline (Fetch → Train → Export)")

# ---------------- Chunk 2/5 ----------------
# engineered-feature helpers

def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute a compact set of engineered features from OHLCV."""
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)

    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build sequences ending at each index t: [t-seq_len+1, ..., t].
    Returns shape [N, seq_len, F]
    """
    Nrows, F = features.shape
    X = np.zeros((len(indices), seq_len, F), dtype=features.dtype)
    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1
        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0:t+1]])
        else:
            seq = features[t0:t+1]
        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])
        X[i] = seq[-seq_len:]
    return X


# torch Datasets (only if torch is available)
if torch is not None:
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        def __init__(self, X_seq: np.ndarray, y: np.ndarray):
            self.X = X_seq.astype(np.float32)  # [N, T, F]
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].transpose(1,0)  # [F, T]
            y = self.y[idx]
            return x, y

    class TabDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
else:
    SequenceDataset = None
    TabDataset = None


# utilities: ensure unique index
def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


# candidate generation
def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long"
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    for col in ("high", "low", "close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")
    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(window=atr_window, min_periods=1).mean()
    recs = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr = float(bars["atr"].iat[i])
        if atr <= 0 or math.isnan(atr):
            continue
        sl_px = entry_px - k_sl * atr if direction == "long" else entry_px + k_sl * atr
        tp_px = entry_px + k_tp * atr if direction == "long" else entry_px - k_tp * atr
        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_px = float(bars["close"].iat[end_idx])
        for j in range(i + 1, end_idx + 1):
            hi = float(bars["high"].iat[j]); lo = float(bars["low"].iat[j])
            if hi >= tp_px:
                label, hit_idx, hit_px = 1, j, tp_px
                break
            if lo <= sl_px:
                label, hit_idx, hit_px = 0, j, sl_px
                break
        end_t = bars.index[hit_idx]
        realized_return = (hit_px - entry_px) / entry_px
        dur_min = (end_t - t).total_seconds() / 60.0
        recs.append({
            "candidate_time": t,
            "entry_price": float(entry_px),
            "atr": float(atr),
            "sl_price": float(sl_px),
            "tp_price": float(tp_px),
            "end_time": end_t,
            "label": int(label),
            "duration": float(dur_min),
            "realized_return": float(realized_return),
            "direction": "long"
        })
    return pd.DataFrame(recs)


# simulate limits + summarization
def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str = "GC=F",
                    sl: float = 0.02,
                    tp: float = 0.04,
                    max_holding: int = 60) -> pd.DataFrame:
    if df is None or df.empty or bars is None or bars.empty:
        return pd.DataFrame()
    trades = []
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    for _, row in df.iterrows():
        lbl = int(row.get(label_col, 0))
        if lbl == 0:
            continue
        entry_t = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_t not in bars.index:
            continue
        entry_px = float(bars.loc[entry_t, "close"])
        direction = 1 if lbl > 0 else -1
        sl_px = entry_px * (1 - sl) if direction > 0 else entry_px * (1 + sl)
        tp_px = entry_px * (1 + tp) if direction > 0 else entry_px * (1 - tp)
        exit_t, exit_px, pnl = None, None, None
        segment = bars.loc[entry_t:].head(max_holding)
        if segment.empty:
            continue
        for t, b in segment.iterrows():
            lo, hi = float(b["low"]), float(b["high"])
            if direction > 0:
                if lo <= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl
                    break
                if hi >= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp
                    break
            else:
                if hi >= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl
                    break
                if lo <= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp
                    break
        if exit_t is None:
            last_bar = segment.iloc[-1]
            exit_t = last_bar.name
            exit_px = float(last_bar["close"])
            pnl = (exit_px - entry_px) / entry_px * direction
        trades.append({
            "symbol": symbol,
            "entry_time": entry_t,
            "entry_price": entry_px,
            "direction": direction,
            "exit_time": exit_t,
            "exit_price": exit_px,
            "pnl": float(pnl)
        })
    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    total_trades = len(trades)
    win_rate = float((trades["pnl"] > 0).mean())
    avg_pnl = float(trades["pnl"].mean())
    median_pnl = float(trades["pnl"].median())
    total_pnl = float(trades["pnl"].sum())
    max_dd = float(trades["pnl"].cumsum().min())
    start_time = trades["entry_time"].min()
    end_time = trades["exit_time"].max()
    return pd.DataFrame([{
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "start_time": start_time,
        "end_time": end_time
    }])

# ---------------- Chunk 3/5 ----------------
# Model blocks (if torch available)
if torch is not None:
    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k, d, pdrop):
            super().__init__()
            pad = (k - 1) * d // 2
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
            self.bn = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(pdrop)
            self.res = (c_in == c_out)
        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
            out = self.drop(out)
            if self.res:
                out = out + x
            return out

    class Level1ScopeCNN(nn.Module):
        def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                k = kernel_sizes[min(i, len(kernel_sizes)-1)]
                d = dilations[min(i, len(dilations)-1)]
                blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
            self.blocks = nn.Sequential(*blocks)
            self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)
        @property
        def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
        def forward(self, x):
            z = self.blocks(x)
            z = self.project(z)
            z_pool = z.mean(dim=-1)
            logit = self.head(z_pool)
            return logit, z_pool

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim=1, dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
                last = h
            layers += [nn.Linear(last, out_dim)]
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    class Level3ShootMLP(nn.Module):
        def __init__(self, in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True):
            super().__init__()
            self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
            self.cls_head = nn.Linear(128, 1)
            self.reg_head = nn.Linear(128, 1) if use_regression_head else None
        def forward(self, x):
            h = self.backbone(x)
            logit = self.cls_head(h)
            ret = self.reg_head(h) if self.reg_head is not None else None
            return logit, ret

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = nn.Parameter(torch.zeros(1))
        def forward(self, logits):
            T = torch.exp(self.log_temp)
            return logits / T
        def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200, lr=1e-2):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
            y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=device)
            opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
            bce = nn.BCEWithLogitsLoss()
            def closure():
                opt.zero_grad()
                scaled = self.forward(logits_t)
                loss = bce(scaled, y_t)
                loss.backward()
                return loss
            try:
                opt.step(closure)
            except Exception as e:
                logger.warning("Temp scaler LBFGS failed: %s", e)
        def transform(self, logits: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                device = next(self.parameters()).device
                logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
                scaled = self.forward(logits_t).cpu().numpy()
            return scaled.reshape(-1)


# ------------ Module-level simple wrapper classes for export (fixes pickling) ------------
class L2ExportWrapper:
    """Simple export wrapper for L2 models (xgb or mlp)."""
    def __init__(self):
        self.booster = None
    def save_model(self, path: str):
        if hasattr(self.booster, "save_model"):
            return self.booster.save_model(path)
        joblib.dump(self.booster, path)
    def feature_importance(self):
        try:
            if hasattr(self.booster, "get_booster"):
                imp = self.booster.get_booster().get_score(importance_type="gain")
            elif hasattr(self.booster, "get_score"):
                imp = self.booster.get_score(importance_type="gain")
            else:
                return pd.DataFrame([{"feature":"none","gain":0.0}])
            return (pd.DataFrame([(k, imp.get(k,0.0)) for k in sorted(imp.keys())], columns=["feature","gain"])
                    .sort_values("gain", ascending=False).reset_index(drop=True))
        except Exception:
            return pd.DataFrame([{"feature":"none","gain":0.0}])


class L3ExportWrapper:
    """Export wrapper for L3 torch models."""
    def __init__(self):
        self.model = None
    def feature_importance(self):
        # placeholder; real importance should be computed separately
        return pd.DataFrame([{"feature":"l3_emb","gain":1.0}])

# ---------------- Chunk 4/5 ----------------
# training helpers
def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def train_torch_classifier(model: nn.Module,
                           train_ds,
                           val_ds,
                           lr: float = 1e-3,
                           epochs: int = 10,
                           patience: int = 3,
                           pos_weight: float = 1.0,
                           device: str = "auto",
                           st_progress: Optional[st.delta_generator.Progress] = None,
                           progress_offset: int = 0,
                           progress_total: int = 100):
    """
    Train with optional Streamlit progress updater.
    progress_total is total units to represent; the function will update progress_offset..progress_offset+progress_total
    """
    dev = _device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor([pos_weight], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 128), shuffle=True)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=getattr(val_ds, "batch_size", 1024), shuffle=False)
    best_loss = float("inf"); best_state = None; no_imp = 0
    history = {"train": [], "val": []}
    total_steps = epochs
    for ep in range(epochs):
        model.train()
        running_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / max(1, n)
        # val
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                vloss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
        val_loss = vloss / max(1, vn)
        history["train"].append(train_loss); history["val"].append(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_

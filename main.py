# ===============================
# CHUNK 1/5: IMPORTS, CONFIG, UI
# ===============================
import os, time, json, logging, traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
import aiohttp
import asyncio
import joblib

# ML libs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import xgboost as xgb
except Exception:
    xgb = None

# optional: yahooquery
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

# supabase wrapper (user provided)
try:
    from supabase_client_wrapper import SupabaseClientWrapper
except Exception as e:
    SupabaseClientWrapper = None
    logging.warning("Supabase wrapper import failed: %s", e)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger("single_layer_cascade")

# Streamlit page
st.set_page_config(page_title="Single-Model Cascade Trainer", layout="wide")
st.title("Single-model Cascade Trainer — Scope → Aim → Shoot (combined)")

# Sidebar: data and model config
st.sidebar.header("Data sources & range")
symbol_futures = st.sidebar.text_input("Futures symbol (Yahoo)", value="GC=F")
symbol_cfd     = st.sidebar.text_input("CFD symbol (Finage)", value="GCUSD")
default_days = 180
start_date = st.sidebar.date_input("Start date", value=(datetime.utcnow() - timedelta(days=default_days)).date())
end_date   = st.sidebar.date_input("End date",   value=datetime.utcnow().date())
interval = "1d"  # fixed per your requirement

st.sidebar.header("Finage (CFD) settings")
finage_base = st.sidebar.text_input("Finage base URL", value=os.getenv("FINAGE_BASE_URL", "https://api.finage.co.uk"))
finage_api_key = st.sidebar.text_input("Finage API key", value=os.getenv("FINAGE_API_KEY", ""))

st.sidebar.header("Model / training")
prefer_xgb = st.sidebar.checkbox("Prefer XGBoost if installed", value=True)
test_size   = float(st.sidebar.slider("Validation fraction", 0.05, 0.4, 0.2))
random_seed = int(st.sidebar.number_input("Random seed", value=42))
train_btn = st.sidebar.button("Fetch → Label → Train (single model)")

# Supabase client init (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL","")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY","")
sb_client: Optional[SupabaseClientWrapper] = None
if SupabaseClientWrapper and SUPABASE_URL and SUPABASE_KEY:
    try:
        sb_client = SupabaseClientWrapper(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.warning("Supabase client init failed: %s", e)

# ============================================
# CHUNK 2/5: DATA FETCH + FEATURES + CANDIDATES
# ============================================

# Yahoo fetch (sync wrapped)
def safe_fetch_yahoo(symbol: str, start: str, end: str, interval: str="1d") -> pd.DataFrame:
    if YahooTicker is None:
        logger.warning("yahooquery missing")
        return pd.DataFrame()
    try:
        t = YahooTicker(symbol)
        raw = t.history(start=start, end=end, interval=interval)
        if raw is None:
            return pd.DataFrame()
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw and "adjclose" in raw:
            raw["close"] = raw["adjclose"]
        df = raw[~raw.index.duplicated(keep="first")].sort_index()
        return df
    except Exception as e:
        logger.exception("Yahoo fetch failed: %s", e)
        return pd.DataFrame()

# Finage async fetch for Forex aggregates (1d)
async def fetch_cfd_finage(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    url = f"{finage_base.rstrip('/')}/agg/forex/{symbol}/1/day/{start}/{end}?apikey={api_key}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    logger.warning("Finage returned status %s for %s", resp.status, url)
                    return pd.DataFrame()
                j = await resp.json()
                results = j.get("results") if isinstance(j, dict) else j
                if not results:
                    return pd.DataFrame()
                df = pd.DataFrame(results)
                # Finage returns 't' as epoch ms in example — try both ms and s
                if "t" in df.columns:
                    # try ms first
                    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", errors="coerce")
                    if df["timestamp"].isna().all():
                        df["timestamp"] = pd.to_datetime(df["t"], unit="s", errors="coerce")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.set_index("timestamp")
                colmap = {c: c.lower() for c in df.columns}
                df.rename(columns=colmap, inplace=True)
                # map expected names
                rename = {}
                for c in df.columns:
                    if c in ("o","open"): rename[c]="open"
                    if c in ("h","high"): rename[c]="high"
                    if c in ("l","low"): rename[c]="low"
                    if c in ("c","close","close_price"): rename[c]="close"
                    if c=="v" or "volume" in c: rename[c]="volume"
                df.rename(columns=rename, inplace=True)
                # ensure at least close exists
                if "close" not in df.columns:
                    poss = [c for c in df.columns if "price" in c or "close" in c]
                    if poss:
                        df["close"] = df[poss[0]]
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="first")]
                return df
        except Exception as e:
            logger.exception("Finage fetch failed: %s", e)
            return pd.DataFrame()

# concurrent gather wrapper using asyncio
async def _fetch_concurrent_async(fut_sym, cfd_sym, start_iso, end_iso, api_key):
    fut_task = asyncio.to_thread(safe_fetch_yahoo, fut_sym, start_iso, end_iso, "1d")
    cfd_task = fetch_cfd_finage(cfd_sym, start_iso, end_iso, api_key)
    fut_df, cfd_df = await asyncio.gather(fut_task, cfd_task)
    return fut_df, cfd_df

def fetch_both(symbol_fut, symbol_cfd, start_iso, end_iso, api_key):
    return asyncio.run(_fetch_concurrent_async(symbol_fut, symbol_cfd, start_iso, end_iso, api_key))

# engineered features (from Yahoo / CFD close series)
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float) if 'high' in df.columns else c
    l = df['low'].astype(float) if 'low' in df.columns else c
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)
    f['ret1'] = c.pct_change().fillna(0.0)
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        f[f'vol_{w}'] = f['ret1'].rolling(w).std().fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min)/denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# candidate labels produced from futures (triple-barrier style)
def generate_candidates_and_labels_from_futures(bars: pd.DataFrame, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60):
    if bars is None or bars.empty: return pd.DataFrame()
    bars = bars.copy(); bars.index = pd.to_datetime(bars.index)
    bars['tr'] = (bars['high'] - bars['low']).abs().fillna(0.0)
    bars['atr'] = bars['tr'].rolling(atr_window, min_periods=1).mean()
    recs=[]
    for i in range(lookback, len(bars)):
        t = bars.index[i]; px = float(bars.close.iat[i]); atr=float(bars.atr.iat[i])
        if atr<=0 or np.isnan(atr): continue
        sl_px = px - k_sl*atr; tp_px = px + k_tp*atr
        end_i = min(i+max_bars, len(bars)-1)
        label=0; hit_i=end_i
        for j in range(i+1, end_i+1):
            hi = float(bars.high.iat[j]); lo=float(bars.low.iat[j])
            if hi>=tp_px: label=1; hit_i=j; break
            if lo<=sl_px: label=0; hit_i=j; break
        recs.append({"candidate_time": t, "label": int(label), "entry_px": px, "hit_idx": hit_i})
    return pd.DataFrame(recs)

# =====================================================
# CHUNK 3/5: ALIGN LABELS → BUILD COMBINED FEATURES → TRAIN
# =====================================================

def align_labels_to_cfd(candidates: pd.DataFrame, cfd_bars: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Map each candidate_time into an integer index in cfd_bars (nearest prior)."""
    if candidates is None or candidates.empty or cfd_bars is None or cfd_bars.empty:
        return pd.DataFrame(), {}
    idx_map = {t:i for i,t in enumerate(cfd_bars.index)}
    t_indices=[]
    for t in pd.to_datetime(candidates['candidate_time']):
        if t in idx_map:
            t_indices.append(idx_map[t])
        else:
            locs = cfd_bars.index[cfd_bars.index <= t]
            t_indices.append(idx_map[locs[-1]] if len(locs) else 0)
    events = pd.DataFrame({"t": np.array(t_indices, dtype=int), "y": candidates['label'].astype(int).values})
    return events, {"map": idx_map}

def build_scope_aim_shoot_features(futures_bars: pd.DataFrame, cfd_bars: pd.DataFrame, events: pd.DataFrame):
    """
    Build a combined feature matrix per event index:
    - Scope (L1): sequence-level aggregated statistics from futures features
    - Aim   (L2): tabular features from futures engineered features
    - Shoot (L3): execution-related micro-features from CFD (slippage, volume)
    Returns X (2D numpy), y (labels), index mapping.
    """
    # features from futures (labels source)
    fut_eng = compute_engineered_features(futures_bars)
    cfd_eng = compute_engineered_features(cfd_bars)
    # For scope, compute short sequence aggregates around candidate times on futures
    def seq_aggregates(df, idx, window=10):
        out=[]
        arr = df['ret1'].values if 'ret1' in df.columns else np.zeros(len(df))
        for t in idx:
            t = int(t)
            start = max(0, t-window+1); seq = arr[start:t+1]
            out.append([seq.mean(), seq.std(), seq[-1] if len(seq) else 0.0])
        return np.array(out)
    # event indices refer to CFD positions; need to map back to futures time if possible
    # We'll derive features using futures indices nearest by timestamp
    fut_idx_map = {t:i for i,t in enumerate(futures_bars.index)}
    mapped_fut_indices=[]
    for ev_t in events['t']:
        cfd_t = cfd_bars.index[int(ev_t)]
        # find nearest futures index <= cfd_t
        locs = futures_bars.index[futures_bars.index <= cfd_t]
        mapped_fut_indices.append(fut_idx_map[locs[-1]] if len(locs) else 0)
    mapped_fut_indices = np.array(mapped_fut_indices, dtype=int)
    # build scope (sequence aggregates on futures)
    scope_feats = seq_aggregates(fut_eng, mapped_fut_indices, window=8)
    # aim: take tabular engineered features from futures at mapped indices
    aim_cols = [c for c in fut_eng.columns if c.startswith(("mom_","vol_","chanpos_"))]
    aim_feats = fut_eng.iloc[mapped_fut_indices][aim_cols].fillna(0.0).values if len(aim_cols)>0 else np.zeros((len(events),0))
    # shoot: micro execution features from CFD at event index (e.g., volume, recent volatility)
    shoot_cols = []
    shoot_feats=[]
    for t in events['t'].astype(int).values:
        # use lookback returns and volume
        window = 5
        start = max(0, t-window+1)
        seg = cfd_bars.iloc[start:t+1]
        if seg.empty:
            shoot_feats.append([0.0,0.0])
        else:
            v_mean = seg['volume'].astype(float).mean() if 'volume' in seg.columns else 0.0
            ret_mean = seg['close'].pct_change().fillna(0.0).mean()
            shoot_feats.append([v_mean, ret_mean])
    shoot_feats = np.array(shoot_feats)
    # combine all pieces
    X = np.hstack([scope_feats, aim_feats, shoot_feats])
    y = events['y'].astype(int).values
    return X, y, {"mapped_fut_indices": mapped_fut_indices}

def train_single_model_on_combined_features(X: np.ndarray, y: np.ndarray, prefer_xgb=True, test_size=0.2, seed=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_va, y_tr, y_va = train_test_split(Xs, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y))>1 else None)
    model = None; metrics={}
    if prefer_xgb and xgb is not None:
        try:
            clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=seed)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            model = clf
            yhat = clf.predict_proba(X_va)[:,1]
            metrics['auc'] = float(roc_auc_score(y_va, yhat)) if len(np.unique(y_va))>1 else None
        except Exception as e:
            logger.exception("xgb training failed: %s", e)
            model = None
    if model is None:
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_tr, y_tr)
        model = clf
        try:
            yhat = clf.predict_proba(X_va)[:,1]
            metrics['auc'] = float(roc_auc_score(y_va, yhat)) if len(np.unique(y_va))>1 else None
        except Exception:
            metrics['auc'] = None
    ypred = (model.predict_proba(X_va)[:,1] >= 0.5).astype(int)
    metrics['accuracy'] = float(accuracy_score(y_va, ypred))
    return {"model": model, "scaler": scaler, "metrics": metrics}

# =====================================================
# CHUNK 4/5: SIMULATE LIMITS, SUMMARIZE, SAVE, EXPORT
# =====================================================

def simulate_limits(df: pd.DataFrame, bars: pd.DataFrame, label_col="pred_label", sl=0.02, tp=0.04, max_holding=60, symbol="CFD"):
    if df is None or df.empty or bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars.copy(); bars.index = pd.to_datetime(bars.index)
    trades=[]
    for _, r in df.iterrows():
        if int(r.get(label_col,0))==0: continue
        entry_t = pd.to_datetime(r.get("candidate_time", r.name))
        if entry_t not in bars.index: continue
        entry_px = float(bars.loc[entry_t,"close"])
        sl_px = entry_px*(1 - sl)
        tp_px = entry_px*(1 + tp)
        exit_t=None; exit_px=None; pnl=None
        seg = bars.loc[entry_t:].head(max_holding)
        for t,b in seg.iterrows():
            if float(b["low"])<=sl_px:
                exit_t, exit_px, pnl = t, sl_px, -sl; break
            if float(b["high"])>=tp_px:
                exit_t, exit_px, pnl = t, tp_px, tp; break
        if exit_t is None:
            last = seg.iloc[-1]
            exit_t = last.name; exit_px=float(last["close"]); pnl=(exit_px-entry_px)/entry_px
        trades.append({"symbol": symbol, "entry_time": entry_t, "entry_price": entry_px, "exit_time": exit_t, "exit_price": exit_px, "pnl": float(pnl)})
    return pd.DataFrame(trades)

def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty: return pd.DataFrame()
    total = len(trades)
    win_rate = float((trades.pnl>0).mean())
    avg = float(trades.pnl.mean())
    med = float(trades.pnl.median())
    total_pnl = float(trades.pnl.sum())
    max_dd = float(trades.pnl.cumsum().min())
    return pd.DataFrame([{"total_trades": total, "win_rate": win_rate, "avg_pnl": avg, "median_pnl": med, "total_pnl": total_pnl, "max_drawdown": max_dd}])

def save_layer_summary(sb: Optional[SupabaseClientWrapper], symbol: str, summary_rows: List[Dict[str,Any]]):
    if sb is None or not summary_rows: return
    now = datetime.utcnow().isoformat()
    rows=[]
    for r in summary_rows:
        rows.append({
            "symbol": symbol,
            "mode": r.get("mode","single"),
            "total_trades": int(r.get("total_trades") or 0),
            "win_rate": float(r.get("win_rate") or 0.0),
            "avg_pnl": float(r.get("avg_pnl") or 0.0),
            "median_pnl": float(r.get("median_pnl") or 0.0),
            "total_pnl": float(r.get("total_pnl") or 0.0),
            "max_drawdown": float(r.get("max_drawdown") or 0.0),
            "start_time": str(r.get("start_time")) if r.get("start_time") is not None else None,
            "end_time": str(r.get("end_time")) if r.get("end_time") is not None else None,
            "created_at": now
        })
    try:
        resp = sb.insert_data("layer_summary", rows)
        if not resp.get("success"):
            logger.warning("Supabase insert non-success: %s", resp)
    except Exception as e:
        logger.exception("Supabase save error: %s", e)

def export_artifact(model_obj: Any, scaler: Any, metrics: Dict[str,Any], basename="single_cascade"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = f"{basename}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump({"model":model_obj, "scaler":scaler, "metrics":metrics}, model_path)
    return {"out_dir": out_dir, "model_path": model_path}
# ====================================================
# CHUNK 5/5: STREAMLIT PIPELINE ORCHESTRATION & OUTPUT
# ====================================================

status = st.empty()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Futures (label source)")
    st.write(symbol_futures)
with col2:
    st.subheader("CFD (training / execution)")
    st.write(symbol_cfd)

def run_pipeline_single_model():
    status.info("Fetching data concurrently...")
    start_iso = pd.Timestamp(start_date).isoformat()
    end_iso = pd.Timestamp(end_date).isoformat()
    fut_df, cfd_df = fetch_both(symbol_futures, symbol_cfd, start_iso, end_iso, finage_api_key)

    if fut_df is None or fut_df.empty:
        status.error("Futures (Yahoo) fetch failed or returned no data.")
        return
    if cfd_df is None or cfd_df.empty:
        status.error("CFD (Finage) fetch failed or returned no data.")
        return
    status.success(f"Fetched futures ({len(fut_df)}) and CFD ({len(cfd_df)}) rows.")

    # generate candidates (labels) from futures
    status.info("Generating labels from futures...")
    cands = generate_candidates_and_labels_from_futures(fut_df, lookback=32, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=30)
    if cands.empty:
        status.warning("No candidates generated from futures.")
        return
    status.success(f"Generated {len(cands)} candidates.")

    # align labels to CFD indices
    events, mapping = align_labels_to_cfd(cands, cfd_df)
    if events.empty:
        status.error("Label alignment to CFD failed.")
        return
    status.info(f"Aligned {len(events)} events to CFD indices.")

    # build combined features and train single model
    status.info("Building combined features (Scope/Aim/Shoot) and training single model...")
    X, y, info = build_scope_aim_shoot_features(fut_df, cfd_df, events)
    if X is None or X.size==0:
        status.error("No features built.")
        return
    try:
        res = train_single_model_on_combined_features(X, y, prefer_xgb=(prefer_xgb and xgb is not None), test_size=test_size, seed=random_seed)
    except Exception as e:
        logger.exception("Training failed: %s", e)
        st.error(f"Training failed: {e}")
        return
    status.success("Training finished.")
    st.subheader("Training metrics")
    st.json(res["metrics"])

    # produce predictions across CFD events for backtest
    scaler = res["scaler"]; model = res["model"]
    Xs_all = scaler.transform(X)
    probs = model.predict_proba(Xs_all)[:,1]
    events_df = events.copy()
    events_df["pred_prob"] = probs
    events_df["pred_label"] = (events_df["pred_prob"] >= 0.5).astype(int)

    # merge predictions back into candidate table for simulation
    merged = cands.reset_index(drop=True).copy()
    # assume index alignment between cands and events
    merged["pred_prob"] = events_df["pred_prob"].values
    merged["pred_label"] = events_df["pred_label"].values

    # simulate limit order execution on CFD
    trades = simulate_limits(merged, cfd_df, label_col="pred_label", sl=0.02, tp=0.04, max_holding=30, symbol=symbol_cfd)
    if trades.empty:
        st.warning("No simulated trades.")
    else:
        st.subheader("Simulated trades (head)")
        st.dataframe(trades.head(50))
        st.subheader("Backtest summary")
        st.dataframe(summarize_trades(trades))
        # save summary to supabase if configured
        try:
            summ = summarize_trades(trades)
            if not summ.empty and sb_client:
                rows = [dict(summ.iloc[0])]
                rows[0]["mode"] = "single_model"
                save_layer_summary(sb_client, symbol_futures, rows)
        except Exception as e:
            logger.exception("Saving summary failed: %s", e)

    # export artifact
    art = export_artifact(model, scaler, res["metrics"], basename=f"single_cascade_{symbol_futures.replace('/','_')}")
    st.success(f"Exported artifact: {art['model_path']}")
    st.write("Artifact directory:", art["out_dir"])

if train_btn:
    try:
        run_pipeline_single_model()
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        st.error(f"Pipeline error: {e}")



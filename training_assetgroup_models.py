# prop_firm_cascade_trainer.py
"""
Advanced prop firm trading model trainer with adaptive risk management
- Designed for challenges with 2% max drawdown from open positions
- Adaptive RR scaling from 1:1.1 to 1:5 based on model performance
- Multiple specialized models per asset group with optimized entry levels
- Integrated threshold parameter sweeping and optimization
- Performance distribution analysis with HF model inference for parameter adjustments
"""

import os
import uuid
import json
import math
import logging
import requests
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

# ML and optimization
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, ParameterGrid, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

# Optional libs with import guards
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception:
    xgb = None
    XGBClassifier = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except Exception:
    lgb = None
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
except Exception:
    torch = None
    nn = None

# HF
try:
    from huggingface_hub import HfApi, upload_file, InferenceClient
except Exception:
    HfApi = None
    upload_file = None
    InferenceClient = None

# Supabase client
try:
    from supabase import create_client
except Exception:
    create_client = None

# Hyperopt for parameter sweeping
try:
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
except Exception:
    hp = None
    fmin = None
    tpe = None
    STATUS_OK = None
    Trials = None

# Configuration
SUPABASE_URL = "https://jubcotqsbvguwzklngzd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1YmNvdHFzYnZndXd6a2xuZ3pkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTU0MjA3MCwiZXhwIjoxODUxMTgwNzB9.1HV-o9JFa_nCZGXcoap2OgOCKjRSlyFSRvKmYk70eDk"
HF_TOKEN = os.getenv("HF_TOKEN")
FINNAGE_API_KEY = os.getenv("FINNAGE_API_KEY")

# Mapping Yahoo -> Finnage symbols
MARKET_SYMBOLS_MAP = {
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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prop_firm_cascade")
logger.setLevel(logging.INFO)

# ---- Risk-Reward Profiles ----
RR_PROFILES = {
    "conservative": {"min_rr": 1.1, "max_rr": 1.8, "sl_factor": 1.0},  # Lower risk
    "balanced": {"min_rr": 1.5, "max_rr": 3.0, "sl_factor": 1.0},      # Balanced approach
    "aggressive": {"min_rr": 2.0, "max_rr": 5.0, "sl_factor": 1.0}     # Higher reward potential
}

# ---- Entry Level Definitions ----
ENTRY_LEVELS = {
    "shallow": {
        "confidence_threshold": 0.35,
        "risk_adjustment": 0.5,     # 50% of base risk
        "sl_multiplier": 0.8,       # Tighter stop losses for shallower entries
        "holding_period_factor": 0.7  # Shorter holding periods
    },
    "medium": {
        "confidence_threshold": 0.55,
        "risk_adjustment": 1.0,     # 100% of base risk
        "sl_multiplier": 1.0,
        "holding_period_factor": 1.0
    },
    "deep": {
        "confidence_threshold": 0.75, 
        "risk_adjustment": 1.3,     # 130% of base risk for high confidence
        "sl_multiplier": 1.2,       # Allow wider stops for deep entries
        "holding_period_factor": 1.2  # Allow longer holding periods
    }
}

# ---- Adaptive RR Configuration ----
ADAPTIVE_RR_CONFIG = {
    "high_winrate": {  # >60%
        "percentile_low": 0.0,   # Use lower percentile of RR range
        "percentile_high": 0.3
    },
    "medium_winrate": {  # 50-60%
        "percentile_low": 0.3,   # Use middle percentile of RR range
        "percentile_high": 0.7
    },
    "low_winrate": {  # <50%
        "percentile_low": 0.7,   # Use higher percentile of RR range (higher RR)
        "percentile_high": 1.0
    }
}

# ---- Risk Manager ----
class RiskManager:
    """
    Prop firm compatible risk manager.
    Ensures trades comply with max drawdown limits and adjusts position sizing.
    """
    def __init__(
        self, 
        max_drawdown_pct: float = 2.0,
        max_daily_loss_pct: float = 1.0,
        max_position_risk_pct: float = 0.5,
        account_size: float = 100000.0
    ):
        self.max_drawdown_pct = max_drawdown_pct / 100.0
        self.max_daily_loss_pct = max_daily_loss_pct / 100.0
        self.max_position_risk_pct = max_position_risk_pct / 100.0
        self.account_size = account_size
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.last_reset = datetime.now().date()
        
    def reset_daily_metrics(self):
        """Reset daily PnL tracking"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = today
    
    def update_drawdown(self, pnl: float):
        """Update current drawdown based on new P&L"""
        self.daily_pnl += pnl
        self.current_drawdown = max(self.current_drawdown - pnl/self.account_size, 0.0)
        if pnl < 0:
            self.current_drawdown = max(self.current_drawdown, -pnl/self.account_size)
            
    def get_position_size(self, risk_per_trade: float, stop_loss_pct: float) -> Tuple[float, bool]:
        """
        Calculate safe position size based on risk parameters
        Returns:
            - position size
            - approval flag (True if trade is approved)
        """
        self.reset_daily_metrics()
        
        # Check if we're within prop firm limits
        if self.current_drawdown >= self.max_drawdown_pct:
            return 0.0, False, "Max drawdown limit reached"
            
        if self.daily_pnl / self.account_size <= -self.max_daily_loss_pct:
            return 0.0, False, "Daily loss limit reached"
        
        # Calculate theoretical drawdown if all open positions hit stop loss
        theoretical_dd = self.current_drawdown
        for pos_id, pos in self.open_positions.items():
            theoretical_dd += (pos['risk_amount'] / self.account_size)
            
        # Add potential new position risk
        risk_amount = self.account_size * risk_per_trade
        new_theoretical_dd = theoretical_dd + (risk_amount / self.account_size)
        
        if new_theoretical_dd > self.max_drawdown_pct:
            # Scale down position to fit within limits
            available_risk = self.max_drawdown_pct - theoretical_dd
            risk_amount = self.account_size * available_risk
            if risk_amount <= 0:
                return 0.0, False, "No risk budget available"
                
        # Calculate position size based on stop loss percentage
        if stop_loss_pct <= 0:
            return 0.0, False, "Invalid stop loss percentage"
            
        position_size = risk_amount / stop_loss_pct
        
        return position_size, True, "Trade approved"
    
    def register_position(
        self, 
        position_id: str, 
        entry_price: float,
        stop_loss_price: float,
        position_size: float
    ):
        """Register a new open position"""
        risk_amount = abs(entry_price - stop_loss_price) * position_size / entry_price
        self.open_positions[position_id] = {
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'position_size': position_size,
            'risk_amount': risk_amount
        }
        
    def close_position(self, position_id: str, exit_price: float):
        """Close a position and update metrics"""
        if position_id not in self.open_positions:
            return False
            
        pos = self.open_positions[position_id]
        pnl = (exit_price - pos['entry_price']) * pos['position_size'] / pos['entry_price']
        self.update_drawdown(pnl)
        del self.open_positions[position_id]
        return pnl

# ---- Multi-Level Entry Threshold System ----
class MultiLevelEntryThreshold:
    """
    Multi-level adaptive entry threshold system for prop firm trading.
    Handles shallow (L1), medium (L2), and deep (L3) trade entries
    based on model confidence and volatility-normalized signals.
    """
    def __init__(
        self,
        shallow_threshold: float = 0.35,
        medium_threshold: float = 0.55,
        deep_threshold: float = 0.75,
        vol_sensitivity: float = 0.4,
        adaptive_scaling: bool = True,
        rr_profile: str = "balanced"
    ):
        """
        Parameters:
            shallow_threshold: Entry threshold for light conviction trades
            medium_threshold: Entry threshold for moderate conviction trades
            deep_threshold: Entry threshold for high conviction trades
            vol_sensitivity: Adjusts thresholds based on volatility environment
            adaptive_scaling: Enables dynamic scaling of thresholds using realized volatility
            rr_profile: Risk/reward profile (conservative, balanced, aggressive)
        """
        self.shallow_threshold = shallow_threshold
        self.medium_threshold = medium_threshold
        self.deep_threshold = deep_threshold
        self.vol_sensitivity = vol_sensitivity
        self.adaptive_scaling = adaptive_scaling
        self.rr_profile = rr_profile
        self.risk_manager = RiskManager()
        
    def _adjust_thresholds_for_vol(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Adjusts thresholds dynamically based on realized volatility"""
        if not self.adaptive_scaling or "volatility" not in df.columns:
            return self.shallow_threshold, self.medium_threshold, self.deep_threshold

        vol_mean = df["volatility"].mean()
        vol_std = df["volatility"].std()

        # Higher volatility = higher thresholds (more conservative)
        adj_factor = 1 + (vol_std / vol_mean) * self.vol_sensitivity
        shallow_adj = self.shallow_threshold * adj_factor
        medium_adj = self.medium_threshold * adj_factor
        deep_adj = self.deep_threshold * adj_factor

        return min(shallow_adj, 0.9), min(medium_adj, 0.95), min(deep_adj, 0.98)

    def assign_entry_level(self, df: pd.DataFrame, signal_col: str = "signal") -> pd.DataFrame:
        """
        Assigns an entry level (shallow, medium, deep) based on signal strength.
        The output is a DataFrame with a new column: 'entry_level'.
        """
        shallow, medium, deep = self._adjust_thresholds_for_vol(df)

        df["entry_level"] = np.select(
            [
                (df[signal_col] >= deep),
                (df[signal_col] >= medium) & (df[signal_col] < deep),
                (df[signal_col] >= shallow) & (df[signal_col] < medium),
            ],
            ["deep", "medium", "shallow"],
            default="none"
        )

        return df
    
    def get_rr_for_winrate(self, winrate: float) -> Tuple[float, float]:
        """Get adjusted risk-reward based on model winrate"""
        profile = RR_PROFILES[self.rr_profile]
        min_rr, max_rr = profile["min_rr"], profile["max_rr"]
        rr_range = max_rr - min_rr
        
        if winrate >= 0.6:  # High winrate - lower RR
            low, high = ADAPTIVE_RR_CONFIG["high_winrate"]["percentile_low"], ADAPTIVE_RR_CONFIG["high_winrate"]["percentile_high"]
            percentile = low + (high - low) * np.random.random()  
            return min_rr + percentile * rr_range
        elif winrate >= 0.5:  # Medium winrate - balanced RR
            low, high = ADAPTIVE_RR_CONFIG["medium_winrate"]["percentile_low"], ADAPTIVE_RR_CONFIG["medium_winrate"]["percentile_high"]
            percentile = low + (high - low) * np.random.random()
            return min_rr + percentile * rr_range
        else:  # Low winrate - higher RR
            low, high = ADAPTIVE_RR_CONFIG["low_winrate"]["percentile_low"], ADAPTIVE_RR_CONFIG["low_winrate"]["percentile_high"]
            percentile = low + (high - low) * np.random.random()
            return min_rr + percentile * rr_range
            
    def get_entry_parameters(
        self, 
        entry_level: str, 
        base_risk: float,
        model_winrate: float = 0.5
    ) -> Dict[str, Any]:
        """
        Returns adjusted risk, stop loss, and take profit parameters
        based on entry level and model winrate.
        """
        # Base parameters from entry levels
        level_params = ENTRY_LEVELS.get(entry_level, ENTRY_LEVELS["medium"])
        risk_multiplier = level_params["risk_adjustment"]
        sl_mult = level_params["sl_multiplier"]
        
        # Get RR based on winrate
        rr = self.get_rr_for_winrate(model_winrate)
        
        # Calculate risk per trade (percentage of account)
        adjusted_risk = base_risk * risk_multiplier
        
        return {
            "entry_level": entry_level,
            "risk_per_trade": adjusted_risk,
            "take_profit_mult": rr * sl_mult,  # RR * SL multiplier
            "stop_loss_mult": sl_mult,
            "rr_ratio": rr
        }

    def evaluate_entries(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        base_risk: float = 0.01,
        model_winrate: float = 0.5
    ) -> pd.DataFrame:
        """
        Combines entry level assignment with calculated trade parameters.
        Returns a DataFrame with full entry configuration.
        """
        df = self.assign_entry_level(df, signal_col)
        results = []

        for _, row in df.iterrows():
            if row["entry_level"] == "none":
                continue

            params = self.get_entry_parameters(row["entry_level"], base_risk, model_winrate)
            
            # Add risk manager approval check
            position_size, approved, message = self.risk_manager.get_position_size(
                params["risk_per_trade"], 
                params["stop_loss_mult"] * row.get("atr_pct", 0.01)
            )
            
            results.append({
                "ticker": row.get("ticker", ""),
                "timestamp": row.get("timestamp", ""),
                "entry_signal": row[signal_col],
                "entry_price": row.get("close", 0),
                "position_size": position_size if approved else 0,
                "approved": approved,
                "message": message,
                **params
            })

        return pd.DataFrame(results) if results else pd.DataFrame()

# ---- Advanced Feature Generation ----
def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate advanced features for trading model
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    f = pd.DataFrame(index=df.index)
    
    # Price and returns
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    
    # Basic returns
    f['ret1'] = c.pct_change().fillna(0)
    f['logret1'] = np.log1p(f['ret1'].clip(lower=-0.999999))
    
    # Volatility measures
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    f['tr'] = tr
    f['atr_pct'] = tr / c  # ATR as percentage of price
    
    # Rolling windows
    windows = [5, 10, 20, 40, 60]
    for w in windows:
        # Momentum features
        f[f'mom_{w}'] = c / c.shift(w) - 1
        f[f'rank_{w}'] = c.rolling(w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Volatility features
        f[f'vol_{w}'] = f['ret1'].rolling(w).std()
        f[f'atr_{w}'] = tr.rolling(w).mean()
        f[f'atr_pct_{w}'] = f[f'atr_{w}'] / c
        
        # Range features
        roll_max = h.rolling(w).max()
        roll_min = l.rolling(w).min()
        f[f'hilo_range_{w}'] = (roll_max - roll_min) / c
        f[f'close_to_max_{w}'] = (c - roll_min) / (roll_max - roll_min)
        
        # Volume features
        if 'volume' in df.columns:
            f[f'vol_z_{w}'] = (v - v.rolling(w).mean()) / v.rolling(w).std()
            f[f'vol_ratio_{w}'] = v / v.rolling(w).mean()
        
        # Candlestick patterns
        f[f'body_pct_{w}'] = ((c - o) / tr).rolling(w).mean()  # Body size as % of range
        f[f'upper_wick_{w}'] = ((h - np.maximum(o, c)) / tr).rolling(w).mean()
        f[f'lower_wick_{w}'] = ((np.minimum(o, c) - l) / tr).rolling(w).mean()
    
    # Trend strength
    for w in [20, 40, 60]:
        ema_w = c.ewm(span=w, adjust=False).mean()
        f[f'ema_dist_{w}'] = (c / ema_w - 1) * 100  # Distance from EMA in percentage
        
        # Slope of EMA (trend direction and strength)
        f[f'ema_slope_{w}'] = ema_w.pct_change(5) * 100
    
    # Clean up
    return f.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

# ---- Candidate Generator ----
def generate_signal_candidates(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp_range: Tuple[float, float] = (1.1, 5.0), 
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 100,
    direction: str = "long",
    min_volume: Optional[float] = None,
    entry_levels: Dict[str, Dict[str, float]] = ENTRY_LEVELS,
    model_winrate: float = 0.5
) -> pd.DataFrame:
    """
    Generate trade candidates with multiple TP targets based on ATR multiples
    """
    if bars is None or bars.empty:
        return pd.DataFrame()
        
    df = bars.copy().sort_index()
    df.index = pd.to_datetime(df.index)
    
    # Calculate ATR
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    df['tr'] = tr
    df['atr'] = tr.rolling(window=atr_window, min_periods=1).mean().fillna(method='bfill')
    df['atr_pct'] = df['atr'] / df['close']  # ATR as percentage of price
    
    # Determine RR ratio based on model winrate
    entry_thresholds = MultiLevelEntryThreshold()
    k_tp = entry_thresholds.get_rr_for_winrate(model_winrate) * k_sl
    
    records = []
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
            
        # Generate candidates for each entry level
        for level_name, level_params in entry_levels.items():
            sl_mult = level_params['sl_multiplier']
            
            if direction == "long":
                sl_px = entry_px - k_sl * sl_mult * atr
                tp_px = entry_px + k_tp * sl_mult * atr
            else:
                sl_px = entry_px + k_sl * sl_mult * atr
                tp_px = entry_px - k_tp * sl_mult * atr
                
            # Calculate max lookforward window based on holding period factor
            max_bars_adjusted = int(max_bars * level_params.get('holding_period_factor', 1.0))
            end_i = min(i + max_bars_adjusted, n - 1)
            
            label = 0
            hit_i = end_i
            hit_px = float(df['close'].iat[end_i])
            hit_reason = "timeout"
            
            for j in range(i+1, end_i+1):
                hi = float(df['high'].iat[j])
                lo = float(df['low'].iat[j])
                
                if direction == "long":
                    if hi >= tp_px:
                        label = 1
                        hit_i = j
                        hit_px = tp_px
                        hit_reason = "tp"
                        break
                    if lo <= sl_px:
                        label = 0
                        hit_i = j
                        hit_px = sl_px
                        hit_reason = "sl"
                        break
                else:
                    if lo <= tp_px:
                        label = 1
                        hit_i = j
                        hit_px = tp_px
                        hit_reason = "tp"
                        break
                    if hi >= sl_px:
                        label = 0
                        hit_i = j
                        hit_px = sl_px
                        hit_reason = "sl"
                        break
            
            realized_return = (hit_px - entry_px) / entry_px if direction == "long" else (entry_px - hit_px) / entry_px
            duration_bars = hit_i - i
            duration_min = (df.index[hit_i] - entry_time).total_seconds() / 60.0
            
            rec = {
                "candidate_time": entry_time,
                "entry_price": entry_px,
                "tp_price": tp_px,
                "sl_price": sl_px,
                "rr_ratio": k_tp/k_sl,
                "label": int(label),
                "hit_idx": int(hit_i),
                "hit_reason": hit_reason,
                "realized_return": float(realized_return),
                "duration_bars": int(duration_bars),
                "duration_min": float(duration_min),
                "direction": direction,
                "entry_level": level_name,
                "atr_value": atr,
                "atr_pct": atr / entry_px,
                "sl_mult": sl_mult,
                "sl_pct": abs(sl_px - entry_px) / entry_px
            }
            records.append(rec)
    
    return pd.DataFrame.from_records(records)

# ---- Adaptive Cascade Model ----
if torch is not None:
    class AdaptiveConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k=3, d=1, p=0.1, activation='relu'):
            super().__init__()
            pad = (k-1)*d//2
            self.conv = nn.Conv1d(c_in, c_out, k, dilation=d, padding=pad)
            self.bn = nn.BatchNorm1d(c_out)
            
            if activation == 'relu':
                self.act = nn.ReLU()
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'selu':
                self.act = nn.SELU()
            else:
                self.act = nn.ReLU()
                
            self.drop = nn.Dropout(p)
            self.res = (c_in == c_out)
            
        def forward(self, x):
            y = self.conv(x)
            y = self.bn(y)
            y = self.act(y)
            y = self.drop(y)
            return x + y if self.res else y
            
    class PropFirmCascadeModel(nn.Module):
        """
        Advanced CNN model for prop firm trading with specialized entry level outputs
        """
        def __init__(
            self, 
            in_features: int, 
            channels: List[int] = [32, 64, 128, 128], 
            dropout: float = 0.2,
            activation: str = 'gelu'
        ):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            
            for i in range(len(channels)):
                blocks.append(AdaptiveConvBlock(
                    chs[i], chs[i+1], 
                    k=3, d=2**min(i, 5), 
                    p=dropout, 
                    activation=activation
                ))
                
            self.blocks = nn.Sequential(*blocks)
            
            # Attention layer
            self.attention = nn.Sequential(
                nn.Conv1d(chs[-1], 1, kernel_size=1),
                nn.Softmax(dim=2)
            )
            
            # Multiple heads for different entry levels
            self.proj = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            
            # Entry level classifiers
            self.shallow_head = nn.Linear(chs[-1], 1)
            self.medium_head = nn.Linear(chs[-1], 1)
            self.deep_head = nn.Linear(chs[-1], 1)
            
            # Shared embedding for downstream models
            self.shared_head = nn.Linear(chs[-1], 1)
            
        def forward(self, x):
            # x: [B, F, T]
            z = self.blocks(x)
            z = self.proj(z)
            
            # Attention-weighted pooling
            a = self.attention(z)
            emb = (z * a).sum(dim=2)
            
            # Multiple outputs for different entry levels
            shallow_logit = self.shallow_head(emb)
            medium_logit = self.medium_head(emb)
            deep_logit = self.deep_head(emb)
            shared_logit = self.shared_head(emb)
            
            return {
                "shallow": shallow_logit,
                "medium": medium_logit, 
                "deep": deep_logit,
                "shared": shared_logit,
                "embedding": emb
            }
else:
    PropFirmCascadeModel = None

class PropFirmModelEnsemble:
    """
    Ensemble of multiple specialized models for different entry levels and asset characteristics
    """
    def __init__(
        self, 
        seq_len: int = 64, 
        feat_windows: Tuple[int,...] = (5, 10, 20, 40, 60),
        ensemble_size: int = 3,
        model_types: List[str] = None,
        device: str = 'auto',
        max_drawdown_pct: float = 2.0,
    ):
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.ensemble_size = ensemble_size
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_types = model_types or ['cnn', 'xgboost', 'lgbm']
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        
        # Ensemble model containers
        self.models = {
            "shallow": [],
            "medium": [],
            "deep": []
        }
        
        # Performance metrics for each model
        self.model_metrics = {
            "shallow": [],
            "medium": [],
            "deep": []
        }
        
        # Entry level parameters
        self.entry_levels = ENTRY_LEVELS
        
        # Risk management
        self.risk_manager = RiskManager(max_drawdown_pct=max_drawdown_pct)
        
        # Flag to indicate if models are trained
        self._fitted = False

    def _to_sequences(self, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Convert feature matrix to sequence data for time series models"""
        N, F = features.shape
        out = np.zeros((len(indices), self.seq_len, F), dtype=features.dtype)
        
        for i, t in enumerate(indices):
            t = int(t)
            t0 = t - self.seq_len + 1
            seq = features[max(0, t0):t+1]
            
            if seq.shape[0] < self.seq_len:
                pad = np.repeat(seq[[0]], self.seq_len - seq.shape[0], axis=0)
                seq = np.vstack([pad, seq])
                
            out[i] = seq[-self.seq_len:]
            
        return out
        
    def _create_model(self, model_type: str, input_dim: int, level_name: str = 'medium'):
        """Factory method to create a model based on type"""
        if model_type == 'cnn' and torch is not None:
            # Create PyTorch CNN model
            return PropFirmCascadeModel(
                in_features=input_dim, 
                channels=[32, 64, 128, 128], 
                dropout=0.2,
                activation='gelu'
            )
        elif model_type == 'xgboost' and XGBClassifier is not None:
            # Create XGBoost model with optimized parameters
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.05, 
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=2.0,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        elif model_type == 'lgbm' and LGBMClassifier is not None:
            # Create LightGBM model
            return LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0
            )
        elif model_type == 'catboost' and CatBoostClassifier is not None:
            # Create CatBoost model
            return CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                bootstrap_type='Bayesian',
                random_strength=1,
                verbose=0
            )
        else:
            # Fallback to XGBoost if available
            if XGBClassifier is not None:
                return XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss")
            else:
                raise ValueError(f"Model type {model_type} not available with current imports")
        
    def _train_torch_model(self, model, ds_tr, ds_va, epochs=20, patience=5, batch_size=128, lr=1e-3):
        """Train a PyTorch model with early stopping"""
        if torch is None:
            raise ValueError("PyTorch is required but not installed")
            
        device = torch.device(self.device)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        
        tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(ds_va, batch_size=batch_size*4, shuffle=False)
        
        best_loss = float('inf')
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for x_batch, y_batch in tr_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                outputs = model(x_batch)
                # Use the appropriate output based on level
                loss = criterion(outputs["shared"], y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item() * len(x_batch)
                
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for x_batch, y_batch in va_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    loss = criterion(outputs["shared"], y_batch)
                    
                    val_loss += loss.item() * len(x_batch)
                    val_preds.append(torch.sigmoid(outputs["shared"]).cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())
            
            val_loss /= len(ds_va)
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)
            
        # Get validation metrics
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for x_batch, y_batch in va_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                val_preds.append(torch.sigmoid(outputs["shared"]).cpu().numpy())
                val_labels.append(y_batch.numpy())
        
        val_preds = np.concatenate(val_preds).flatten()
        val_labels = np.concatenate(val_labels).flatten()
        val_auc = roc_auc_score(val_labels, val_preds)
        
        return model, {"val_loss": best_loss, "val_auc": val_auc}
        
    def _train_tree_model(self, model, X_tr, y_tr, X_va, y_va):
        """Train a tree-based model (XGBoost, LightGBM, CatBoost)"""
        if hasattr(model, 'fit'):
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Get validation metrics
            if hasattr(model, 'predict_proba'):
                val_preds = model.predict_proba(X_va)[:, 1]
                val_auc = roc_auc_score(y_va, val_preds)
            else:
                val_preds = model.predict(X_va)
                val_auc = roc_auc_score(y_va, val_preds)
                
            return model, {"val_auc": val_auc}
        else:
            raise ValueError("Model doesn't have a fit method")

    def fit(
        self, 
        bars: pd.DataFrame, 
        candidates: pd.DataFrame, 
        epochs_cnn: int = 20,
        train_size: float = 0.7,
        val_size: float = 0.15,
        parameter_sweep: bool = True,
    ):
        """
        Train multiple specialized models for different entry levels
        
        Args:
            bars: DataFrame with OHLCV data
            candidates: DataFrame with labeled trade candidates
            epochs_cnn: Number of epochs for CNN models
            train_size: Fraction of data to use for training
            val_size: Fraction of data to use for validation
            parameter_sweep: Whether to run hyperparameter sweep
        """
        if candidates is None or candidates.empty:
            raise ValueError("No candidates provided for training")
        
        # Prepare features
        features = compute_advanced_features(bars)
        
        # Split by entry level if available
        if 'entry_level' in candidates.columns:
            level_candidates = {
                level: candidates[candidates['entry_level'] == level]
                for level in self.entry_levels.keys()
            }
        else:
            # If no entry level in candidates, treat all as medium
            level_candidates = {
                "medium": candidates,
                "shallow": candidates.sample(frac=0.7, random_state=42),
                "deep": candidates.sample(frac=0.7, random_state=24)
            }
            
        # Process each entry level
        for level, cands in level_candidates.items():
            if cands.empty:
                logger.warning(f"No candidates for entry level: {level}")
                continue
                
            # Extract labeled events from candidates
            indices = []
            labels = []
            
            for _, row in cands.iterrows():
                try:
                    # Find the index in bars that corresponds to candidate time
                    t = pd.to_datetime(row['candidate_time'])
                    locs = bars.index[bars.index <= t]
                    if len(locs) > 0:
                        idx = bars.index.get_loc(locs[-1])
                        indices.append(idx)
                        labels.append(int(row.get('label', 0)))
                except Exception as e:
                    continue
            
            if not indices:
                logger.warning(f"No valid indices for level: {level}")
                continue
                
            # Convert to numpy arrays
            indices = np.array(indices)
            labels = np.array(labels)
            
            # Time series split to prevent lookahead bias
            ts_split = TimeSeriesSplit(n_splits=5, test_size=int(len(indices) * val_size))
            for train_idx, test_idx in ts_split.split(indices):
                train_indices = indices[train_idx]
                val_indices = indices[test_idx]
                
                y_train = labels[train_idx]
                y_val = labels[test_idx]
                
                # Prepare sequence features
                X_seq = self._to_sequences(features.values, indices)
                X_seq_train = X_seq[train_idx]
                X_seq_val = X_seq[test_idx]
                
                # Standardize sequences
                X_seq_flat = X_seq.reshape(-1, X_seq.shape[2])
                self.scaler_seq.fit(X_seq_flat)
                
                X_seq_train_std = self.scaler_seq.transform(X_seq_train.reshape(-1, X_seq_train.shape[2]))
                X_seq_train_std = X_seq_train_std.reshape(X_seq_train.shape)
                
                X_seq_val_std = self.scaler_seq.transform(X_seq_val.reshape(-1, X_seq_val.shape[2]))
                X_seq_val_std = X_seq_val_std.reshape(X_seq_val.shape)
                
                # Train different model types for this entry level
                for model_type in self.model_types:
                    if model_type == 'cnn':
                        if torch is None:
                            logger.warning("PyTorch not available, skipping CNN")
                            continue
                            
                        # Prepare PyTorch dataset
                        class SeqDataset(Dataset):
                            def __init__(self, X, y):
                                self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
                                self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
                                
                            def __len__(self):
                                return len(self.X)
                                
                            def __getitem__(self, idx):
                                return self.X[idx], self.y[idx]
                        
                        train_ds = SeqDataset(X_seq_train_std, y_train)
                        val_ds = SeqDataset(X_seq_val_std, y_val)
                        
                        # Create and train model
                        model = self._create_model('cnn', X_seq_train.shape[2], level)
                        model, metrics = self._train_torch_model(
                            model, train_ds, val_ds, 
                            epochs=epochs_cnn, patience=5
                        )
                        
                    else:  # Tree-based models
                        # Flatten sequences for tree models
                        X_train_flat = X_seq_train_std.reshape(X_seq_train_std.shape[0], -1)
                        X_val_flat = X_seq_val_std.reshape(X_seq_val_std.shape[0], -1)
                        
                        # Create and train model
                        model = self._create_model(model_type, X_train_flat.shape[1], level)
                        model, metrics = self._train_tree_model(
                            model, X_train_flat, y_train, X_val_flat, y_val
                        )
                
                    # Store model and metrics
                    self.models[level].append((model_type, model))
                    self.model_metrics[level].append(metrics)
                    
                # Only use the last split
                break
                
        self._fitted = True
        
        # Parameter sweep if requested
        if parameter_sweep:
            self.run_parameter_sweep(bars, candidates)
            
        return self
        
    def run_parameter_sweep(self, bars: pd.DataFrame, candidates: pd.DataFrame):
        """Run hyperparameter sweep to find optimal thresholds"""
        param_grid = {
            'shallow_threshold': [0.25, 0.3, 0.35, 0.4, 0.45],
            'medium_threshold': [0.45, 0.5, 0.55, 0.6, 0.65],
            'deep_threshold': [0.65, 0.7, 0.75, 0.8, 0.85],
            'vol_sensitivity': [0.2, 0.3, 0.4, 0.5, 0.6],
            'rr_min': [1.1, 1.3, 1.5, 1.8],
            'rr_max': [2.5, 3.0, 3.5, 4.0, 5.0],
        }
        
        if hp is None or fmin is None or tpe is None:
            logger.warning("hyperopt not available, skipping parameter sweep")
            return
            
        # Define the search space
        space = {
            'shallow_threshold': hp.choice('shallow_threshold', param_grid['shallow_threshold']),
            'medium_threshold': hp.choice('medium_threshold', param_grid['medium_threshold']),
            'deep_threshold': hp.choice('deep_threshold', param_grid['deep_threshold']),
            'vol_sensitivity': hp.choice('vol_sensitivity', param_grid['vol_sensitivity']),
            'rr_min': hp.choice('rr_min', param_grid['rr_min']),
            'rr_max': hp.choice('rr_max', param_grid['rr_max']),
        }
        
        # Define the objective function
        def objective(params):
            # Create entry threshold system with these params
            threshold_system = MultiLevelEntryThreshold(
                shallow_threshold=params['shallow_threshold'],
                medium_threshold=params['medium_threshold'],
                deep_threshold=params['deep_threshold'],
                vol_sensitivity=params['vol_sensitivity'],
            )
            
            # Create a small feature set for volatility adjustment
            features = pd.DataFrame(index=bars.index)
            features['volatility'] = bars['close'].pct_change().rolling(20).std().fillna(0)
            
            # Generate entries with these parameters
            preds = self.predict_proba(bars)
            if preds.empty:
                return {'loss': float('inf'), 'status': STATUS_OK}
                
            preds['timestamp'] = bars.index[preds['t'].astype(int).values]
            preds['close'] = bars['close'].values[preds['t'].astype(int).values]
            preds['atr_pct'] = 0.01  # Placeholder, would come from bars
            
            # Evaluate entries - use a sample to speed up
            entries_sample = threshold_system.evaluate_entries(
                preds.sample(min(1000, len(preds)), random_state=42),
                signal_col='p3',
                base_risk=0.01,
                model_winrate=0.55  # Midpoint estimate
            )
            
            if entries_sample.empty:
                return {'loss': float('inf'), 'status': STATUS_OK}
                
            # Calculate risk-adjusted score
            # We want more approved trades with higher expected value
            n_approved = entries_sample['approved'].sum()
            expected_value = entries_sample[entries_sample['approved']]['rr_ratio'].mean() * 0.55
            
            if n_approved == 0 or math.isnan(expected_value):
                return {'loss': float('inf'), 'status': STATUS_OK}
                
            # Loss is negative of risk-adjusted reward
            loss = -1 * (n_approved/len(entries_sample)) * expected_value
            
            return {
                'loss': loss,
                'n_approved': n_approved,
                'expected_value': expected_value,
                'status': STATUS_OK
            }
            
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=25,
            trials=trials
        )
        
        # Store best parameters
        self.best_params = {
            'shallow_threshold': param_grid['shallow_threshold'][best['shallow_threshold']],
            'medium_threshold': param_grid['medium_threshold'][best['medium_threshold']],
            'deep_threshold': param_grid['deep_threshold'][best['deep_threshold']],
            'vol_sensitivity': param_grid['vol_sensitivity'][best['vol_sensitivity']],
            'rr_min': param_grid['rr_min'][best['rr_min']],
            'rr_max': param_grid['rr_max'][best['rr_max']],
        }
        
        logger.info(f"Best params from sweep: {self.best_params}")
        return self.best_params
        
    def _get_model_winrate(self, level: str) -> float:
        """Get the estimated winrate for a specific level's models"""
        if not self.model_metrics[level]:
            return 0.5  # Default midpoint
            
        # Get average AUC and convert to approximate winrate
        avg_auc = np.mean([m.get('val_auc', 0.5) for m in self.model_metrics[level]])
        # Convert AUC to approximate winrate (rough heuristic)
        winrate = 0.5 + (avg_auc - 0.5) * 0.5
        return min(max(winrate, 0.4), 0.7)  # Clip to reasonable range
        
    def predict_proba(self, bars: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """
        Generate predictions for all potential entry points
        
        Returns:
            DataFrame with prediction probabilities for each entry level
        """
        if not self._fitted:
            raise ValueError("Models not fitted yet")
            
        # Create feature matrix
        features = compute_advanced_features(bars)
        
        # All possible indices
        indices = np.arange(self.seq_len, len(bars))
        
        # Prepare sequences
        X_seq = self._to_sequences(features.values, indices)
        X_seq_std = self.scaler_seq.transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_seq_std = X_seq_std.reshape(X_seq.shape)
        
        # Initialize prediction arrays
        shallow_preds = np.zeros(len(indices))
        medium_preds = np.zeros(len(indices))
        deep_preds = np.zeros(len(indices))
        
        # Generate predictions from each model and ensemble
        for level, models in self.models.items():
            preds_list = []
            
            for model_type, model in models:
                if model_type == 'cnn':
                    # PyTorch prediction
                    model.eval()
                    batch_size = 256
                    level_preds = []
                    
                    with torch.no_grad():
                        for i in range(0, len(X_seq_std), batch_size):
                            batch = X_seq_std[i:i+batch_size]
                            x_batch = torch.tensor(batch, dtype=torch.float32).transpose(1, 2)
                            x_batch = x_batch.to(self.device)
                            outputs = model(x_batch)
                            
                            # Use the level-specific output
                            logits = outputs.get(level, outputs.get("shared"))
                            probs = torch.sigmoid(logits).cpu().numpy().flatten()
                            level_preds.append(probs)
                            
                    level_preds = np.concatenate(level_preds)
                    
                else:
                    # Tree model prediction
                    X_flat = X_seq_std.reshape(X_seq_std.shape[0], -1)
                    level_preds = model.predict_proba(X_flat)[:, 1]
                
                preds_list.append(level_preds)
            
            # Ensemble predictions (average)
            ensemble_preds = np.mean(preds_list, axis=0) if preds_list else np.zeros(len(indices))
            
            # Store predictions for each level
            if level == 'shallow':
                shallow_preds = ensemble_preds
            elif level == 'medium':
                medium_preds = ensemble_preds
            elif level == 'deep':
                deep_preds = ensemble_preds
        
        # Combine predictions for final signal
        # We scale by 10 to match expected signal range (0-10)
        combined_signal = (shallow_preds * 0.2 + medium_preds * 0.5 + deep_preds * 0.3) * 10
        
        # Filter by threshold
        mask = combined_signal > threshold
        
        # Create result dataframe
        result = pd.DataFrame({
            't': indices[mask],
            'p1': shallow_preds[mask],
            'p2': medium_preds[mask],
            'p3': combined_signal[mask] / 10.0,  # Normalize back to 0-1 range
            'shallow': shallow_preds[mask],
            'medium': medium_preds[mask],
            'deep': deep_preds[mask]
        })
        
        return result
        
    def get_optimal_entry_config(self) -> Dict[str, Dict[str, Any]]:
        """Get optimal entry configuration based on model performance"""
        config = {}
        
        for level in self.entry_levels.keys():
            winrate = self._get_model_winrate(level)
            
            # Start with base entry level parameters
            params = dict(ENTRY_LEVELS[level])
            
            # Adjust based on winrate
            if winrate >= 0.6:  # High winrate model
                # For high winrate, we can be more aggressive with entries
                params["confidence_threshold"] *= 0.9  # Lower threshold
                params["risk_adjustment"] *= 1.2  # More risk (still within prop firm limits)
                rr_cat = "high_winrate"
            elif winrate >= 0.5:  # Medium winrate
                # Keep defaults
                rr_cat = "medium_winrate"
            else:  # Low winrate
                # More conservative
                params["confidence_threshold"] *= 1.1  # Higher threshold
                params["risk_adjustment"] *= 0.8  # Less risk per trade
                rr_cat = "low_winrate"
                
            # Set RR range
            params["rr_percentile_low"] = ADAPTIVE_RR_CONFIG[rr_cat]["percentile_low"]
            params["rr_percentile_high"] = ADAPTIVE_RR_CONFIG[rr_cat]["percentile_high"]
            params["estimated_winrate"] = winrate
            
            config[level] = params
            
        return config
        
    def evaluate_performance(
        self, 
        bars: pd.DataFrame, 
        candidates: pd.DataFrame,
        output_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on candidates
        
        Returns dictionary with performance metrics and visualizations
        """
        # Generate predictions
        preds = self.predict_proba(bars)
        
        if preds.empty or candidates.empty:
            return {"error": "No predictions or candidates available"}
        
        # Match predictions to candidates
        cand_times = pd.to_datetime(candidates['candidate_time'])
        pred_times = bars.index[preds['t'].astype(int).values]
        
        # Map predictions to candidates (nearest time match)
        merged_results = []
        
        for idx, row in candidates.iterrows():
            t = pd.to_datetime(row['candidate_time'])
            # Find closest prediction time
            time_diffs = np.abs((pred_times - t).total_seconds())
            if len(time_diffs) > 0 and np.min(time_diffs) < 86400:  # Within 1 day
                closest_idx = np.argmin(time_diffs)
                merged_row = {**row.to_dict(), **preds.iloc[closest_idx].to_dict()}
                merged_results.append(merged_row)
        
        if not merged_results:
            return {"error": "No matching predictions found for candidates"}
            
        merged_df = pd.DataFrame(merged_results)
        
        # Calculate performance metrics
        metrics = {}
        
        # Use p3 for signal strength
        thresholds = np.linspace(0.3, 0.9, 10)
        
        for thresh in thresholds:
            pred_labels = (merged_df['p3'] >= thresh).astype(int)
            true_labels = merged_df['label'].astype(int)
            
            # Count trades and win rate
            n_trades = (pred_labels == 1).sum()
            if n_trades > 0:
                win_rate = (pred_labels & true_labels).sum() / n_trades
                avg_return = merged_df.loc[pred_labels == 1, 'realized_return'].mean()
            else:
                win_rate = 0
                avg_return = 0
                
            metrics[f"thresh_{thresh:.1f}"] = {
                "n_trades": int(n_trades),
                "win_rate": float(win_rate),
                "avg_return": float(avg_return),
                "expected_value": float(win_rate * avg_return - (1-win_rate) * 0.01)
            }
            
        # Analyze by entry level
        if 'entry_level' in merged_df.columns:
            for level in merged_df['entry_level'].unique():
                level_df = merged_df[merged_df['entry_level'] == level]
                
                # Get optimal threshold
                best_expected_value = -np.inf
                best_threshold = 0.5
                
                for thresh in np.linspace(0.3, 0.9, 20):
                    pred_labels = (level_df['p3'] >= thresh).astype(int)
                    if pred_labels.sum() < 10:
                        continue
                        
                    true_labels = level_df['label'].astype(int)
                    win_rate = (pred_labels & true_labels).sum() / pred_labels.sum()
                    avg_return = level_df.loc[pred_labels == 1, 'realized_return'].mean()
                    expected_value = win_rate * avg_return - (1-win_rate) * 0.01
                    
                    if expected_value > best_expected_value:
                        best_expected_value = expected_value
                        best_threshold = thresh
                
                metrics[f"level_{level}"] = {
                    "optimal_threshold": float(best_threshold),
                    "expected_value": float(best_expected_value),
                    "n_samples": int(len(level_df))
                }
        
        # Overall metrics
        precision, recall, thresholds_pr = precision_recall_curve(
            merged_df['label'].astype(int), 
            merged_df['p3']
        )
        
        metrics["overall"] = {
            "auc": float(roc_auc_score(merged_df['label'].astype(int), merged_df['p3'])),
            "avg_precision": float(np.mean(precision)),
            "model_winrate": float(self._get_model_winrate("medium")),
            "n_candidates": int(len(merged_df))
        }
        
        # Performance distribution
        performance_metrics = {
            "thresholds": metrics,
            "entry_configs": self.get_optimal_entry_config(),
        }
        
        # Add parameter suggestions if best_params exists
        if hasattr(self, 'best_params'):
            performance_metrics["param_suggestions"] = self.best_params
            
        return performance_metrics
        
    def suggest_parameter_adjustments(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance metrics and suggest parameter adjustments
        
        This can be extended to use an LLM if available
        """
        suggestions = {}
        
        # Basic heuristic-based suggestions
        overall_metrics = performance_metrics.get("thresholds", {}).get("overall", {})
        model_winrate = overall_metrics.get("model_winrate", 0.5)
        
        if model_winrate >= 0.6:
            # High winrate model
            suggestions["confidence"] = "High winrate detected. Consider lowering entry thresholds and using more trades."
            suggestions["threshold_adjustment"] = "Decrease by 5-10%"
            suggestions["position_sizing"] = "Can increase base risk slightly"
            suggestions["rr_profile"] = "conservative"
        elif model_winrate >= 0.5:
            # Medium winrate
            suggestions["confidence"] = "Medium winrate detected. Use balanced approach."
            suggestions["threshold_adjustment"] = "Keep current thresholds"
            suggestions["position_sizing"] = "Maintain standard position sizing"
            suggestions["rr_profile"] = "balanced"
        else:
            # Low winrate
            suggestions["confidence"] = "Lower winrate detected. Consider raising thresholds and focusing on quality over quantity."
            suggestions["threshold_adjustment"] = "Increase by 5-10%"
            suggestions["position_sizing"] = "Reduce base risk by 20-30%"
            suggestions["rr_profile"] = "aggressive"
            
        # Use best params from sweep if available
        if "param_suggestions" in performance_metrics:
            suggestions["recommended_params"] = performance_metrics["param_suggestions"]
            
        # Try to get LLM advice if available
        if InferenceClient is not None and HF_TOKEN:
            try:
                client = InferenceClient(token=HF_TOKEN)
                prompt = f"""
                Analyze this trading model performance and suggest parameter adjustments:
                - Overall winrate: {model_winrate:.2f}
                - AUC score: {overall_metrics.get('auc', 'N/A')}
                - Number of samples: {overall_metrics.get('n_candidates', 'N/A')}
                
                Current parameters:
                {json.dumps(self.best_params if hasattr(self, 'best_params') else {}, indent=2)}
                
                The model is designed for prop firm challenges with max 2% drawdown.
                Suggest specific parameter adjustments to improve performance while minimizing drawdown risk.
                """
                
                response = client.text_generation(
                    prompt, 
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    max_new_tokens=300
                )
                
                suggestions["llm_advice"] = response
            except Exception as e:
                logger.warning(f"Failed to get LLM advice: {e}")
                
        return suggestions
        
# ---- PropFirmPipeline ----
class PropFirmPipeline:
    """
    Full trading model pipeline specialized for prop firm challenges
    - Data fetching, preprocessing, feature engineering
    - Multi-model training with specialized entry levels
    - Parameter optimization
    - Metrics logging and visualization
    - Model export and deployment
    """
    def __init__(
        self,
        asset_group: str,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        max_drawdown_pct: float = 2.0,
        data_source: str = 'finnage',  # 'finnage', 'yahoo', or 'both'
        base_risk: float = 0.01,
        rr_profile: str = 'balanced',
        device: str = 'auto'
    ):
        self.asset_group = asset_group
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.max_drawdown_pct = max_drawdown_pct
        self.data_source = data_source
        self.base_risk = base_risk
        self.rr_profile = rr_profile
        self.device = device
        
        self.models = {}
        self.bars_data = {}
        self.candidates = {}
        self.performance_metrics = {}
        
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all tickers in the asset group"""
        logger.info(f"Fetching data for {len(self.tickers)} tickers in {self.asset_group}")
        
        all_bars = {}
        
        for sym in self.tickers:
            logger.info(f"Fetching {sym}")
            
            # Determine Finnage symbol if available
            fin_map = MARKET_SYMBOLS_MAP.get(self.asset_group, {}).get(sym)
            df_finnage = pd.DataFrame()
            df_yahoo = pd.DataFrame()
            
            # Fetch from Finnage if symbol mapping exists and data source includes Finnage
            if fin_map and (self.data_source in ['finnage', 'both']):
                df_finnage = self._fetch_finnage_bars(
                    fin_map, 
                    datetime.combine(self.start_date, datetime.min.time()),
                    datetime.combine(self.end_date, datetime.min.time())
                )
                
            # Fetch from Yahoo if data source includes Yahoo
            if self.data_source in ['yahoo', 'both']:
                df_yahoo = self._fetch_yahoo_bars(
                    sym, 
                    datetime.combine(self.start_date, datetime.min.time()),
                    datetime.combine(self.end_date, datetime.min.time())
                )
                
            # Combine data, preferring Finnage
            if not df_finnage.empty:
                if df_yahoo.empty:
                    df = df_finnage
                else:
                    # Combine, prioritize Finnage
                    df = pd.concat([df_finnage, df_yahoo]).sort_index().drop_duplicates(keep='first')
            else:
                df = df_yahoo
                
            if df is None or df.empty:
                logger.warning(f"No data for {sym}")
                continue
                
            # Normalize and clean data
            df.index = pd.to_datetime(df.index).tz_localize(None) if hasattr(df.index, 'tz') else pd.to_datetime(df.index)
            
            for col in ('open', 'high', 'low', 'close', 'volume'):
                if col not in df.columns:
                    df[col] = df['close'] if col != 'volume' and 'close' in df.columns else 0.0
                    
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df.loc[:, 'symbol'] = sym
            
            all_bars[sym] = df
            logger.info(f"Fetched {len(df)} bars for {sym}")
            
        self.bars_data = all_bars
        return all_bars
            
    def _fetch_finnage_bars(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch data from Finnage API"""
        if not FINNAGE_API_KEY:
            logger.warning("FINNAGE_API_KEY not set; skipping Finnage fetch")
            return pd.DataFrame()
            
        try:
            url = (
                "https://finnhub.io/api/v1/stock/candle"
                f"?symbol={symbol}&resolution=D&from={int(start_dt.timestamp())}"
                f"&to={int(end_dt.timestamp())}&token={FINNAGE_API_KEY}"
            )
            
            logger.debug(f"Finnage URL: {url}")
            resp = requests.get(url, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get('s') != 'ok':
                logger.debug(f"Finnage returned s={data.get('s')} for {symbol}")
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
            logger.exception(f"Finnage fetch error for {symbol}: {e}")
            return pd.DataFrame()
            
    def _fetch_yahoo_bars(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
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
            logger.exception(f"Yahoo fetch failed for {symbol}: {e}")
            return pd.DataFrame()
            
    def generate_candidates(self) -> Dict[str, pd.DataFrame]:
        """Generate labeled candidates for each ticker"""
        if not self.bars_data:
            raise ValueError("No data available. Call fetch_data() first.")
            
        all_candidates = {}
        
        for sym, bars in self.bars_data.items():
            # Generate candidates with multiple entry levels
            cands = generate_signal_candidates(
                bars,
                lookback=64,
                k_tp_range=(1.1, 5.0),
                k_sl=1.0,
                atr_window=14,
                max_bars=100,
                direction="long",
                entry_levels=ENTRY_LEVELS,
                model_winrate=0.5  # Default for initial generation
            )
            
            if cands is None or cands.empty:
                logger.info(f"No candidates for {sym}")
                continue
                
            cands.loc[:, 'symbol'] = sym
            all_candidates[sym] = cands
            logger.info(f"Generated {len(cands)} candidates for {sym}")
            
        self.candidates = all_candidates
        return all_candidates
        
    def train_models(
        self,
        ensemble_size: int = 3,
        model_types: List[str] = None,
        parameter_sweep: bool = True,
        epochs_cnn: int = 20
    ) -> Dict[str, PropFirmModelEnsemble]:
        """Train specialized models for each asset in the group"""
        if not self.bars_data or not self.candidates:
            raise ValueError("Missing data or candidates. Call fetch_data() and generate_candidates() first.")
            
        if model_types is None:
            model_types = ['cnn', 'xgboost'] if xgb is not None else ['cnn']
            
        # Create models for each ticker
        trained_models = {}
        
        for sym, bars in self.bars_data.items():
            if sym not in self.candidates:
                logger.warning(f"No candidates for {sym}, skipping model training")
                continue
                
            cands = self.candidates[sym]
            
            if cands.empty:
                logger.warning(f"Empty candidates for {sym}, skipping model training")
                continue
                
            logger.info(f"Training models for {sym}")
            
            # Create and train the ensemble
            model = PropFirmModelEnsemble(
                seq_len=64,
                feat_windows=(5, 10, 20, 40, 60),
                ensemble_size=ensemble_size,
                model_types=model_types,
                device=self.device,
                max_drawdown_pct=self.max_drawdown_pct
            )
            
            try:
                model.fit(
                    bars,
                    cands,
                    epochs_cnn=epochs_cnn,
                    parameter_sweep=parameter_sweep
                )
                
                trained_models[sym] = model
                logger.info(f"Successfully trained models for {sym}")
                
            except Exception as e:
                logger.exception(f"Error training models for {sym}: {e}")
                
        self.models = trained_models
        return trained_models
        
    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models"""
        if not self.models:
            raise ValueError("No trained models. Call train_models() first.")
            
        all_metrics = {}
        
        for sym, model in self.models.items():
            logger.info(f"Evaluating model for {sym}")
            
            try:
                metrics = model.evaluate_performance(
                    self.bars_data[sym], 
                    self.candidates[sym]
                )
                
                # Get parameter suggestions
                suggestions = model.suggest_parameter_adjustments(metrics)
                metrics['parameter_suggestions'] = suggestions
                
                all_metrics[sym] = metrics
                logger.info(f"Evaluation complete for {sym}")
                
            except Exception as e:
                logger.exception(f"Error evaluating model for {sym}: {e}")
                all_metrics[sym] = {"error": str(e)}
                
        self.performance_metrics = all_metrics
        return all_metrics
        
    def optimize_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Optimize parameters for each model based on performance"""
        if not self.models or not self.performance_metrics:
            raise ValueError("Missing models or performance metrics. Call train_models() and evaluate_models() first.")
            
        optimized_params = {}
        
        for sym, model in self.models.items():
            logger.info(f"Optimizing parameters for {sym}")
            
            try:
                # Use best parameters from performance metrics if available
                metrics = self.performance_metrics.get(sym, {})
                suggestions = metrics.get('parameter_suggestions', {})
                
                if 'recommended_params' in suggestions:
                    params = suggestions['recommended_params']
                    
                    # Update model's internal parameters
                    for key, value in params.items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                            
                    optimized_params[sym] = params
                    logger.info(f"Updated parameters for {sym}")
                    
            except Exception as e:
                logger.exception(f"Error optimizing parameters for {sym}: {e}")
                
        return optimized_params
        
    def export_models(self, output_dir: str = None) -> Dict[str, str]:
        """Export trained models to files"""
        if not self.models:
            raise ValueError("No trained models. Call train_models() first.")
            
        if output_dir is None:
            output_dir = f"prop_firm_models_{self.asset_group}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        
        for sym, model in self.models.items():
            logger.info(f"Exporting model for {sym}")
            
            try:
                model_path = output_path / f"{sym}_model.pt"
                
                if torch is not None:
                    torch.save(model, str(model_path))
                    export_paths[sym] = str(model_path)
                    logger.info(f"Saved model for {sym} to {model_path}")
                else:
                    logger.warning(f"Torch not available, cannot save model for {sym}")
                    
            except Exception as e:
                logger.exception(f"Error exporting model for {sym}: {e}")
                
        # Export configuration
        try:
            config = {
                'asset_group': self.asset_group,
                'max_drawdown_pct': self.max_drawdown_pct,
                'tickers': self.tickers,
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'performance_metrics': self.performance_metrics
            }
            
            config_path = output_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
                
            export_paths['config'] = str(config_path)
            
        except Exception as e:
            logger.exception(f"Error exporting configuration: {e}")
            
        return export_paths
        
    def upload_to_hf(self, repo_name: str) -> Dict[str, str]:
        """Upload models to HuggingFace"""
        if HF_TOKEN is None or HfApi is None or upload_file is None:
            logger.warning("HF_TOKEN or huggingface_hub not available")
            return {}
            
        exported_paths = self.export_models()
        upload_urls = {}
        
        api = HfApi(token=HF_TOKEN)
        
        for sym, path in exported_paths.items():
            logger.info(f"Uploading {sym} model to HF")
            
            try:
                url = api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{sym}_model.pt" if sym != 'config' else "config.json",
                    repo_id=repo_name,
                    repo_type="model"
                )
                
                upload_urls[sym] = url
                logger.info(f"Uploaded {sym} model to {url}")
                
            except Exception as e:
                logger.exception(f"Error uploading model for {sym}: {e}")
                
        return upload_urls
        
    def log_metrics(self) -> Dict[str, Any]:
        """Log model metrics to Supabase"""
        if not self.performance_metrics:
            raise ValueError("No performance metrics. Call evaluate_models() first.")
            
        if create_client is None:
            logger.warning("supabase-py not installed; skipping Supabase logging")
            return {}
            
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            logger.exception(f"Failed to initialize Supabase client: {e}")
            return {}
            
        correlation_id = str(uuid.uuid4())
        logger.info(f"Logging metrics to Supabase; correlation_id={correlation_id}")
        
        # Prepare metrics
        ticker_stats = {}
        
        for sym, metrics in self.performance_metrics.items():
            overall = metrics.get('thresholds', {}).get('overall', {})
            
            stats = {
                "num_bars": len(self.bars_data.get(sym, [])),
                "num_candidates": len(self.candidates.get(sym, [])),
                "auc": overall.get('auc', float('nan')),
                "model_winrate": overall.get('model_winrate', float('nan')),
                "avg_precision": overall.get('avg_precision', float('nan')),
                "prop_firm_compatible": True
            }
            
            ticker_stats[sym] = stats
            
        # Aggregate group metrics
        group_metrics = {}
        
        for metric in ["num_bars", "num_candidates", "auc", "model_winrate", "avg_precision"]:
            values = [stats.get(metric, float('nan')) for stats in ticker_stats.values()]
            group_metrics[metric] = float(np.nanmean(values))
            
        # Insert per ticker
        for ticker, stats in ticker_stats.items():
            row = {
                "asset_group": self.asset_group,
                "ticker": ticker,
                "metrics_json": json.dumps(stats),
                "num_bars": int(stats.get("num_bars", 0)),
                "num_candidates": int(stats.get("num_candidates", 0)),
                "trained_at": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id,
                "prop_firm_mode": True,
                "max_drawdown": float(self.max_drawdown_pct)
            }
            
            try:
                supabase.table("model_metrics").insert(row).execute()
                logger.debug(f"Inserted metrics for {ticker}")
            except Exception as e:
                logger.exception(f"Supabase insert error for {ticker}: {e}")
                
        # Insert group summary
        group_row = {
            "asset_group": self.asset_group,
            "ticker": None,
            "metrics_json": json.dumps(group_metrics),
            "num_bars": int(np.nanmean([v.get("num_bars", 0) for v in ticker_stats.values()])),
            "num_candidates": int(np.nanmean([v.get("num_candidates", 0) for v in ticker_stats.values()])),
            "trained_at": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "prop_firm_mode": True,
            "max_drawdown": float(self.max_drawdown_pct)
        }
        
        try:
            supabase.table("model_metrics").insert(group_row).execute()
            logger.debug("Inserted group aggregate row")
        except Exception as e:
            logger.exception(f"Supabase insert failed for group row: {e}")
            
        return {"correlation_id": correlation_id, "ticker_stats": ticker_stats}
        
    def run_full_pipeline(
        self,
        ensemble_size: int = 3,
        model_types: List[str] = None,
        parameter_sweep: bool = True,
        epochs_cnn: int = 20,
        hf_repo_name: str = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from data fetching to model export
        
        Returns dictionary with results from each step
        """
        results = {
            "start_time": datetime.utcnow().isoformat(),
            "asset_group": self.asset_group,
            "tickers": self.tickers
        }
        
        try:
            # Fetch data
            logger.info("Fetching data...")
            bars = self.fetch_data()
            results["data_fetch"] = {
                "tickers": list(bars.keys()),
                "total_bars": sum(len(df) for df in bars.values())
            }
            
            # Generate candidates
            logger.info("Generating candidates...")
            candidates = self.generate_candidates()
            results["candidates"] = {
                "tickers": list(candidates.keys()),
                "total_candidates": sum(len(df) for df in candidates.values())
            }
            
            # Train models
            logger.info("Training models...")
            models = self.train_models(
                ensemble_size=ensemble_size,
                model_types=model_types,
                parameter_sweep=parameter_sweep,
                epochs_cnn=epochs_cnn
            )
            results["training"] = {
                "trained_models": list(models.keys())
            }
            
            # Evaluate models
            logger.info("Evaluating models...")
            metrics = self.evaluate_models()
            results["evaluation"] = {
                "evaluated_models": list(metrics.keys())
            }
            
            # Optimize parameters
            logger.info("Optimizing parameters...")
            optimized = self.optimize_parameters()
            results["optimization"] = {
                "optimized_models": list(optimized.keys())
            }
            
            # Export models
            logger.info("Exporting models...")
            export_paths = self.export_models()
            results["export"] = {
                "exported_models": list(export_paths.keys()),
                "paths": export_paths
            }
            
            # Upload to HF if repo name provided
            if hf_repo_name:
                logger.info(f"Uploading to HuggingFace repo: {hf_repo_name}...")
                upload_urls = self.upload_to_hf(hf_repo_name)
                results["hf_upload"] = {
                    "uploaded_models": list(upload_urls.keys()),
                    "urls": upload_urls
                }
                
            # Log metrics to Supabase
            logger.info("Logging metrics to Supabase...")
            log_results = self.log_metrics()
            results["logging"] = log_results
            
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            results["error"] = str(e)
            
        results["end_time"] = datetime.utcnow().isoformat()
        results["duration_seconds"] = (
            datetime.fromisoformat(results["end_time"]) - 
            datetime.fromisoformat(results["start_time"])
        ).total_seconds()
        
        return results

# ---- MultiLevelEntryOptimizer ----
class MultiLevelEntryOptimizer:
    """
    Optimizes multi-level entry thresholds for prop firm challenges
    Finds the best entry parameters for different confidence levels
    """
    def __init__(
        self,
        max_drawdown_pct: float = 2.0,
        rr_range: Tuple[float, float] = (1.1, 5.0),
        base_risk_pct: float = 0.5,
        entry_levels: Dict[str, Dict[str, float]] = ENTRY_LEVELS
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.rr_range = rr_range
        self.base_risk_pct = base_risk_pct
        self.entry_levels = entry_levels
        self.risk_manager = RiskManager(max_drawdown_pct=max_drawdown_pct)
        
    def sweep_threshold_parameters(
        self,
        bars: pd.DataFrame,
        candidates: pd.DataFrame,
        predictions: pd.DataFrame,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Perform parameter sweep to find optimal entry thresholds
        
        Args:
            bars: DataFrame with OHLCV data
            candidates: DataFrame with labeled trade candidates
            predictions: DataFrame with model predictions
            n_trials: Number of parameter combinations to try
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        if hp is None or fmin is None or tpe is None or STATUS_OK is None:
            logger.warning("hyperopt not available, using basic grid search")
            return self._basic_grid_search(bars, candidates, predictions)
            
        # Map predictions to candidates
        merged_df = self._merge_predictions_candidates(predictions, candidates, bars)
        
        if merged_df.empty:
            return {"error": "No predictions matched to candidates"}
            
        # Define the search space
        space = {
            'shallow_threshold': hp.uniform('shallow_threshold', 0.25, 0.45),
            'medium_threshold': hp.uniform('medium_threshold', 0.45, 0.65),
            'deep_threshold': hp.uniform('deep_threshold', 0.65, 0.85),
            'vol_sensitivity': hp.uniform('vol_sensitivity', 0.2, 0.6),
            'rr_min': hp.uniform('rr_min', 1.1, 1.8),
            'rr_max': hp.uniform('rr_max', 2.0, 5.0),
            'sl_factor_shallow': hp.uniform('sl_factor_shallow', 0.6, 1.0),
            'sl_factor_medium': hp.uniform('sl_factor_medium', 0.8, 1.2),
            'sl_factor_deep': hp.uniform('sl_factor_deep', 1.0, 1.4)
        }
        
        # Define the objective function
        def objective(params):
            # Apply thresholds
            df = merged_df.copy()
            
            # Assign entry levels based on thresholds
            df['entry_level'] = np.select(
                [
                    (df['signal'] >= params['deep_threshold']),
                    (df['signal'] >= params['medium_threshold']) & (df['signal'] < params['deep_threshold']),
                    (df['signal'] >= params['shallow_threshold']) & (df['signal'] < params['medium_threshold']),
                ],
                ['deep', 'medium', 'shallow'],
                default='none'
            )
            
            # Filter out non-entries
            df = df[df['entry_level'] != 'none']
            
            if len(df) == 0:
                return {'loss': float('inf'), 'status': STATUS_OK}
                
            # Calculate risk and RR for each level
            df['sl_factor'] = df['entry_level'].map({
                'shallow': params['sl_factor_shallow'],
                'medium': params['sl_factor_medium'],
                'deep': params['sl_factor_deep']
            })
            
            # Calculate R:R ratio based on winrate per level
            level_winrates = {}
            
            for level in ['shallow', 'medium', 'deep']:
                level_df = df[df['entry_level'] == level]
                if len(level_df) > 0:
                    winrate = level_df['label'].mean()
                    level_winrates[level] = winrate
                else:
                    level_winrates[level] = 0.5  # Default
                    
            # Assign RR based on winrate
            df['rr_ratio'] = df.apply(
                lambda row: self._get_rr_for_winrate(
                    level_winrates.get(row['entry_level'], 0.5),
                    params['rr_min'],
                    params['rr_max']
                ),
                axis=1
            )
            
            # Calculate expected profit
            df['sl_pct'] = df.apply(
                lambda row: float(row.get('atr_pct', 0.01)) * row['sl_factor'],
                axis=1
            )
            
            df['tp_pct'] = df['sl_pct'] * df['rr_ratio']
            df['expected_value'] = (df['tp_pct'] * level_winrates.get(df['entry_level'], 0.5) - 
                                   df['sl_pct'] * (1 - level_winrates.get(df['entry_level'], 0.5)))
            
            # Calculate risk-adjusted return
            total_ev = df['expected_value'].sum()
            n_trades = len(df)
            
            # Calculate drawdown risk - rough estimate
            avg_trade_risk = df['sl_pct'].mean() * self.base_risk_pct
            max_concurrent_trades = self.max_drawdown_pct / avg_trade_risk
            
            # Calculate score - balancing expected return and drawdown protection
            if max_concurrent_trades < 3:
                # Penalize very high risk per trade
                score = total_ev * (max_concurrent_trades / 3)
            else:
                # Normal scoring
                score = total_ev
                
            # Loss is negative of score (we want to maximize score)
            return {
                'loss': -score,
                'total_ev': float(total_ev),
                'n_trades': int(n_trades),
                'avg_risk': float(avg_trade_risk),
                'max_concurrent_trades': float(max_concurrent_trades),
                'status': STATUS_OK
            }
            
        # Run the optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials
        )
        
        # Get best parameters
        best_params = {
            'shallow_threshold': best['shallow_threshold'],
            'medium_threshold': best['medium_threshold'],
            'deep_threshold': best['deep_threshold'],
            'vol_sensitivity': best['vol_sensitivity'],
            'rr_min': best['rr_min'],
            'rr_max': best['rr_max'],
            'sl_factor_shallow': best['sl_factor_shallow'],
            'sl_factor_medium': best['sl_factor_medium'],
            'sl_factor_deep': best['sl_factor_deep']
        }
        
        # Get performance for best parameters
        best_result = trials.best_trial['result']
        
        return {
            'optimized_params': best_params,
            'performance': {
                'total_expected_value': best_result['total_ev'],
                'n_trades': best_result['n_trades'],
                'avg_risk_per_trade': best_result['avg_risk'],
                'max_concurrent_trades': best_result['max_concurrent_trades']
            },
            'all_trials': [t['result'] for t in trials.trials]
        }
        
    def _basic_grid_search(
        self,
        bars: pd.DataFrame,
        candidates: pd.DataFrame, 
        predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """Basic grid search if hyperopt is not available"""
        # Map predictions to candidates
        merged_df = self._merge_predictions_candidates(predictions, candidates, bars)
        
        if merged_df.empty:
            return {"error": "No predictions matched to candidates"}
            
        # Create parameter grid
        param_grid = {
            'shallow_threshold': [0.3, 0.35, 0.4],
            'medium_threshold': [0.5, 0.55, 0.6],
            'deep_threshold': [0.7, 0.75, 0.8],
            'rr_min': [1.1, 1.5],
            'rr_max': [3.0, 4.0, 5.0],
        }
        
        all_results = []
        
        # Try all combinations
        for shallow in param_grid['shallow_threshold']:
            for medium in param_grid['medium_threshold']:
                for deep in param_grid['deep_threshold']:
                    for rr_min in param_grid['rr_min']:
                        for rr_max in param_grid['rr_max']:
                            params = {
                                'shallow_threshold': shallow,
                                'medium_threshold': medium,
                                'deep_threshold': deep,
                                'rr_min': rr_min,
                                'rr_max': rr_max,
                                'sl_factor_shallow': 0.8,
                                'sl_factor_medium': 1.0,
                                'sl_factor_deep': 1.2
                            }
                            
                            # Apply thresholds
                            df = merged_df.copy()
                            
                            # Assign entry levels
                            df['entry_level'] = np.select(
                                [
                                    (df['signal'] >= deep),
                                    (df['signal'] >= medium) & (df['signal'] < deep),
                                    (df['signal'] >= shallow) & (df['signal'] < medium),
                                ],
                                ['deep', 'medium', 'shallow'],
                                default='none'
                            )
                            
                            # Filter out non-entries
                            df = df[df['entry_level'] != 'none']
                            
                            if len(df) == 0:
                                continue
                                
                            # Calculate basic metrics
                            n_trades = len(df)
                            avg_winrate = df['label'].mean()
                            
                            # Add to results
                            all_results.append({
                                'params': params,
                                'n_trades': n_trades,
                                'avg_winrate': avg_winrate,
                                'score': n_trades * avg_winrate
                            })
                            
        # Find best result
        if not all_results:
            return {"error": "No valid parameter combinations found"}
            
        best_result = max(all_results, key=lambda x: x['score'])
        
        return {
            'optimized_params': best_result['params'],
            'performance': {
                'n_trades': best_result['n_trades'],
                'avg_winrate': best_result['avg_winrate'],
                'score': best_result['score']
            },
            'all_results': all_results
        }
                                
    def _merge_predictions_candidates(
        self,
        predictions: pd.DataFrame,
        candidates: pd.DataFrame,
        bars: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Match predictions to candidates for evaluation
        """
        # Create mapping from time index to prediction
        pred_map = {}
        
        for _, row in predictions.iterrows():
            t_idx = int(row['t'])
            if 0 <= t_idx < len(bars):
                pred_map[bars.index[t_idx]] = row.to_dict()
                
        # Match candidates to predictions
        merged_results = []
        
        for _, cand in candidates.iterrows():
            t = pd.to_datetime(cand['candidate_time'])
            
            # Find exact match
            if t in pred_map:
                merged_row = {**cand.to_dict(), **pred_map[t]}
                merged_results.append(merged_row)
            else:
                # Find nearest prediction within 1 day
                nearest_t = None
                min_diff = pd.Timedelta(days=1)
                
                for pred_t in pred_map.keys():
                    diff = abs(pred_t - t)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_t = pred_t
                        
                if nearest_t is not None:
                    merged_row = {**cand.to_dict(), **pred_map[nearest_t]}
                    merged_results.append(merged_row)
                    
        if not merged_results:
            return pd.DataFrame()
            
        merged_df = pd.DataFrame(merged_results)
        
        # Add signal column (use p3 if available, otherwise combined)
        if 'p3' in merged_df.columns:
            merged_df['signal'] = merged_df['p3']
        elif 'shallow' in merged_df.columns and 'medium' in merged_df.columns and 'deep' in merged_df.columns:
            merged_df['signal'] = (
                merged_df['shallow'] * 0.2 + 
                merged_df['medium'] * 0.5 + 
                merged_df['deep'] * 0.3
            )
        else:
            # Default to p1 if available
            merged_df['signal'] = merged_df.get('p1', 0.5)
            
        return merged_df
        
    def _get_rr_for_winrate(
        self,
        winrate: float,
        min_rr: float,
        max_rr: float
    ) -> float:
        """
        Get adjusted risk-reward based on model winrate
        
        Higher winrate = lower RR needed
        Lower winrate = higher RR needed
        """
        rr_range = max_rr - min_rr
        
        if winrate >= 0.6:  # High winrate
            low, high = ADAPTIVE_RR_CONFIG["high_winrate"]["percentile_low"], ADAPTIVE_RR_CONFIG["high_winrate"]["percentile_high"]
            percentile = low + (high - low) * np.random.random()
            return min_rr + percentile * rr_range
            
        elif winrate >= 0.5:  # Medium winrate
            low, high = ADAPTIVE_RR_CONFIG["medium_winrate"]["percentile_low"], ADAPTIVE_RR_CONFIG["medium_winrate"]["percentile_high"]
            percentile = low + (high - low) * np.random.random()
            return min_rr + percentile * rr_range
            
        else:  # Low winrate
            low, high = ADAPTIVE_RR_CONFIG["low_winrate"]["percentile_low"], ADAPTIVE_RR_CONFIG["low_winrate"]["percentile_high"]
            percentile = low + (high - low) * np.random.random()
            return min_rr + percentile * rr_range
            
    def get_entry_params_for_model(
        self,
        model_winrates: Dict[str, float],
        optimized_params: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate optimal entry parameters for a model with known winrates
        
        Args:
            model_winrates: Dict with winrates for each entry level
            optimized_params: Optional dict with optimized parameters from sweep
            
        Returns:
            Dict with entry parameters for each level
        """
        entry_params = {}
        
        # Use optimized params if available
        if optimized_params:
            shallow_threshold = optimized_params.get('shallow_threshold', 0.35)
            medium_threshold = optimized_params.get('medium_threshold', 0.55)
            deep_threshold = optimized_params.get('deep_threshold', 0.75)
            
            sl_factor_shallow = optimized_params.get('sl_factor_shallow', 0.8)
            sl_factor_medium = optimized_params.get('sl_factor_medium', 1.0)
            sl_factor_deep = optimized_params.get('sl_factor_deep', 1.2)
            
            rr_min = optimized_params.get('rr_min', 1.1)
            rr_max = optimized_params.get('rr_max', 5.0)
        else:
            # Use defaults
            shallow_threshold = 0.35
            medium_threshold = 0.55
            deep_threshold = 0.75
            
            sl_factor_shallow = 0.8
            sl_factor_medium = 1.0
            sl_factor_deep = 1.2
            
            rr_min = 1.1
            rr_max = 5.0
            
        # Generate parameters for each entry level
        for level in ['shallow', 'medium', 'deep']:
            winrate = model_winrates.get(level, 0.5)
            
            # Base parameters
            if level == 'shallow':
                threshold = shallow_threshold
                sl_factor = sl_factor_shallow
                risk_adjustment = 0.5
            elif level == 'medium':
                threshold = medium_threshold
                sl_factor = sl_factor_medium
                risk_adjustment = 1.0
            else:  # deep
                threshold = deep_threshold
                sl_factor = sl_factor_deep
                risk_adjustment = 1.3
                
            # Get RR based on winrate
            rr = self._get_rr_for_winrate(winrate, rr_min, rr_max)
            
            # Calculate entry parameters
            entry_params[level] = {
                'confidence_threshold': threshold,
                'risk_adjustment': risk_adjustment,
                'sl_multiplier': sl_factor,
                'tp_multiplier': sl_factor * rr,
                'rr_ratio': rr,
                'holding_period_factor': 0.7 if level == 'shallow' else (1.2 if level == 'deep' else 1.0),
                'estimated_winrate': winrate
            }
            
        return entry_params

# ---- HF Model Inference Helper ----
class ModelInferenceHelper:
    """
    Helper class for running inference with Hugging Face models
    Can be used to get threshold adjustment suggestions from LLMs
    """
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token or HF_TOKEN
        if InferenceClient is None:
            logger.warning("huggingface_hub.InferenceClient not available")
            
    def get_parameter_suggestions(
        self,
        performance_metrics: Dict[str, Any],
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ) -> str:
        """
        Use LLM to suggest parameter adjustments based on performance metrics
        """
        if InferenceClient is None or not self.hf_token:
            return "LLM inference not available. Install huggingface_hub and set HF_TOKEN."
            
        client = InferenceClient(token=self.hf_token)
        
        # Extract key performance metrics
        overall = performance_metrics.get("thresholds", {}).get("overall", {})
        model_winrate = overall.get("model_winrate", 0.5)
        auc = overall.get("auc", 0.5)
        avg_precision = overall.get("avg_precision", 0.5)
        
        # Extract threshold performance
        threshold_metrics = {}
        for k, v in performance_metrics.get("thresholds", {}).items():
            if k.startswith("thresh_"):
                thresh = k.split("_")[1]
                threshold_metrics[thresh] = v
                
        # Format the prompt
        prompt = f"""
        You are an AI assistant helping optimize a trading model for prop firm challenges.
        The model needs to maximize returns while keeping drawdown under 2%.
        
        Here are the performance metrics:
        - Model winrate: {model_winrate:.2f}
        - AUC score: {auc:.2f}
        - Average precision: {avg_precision:.2f}
        
        Performance at different thresholds:
        {json.dumps(threshold_metrics, indent=2)}
        
        Please analyze these metrics and suggest:
        1. The optimal confidence threshold
        2. Risk-reward ratio adjustments
        3. Position sizing recommendations
        4. Any other parameter adjustments to reduce drawdown risk
        
        Focus on prop firm compatibility - keeping drawdown under 2% is critical.
        """
        
        try:
            response = client.text_generation(
                prompt,
                model=model_name,
                max_new_tokens=500,
                temperature=0.3,
                top_p=0.95
            )
            return response
        except Exception as e:
            logger.exception(f"HF inference error: {e}")
            return f"LLM inference failed: {str(e)}"
            
    def visualize_performance_distribution(
        self,
        performance_metrics: Dict[str, Any],
        show_plot: bool = True
    ) -> Any:
        """
        Visualize performance metrics distribution
        
        Returns plot figure if matplotlib is available
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract threshold performance data
            thresholds = []
            win_rates = []
            expected_values = []
            trade_counts = []
            
            for k, v in performance_metrics.get("thresholds", {}).items():
                if k.startswith("thresh_"):
                    thresh = float(k.split("_")[1])
                    thresholds.append(thresh)
                    win_rates.append(v.get("win_rate", 0))
                    expected_values.append(v.get("expected_value", 0))
                    trade_counts.append(v.get("n_trades", 0))
                    
            if not thresholds:
                return None
                
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Plot win rate vs threshold
            axes[0].plot(thresholds, win_rates, 'o-', color='blue')
            axes[0].set_xlabel('Confidence Threshold')
            axes[0].set_ylabel('Win Rate')
            axes[0].set_title('Win Rate by Confidence Threshold')
            axes[0].grid(True)
            
            # Plot expected value vs threshold
            axes[1].plot(thresholds, expected_values, 'o-', color='green')
            axes[1].set_xlabel('Confidence Threshold')
            axes[1].set_ylabel('Expected Value')
            axes[1].set_title('Expected Value by Confidence Threshold')
            axes[1].grid(True)
            
            # Plot trade count vs threshold
            axes[2].plot(thresholds, trade_counts, 'o-', color='red')
            axes[2].set_xlabel('Confidence Threshold')
            axes[2].set_ylabel('Number of Trades')
            axes[2].set_title('Trade Count by Confidence Threshold')
            axes[2].grid(True)
            
            plt.tight_layout()
            
            if show_plot:
                plt.show()
                
            return fig
            
        except ImportError:
            return "Matplotlib and seaborn required for visualization"

# ---- Main function to run pipeline ----
def run_prop_firm_pipeline(
    asset_group: str,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    max_drawdown_pct: float = 2.0,
    hf_repo_name: str = None,
    model_types: List[str] = None
) -> Dict[str, Any]:
    """
    Run the complete prop firm training pipeline
    
    Args:
        asset_group: Asset group name
        tickers: List of ticker symbols
        start_date: Start date for training data
        end_date: End date for training data
        max_drawdown_pct: Maximum drawdown percentage
        hf_repo_name: HuggingFace repository name for model upload
        model_types: List of model types to train
        
    Returns:
        Dictionary with results from the pipeline run
    """
    logger.info(f"Starting prop firm pipeline for {asset_group} with {len(tickers)} tickers")
    
    # Create pipeline
    pipeline = PropFirmPipeline(
        asset_group=asset_group,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        max_drawdown_pct=max_drawdown_pct,
        data_source='both',  # Use both Finnage and Yahoo
        base_risk=0.01,
        rr_profile='balanced'
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        ensemble_size=3,
        model_types=model_types or ['cnn', 'xgboost', 'lgbm'],
        parameter_sweep=True,
        epochs_cnn=20,
        hf_repo_name=hf_repo_name
    )
    
    logger.info(f"Pipeline completed in {results.get('duration_seconds', 0):.1f} seconds")
    return results

# If running as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prop Firm Model Training Pipeline")
    parser.add_argument("--asset-group", type=str, default="FX", help="Asset group name")
    parser.add_argument("--start-date", type=str, default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"), 
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-drawdown", type=float, default=2.0, help="Maximum drawdown percentage")
    parser.add_argument("--repo", type=str, help="HuggingFace repository name")
    parser.add_argument("--models", type=str, default="cnn,xgboost", help="Comma-separated list of model types")
    
    args = parser.parse_args()
    
    # Get tickers for the asset group
    asset_group_tickers = list(MARKET_SYMBOLS_MAP.get(args.asset_group, {}).keys())
    
    if not asset_group_tickers:
        print(f"No tickers found for asset group: {args.asset_group}")
        print(f"Available asset groups: {list(MARKET_SYMBOLS_MAP.keys())}")
        exit(1)
        
    model_types = args.models.split(",")
    
    # Run pipeline
    results = run_prop_firm_pipeline(
        asset_group=args.asset_group,
        tickers=asset_group_tickers,
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
        max_drawdown_pct=args.max_drawdown,
        hf_repo_name=args.repo,
        model_types=model_types
    )
    
    # Print summary
    print("\n=== Pipeline Results ===")
    print(f"Asset Group: {args.asset_group}")
    print(f"Tickers: {len(asset_group_tickers)}")
    print(f"Models Trained: {len(results.get('training', {}).get('trained_models', []))}")
    print(f"Duration: {results.get('duration_seconds', 0):.1f} seconds")
    
    if 'error' in results:
        print(f"Error: {results['error']}")

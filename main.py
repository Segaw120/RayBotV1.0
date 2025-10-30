# futures_single_layer_trainer.py
# Streamlit dashboard for single-layer model (Futures only)
# Scope/Aim/Shoot method simplified to one-layer training
# Created for futures dataset from Yahoo Finance

import os
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# optional dependencies
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Streamlit setup
st.set_page_config(page_title="Futures L1 Trainer", layout="wide")
st.title("Single-Layer Futures Trainer (Scope Model)")

st.sidebar.header("Training Parameters")
symbol = st.sidebar.text_input("Futures Symbol", value="GC=F")
start_date = st.sidebar.date_input("Start Date", value=(datetime.utcnow() - timedelta(days=365)).date())
end_date = st.sidebar.date_input("End Date", value=datetime.utcnow().date())
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m"], index=0)

epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch Size", min_value=8, max_value=1024, value=64)
seq_len = st.sidebar.number_input("Sequence Length", min_value=8, max_value=256, value=64)
device_choice = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
use_torch = st.sidebar.checkbox("Use Torch (if available)", value=True)
train_button = st.sidebar.button("Train Single-Layer Model")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("futures_single_layer")

# === Helper: Compute engineered features ===
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    c, h, l = df["close"], df["high"], df["low"]
    v = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change().fillna(0)
    f["logret"] = np.log1p(f["ret1"].replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f["tr"] = tr.fillna(0)
    for w in windows:
        f[f"vol_{w}"] = f["ret1"].rolling(w).std().fillna(0)
        f[f"mom_{w}"] = (c - c.rolling(w).mean()).fillna(0)
        f[f"tr_mean_{w}"] = tr.rolling(w).mean().fillna(0)
    return f.replace([np.inf, -np.inf], 0).fillna(0)

class FuturesOnlyTrainer:
    def __init__(self, data_loader, model, optimizer, criterion, device, logger=None):
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_epoch(self, epoch_index):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_index, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if batch_index % 50 == 0:
                self._log(
                    f"Epoch [{epoch_index}] Batch [{batch_index}/{len(self.data_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self._log(f"Epoch {epoch_index} completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def validate(self, validation_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

class ScopeAimShootTrainer(FuturesOnlyTrainer):
    """
    Extension of FuturesOnlyTrainer to include the Scope-Aim-Shoot trading logic.
    """

    def __init__(self, data_loader, model, optimizer, criterion, device, entry_logic, logger=None):
        super().__init__(data_loader, model, optimizer, criterion, device, logger)
        self.entry_logic = entry_logic  # Dict of {"scope": fn, "aim": fn, "shoot": fn}

    def generate_signals(self, inputs):
        """
        Apply the Scope-Aim-Shoot logic to create entry signals.
        """
        scoped = self.entry_logic["scope"](inputs)
        aimed = self.entry_logic["aim"](scoped)
        shoot_signal = self.entry_logic["shoot"](aimed)
        return shoot_signal

    def train_with_signals(self, epoch_index):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            signals = self.generate_signals(inputs)

            outputs = self.model(signals)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 50 == 0:
                self._log(f"[Epoch {epoch_index}] Step {batch_idx}/{len(self.data_loader)} | Loss: {loss.item():.5f}")

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self._log(f"Epoch {epoch_index} completed - Loss: {avg_loss:.5f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def evaluate_signals(self, validation_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                signals = self.generate_signals(inputs)
                outputs = self.model(signals)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

# ------------------------------------------
# Entry Logic: Scope → Aim → Shoot
# ------------------------------------------

def scope(inputs):
    """
    Broad pattern recognition stage.
    - Identify trending or volatile zones.
    - Output: filtered tensor emphasizing directional bias.
    """
    # Example: Moving average difference signal
    short_ma = torch.mean(inputs[:, -5:], dim=1)
    long_ma = torch.mean(inputs[:, -20:], dim=1)
    signal = (short_ma - long_ma).unsqueeze(1)
    return signal


def aim(scoped_inputs):
    """
    Refine the scoped signal by identifying potential turning points or breakouts.
    - Output: normalized activation for directional decision-making.
    """
    normalized = torch.tanh(scoped_inputs)
    return normalized


def shoot(aimed_inputs):
    """
    Execute trade decision logic.
    - Translate signal strength into a binary long/short trigger.
    """
    trigger = (aimed_inputs > 0).float()
    return trigger


# ------------------------------------------
# Training Loop Orchestration
# ------------------------------------------

def train_futures_model(data_loader, validation_loader, model, optimizer, criterion, device, epochs=10, logger=None):
    """
    Orchestrates full single-layer model training with Scope-Aim-Shoot logic.
    """

    entry_logic = {"scope": scope, "aim": aim, "shoot": shoot}
    trainer = ScopeAimShootTrainer(
        data_loader=data_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        entry_logic=entry_logic,
        logger=logger
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_with_signals(epoch)
        val_loss, val_acc = trainer.evaluate_signals(validation_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_futures_scope_aim_shoot.pt")

        if logger:
            logger.info(f"[EPOCH {epoch}] Train: {train_loss:.5f}/{train_acc:.4f} | Val: {val_loss:.5f}/{val_acc:.4f}")
        else:
            print(f"[EPOCH {epoch}] Train: {train_loss:.5f}/{train_acc:.4f} | Val: {val_loss:.5f}/{val_acc:.4f}")

    print(f"✅ Training complete. Best validation loss: {best_val_loss:.5f}")
    return model

# ------------------------------------------
# Futures Dataset Loader
# ------------------------------------------

class FuturesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for futures market data.
    Accepts pandas DataFrame containing OHLC + engineered features.
    """

    def __init__(self, df, feature_cols, target_col, sequence_length=32):
        self.features = df[feature_cols].values
        self.targets = df[target_col].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_dataloaders(train_df, val_df, feature_cols, target_col, batch_size=64, seq_len=32):
    train_ds = FuturesDataset(train_df, feature_cols, target_col, seq_len)
    val_ds = FuturesDataset(val_df, feature_cols, target_col, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ------------------------------------------
# Configuration and Main Entry
# ------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch.optim as optim
    from sklearn.model_selection import train_test_split

    # --- Load and preprocess dataset ---
    df = pd.read_csv("futures_data.csv")
    feature_cols = [col for col in df.columns if col not in ["target", "date"]]
    target_col = "target"

    # --- Split data ---
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

    # --- Initialize loaders ---
    train_loader, val_loader = prepare_dataloaders(train_df, val_df, feature_cols, target_col)

    # --- Model, optimizer, criterion ---
    input_dim = len(feature_cols)
    model = SingleLayerFuturesModel(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Train model ---
    trained_model = train_futures_model(
        data_loader=train_loader,
        validation_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=15
    )

    # --- Save final model ---
    torch.save(trained_model.state_dict(), "final_futures_scope_aim_shoot.pt")
    print("✅ Futures model training completed and saved successfully.")


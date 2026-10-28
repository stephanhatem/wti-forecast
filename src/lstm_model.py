"""
LSTM forecaster for the WTI spot price 20 trading days ahead.

Input:  the last 60 trading days of daily features: WTI and Brent spot, plus
        the five weekly EIA series as they stood on each day (only values
        already released, so 60 trading days covers about the last 12 weekly
        reports). Each feature is standardized with training-set mean and
        standard deviation. Shape: (batch, 60, 7).
Output: the predicted 20-day change in WTI. The forecast is today's WTI plus
        that change. Predicting the change instead of the price level keeps
        the target on a similar scale from 2010 to 2026, and "no change" is
        exactly the persistence baseline.
Loss:   mean squared error on the standardized change.
Optimizer: Adam.

Futures are left out of the inputs because they are missing after April
2024, so a model that needed them couldn't forecast validation or test.

Defaults (16 hidden units, learning rate 1e-4) come from a small grid over
hidden size {16, 32} and learning rate {1e-3, 1e-4}. At 1e-3 the model
overfits within an epoch or two; 1e-4 trains more steadily.

Training keeps the epoch with the lowest validation loss. Because validation
is used to choose that epoch, the validation score is slightly optimistic;
the test split is the honest check once the model is final.

Usage:
    from src.lstm_model import make_windows, fit_lstm, lstm_forecast
    data = make_windows(daily)
    model, history = fit_lstm(data)
    forecasts = lstm_forecast(model, data)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = [
    "wti", "brent",
    "crude_production", "refinery_crude_input", "refinery_utilization",
    "commercial_crude_stocks", "crude_net_imports",
]
WINDOW = 60   # trading days of history per forecast


def make_windows(daily: pd.DataFrame, features=FEATURES, window: int = WINDOW) -> dict:
    """Turn the daily table into model inputs.

    One sample per usable forecast date that has a full window of history.
    Returns a dict with the tensors X and y, plus what's needed to turn
    predictions back into prices (dates, split, current price, y_std).
    """
    # Inputs may be filled from the past (never the future) to cover gaps.
    feats = daily[list(features)].ffill()
    train = (daily["split"] == "train").to_numpy()
    mean, std = feats[train].mean(), feats[train].std()
    z = ((feats - mean) / std).to_numpy(dtype="float32")

    change = (daily["wti_target"] - daily["wti"]).to_numpy(dtype="float32")
    usable = daily["usable"].to_numpy(dtype=bool)
    y_std = float(np.std(change[train & usable]))

    rows = [i for i in range(window - 1, len(daily))
            if usable[i] and not np.isnan(z[i - window + 1:i + 1]).any()]
    X = np.stack([z[i - window + 1:i + 1] for i in rows])

    return {
        "X": torch.from_numpy(X),
        "y": torch.from_numpy(change[rows] / y_std),
        "dates": daily.index[rows],
        "split": daily["split"].to_numpy()[rows],
        "current": daily["wti"].to_numpy()[rows],
        "y_std": y_std,
        "features": list(features),
    }


class LSTMForecaster(nn.Module):
    """LSTM over the input window, then a linear layer on the last step."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)               # (batch, window, hidden)
        return self.head(out[:, -1]).squeeze(-1)


def _subset(data: dict, split: str):
    mask = torch.from_numpy(data["split"] == split)
    return data["X"][mask], data["y"][mask]


def fit_lstm(data: dict, hidden: int = 16, lr: float = 1e-4, epochs: int = 30,
             batch_size: int = 64, seed: int = 0):
    """Train on the train split and keep the epoch with the lowest
    validation loss. Returns (model, history), where history has the train
    and validation loss for every epoch."""
    torch.manual_seed(seed)
    X_train, y_train = _subset(data, "train")
    X_val, y_val = _subset(data, "val")

    model = LSTMForecaster(X_train.shape[-1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history.append({"epoch": epoch, "train_loss": total / len(X_train), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history).set_index("epoch")


def lstm_forecast(model: nn.Module, data: dict) -> pd.Series:
    """Forecast price for every sample: today's WTI + predicted change."""
    model.eval()
    with torch.no_grad():
        change = model(data["X"]).numpy() * data["y_std"]
    return pd.Series(data["current"] + change, index=data["dates"], name="lstm")

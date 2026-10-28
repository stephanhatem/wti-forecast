"""
Scoring forecasts against the 20-day-ahead WTI target.

Forecasts are scored only on rows marked `usable` in the daily table (the
target is observed and falls in the same split as the forecast date).
Errors are in dollars per barrel.

Usage:
    from src.evaluate import compare
    compare(daily, {"persistence": p, "futures_f1": f})

Leave the test split out until the models are final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual, forecast) -> float:
    """Mean absolute error: the typical size of a miss."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast) -> float:
    """Root mean squared error: like MAE, but big misses count for more."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def compare(daily: pd.DataFrame, forecasts: dict, splits=("train", "val"),
            same_rows: bool = False) -> pd.DataFrame:
    """Score several forecasts side by side.

    forecasts - {name: Series of forecasts indexed by date}
    same_rows - if True, score every model only on the days where all of
                them have a forecast, so the numbers are directly comparable.

    Returns one row per (split, model) with:
      n        - number of forecasts scored
      coverage - share of the split's usable rows that were scored
      MAE, RMSE
    """
    results = []
    for split in splits:
        rows = daily[daily["usable"] & (daily["split"] == split)]
        table = pd.DataFrame({name: f.reindex(rows.index) for name, f in forecasts.items()})
        shared = table.notna().all(axis=1)
        for name in forecasts:
            mask = shared if same_rows else table[name].notna()
            n = int(mask.sum())
            actual, pred = rows.loc[mask, "wti_target"], table.loc[mask, name]
            results.append({
                "split": split,
                "model": name,
                "n": n,
                "coverage": n / len(rows) if len(rows) else np.nan,
                "MAE": mae(actual, pred) if n else np.nan,
                "RMSE": rmse(actual, pred) if n else np.nan,
            })
    return pd.DataFrame(results).set_index(["split", "model"])

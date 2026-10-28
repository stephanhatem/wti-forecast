"""
Baseline forecasts for the WTI spot price 20 trading days ahead.

Each function takes the `daily` table from `src.data.build_dataset()` and
returns a Series of forecasts on the same dates, so every model can be
scored the same way with `src.evaluate.compare`.

  persistence - the price in 20 trading days will be today's price.
  futures     - the price in 20 trading days will be what the futures
                market is paying today for a given contract.

Open items:
  - The futures columns are empty after 2024-04-05 because EIA stopped
    publishing them. Until a newer source is added, the futures baseline
    covers only the start of validation and none of test.
  - The proposal defines the futures baseline as the contract whose delivery
    month is closest to the target date. That needs the NYMEX expiry
    calendar (see `futures_closest_contract`, not built yet), so for now the
    contract is chosen by hand.
"""
from __future__ import annotations

import pandas as pd

FUTURES_CONTRACTS = ("f1", "f2", "f3", "f4")


def persistence_forecast(daily: pd.DataFrame) -> pd.Series:
    """Forecast for t+20 is the WTI spot price at t."""
    return daily["wti"].rename("persistence")


def futures_forecast(daily: pd.DataFrame, contract: str = "f1") -> pd.Series:
    """Forecast for t+20 is the price of one futures contract at t.

    `contract` is "f1" (nearest delivery month) through "f4". The forecast is
    missing wherever the futures data is missing (all dates after 2024-04-05).
    """
    if contract not in FUTURES_CONTRACTS:
        raise ValueError(f"contract must be one of {FUTURES_CONTRACTS}, got {contract!r}")
    return daily[contract].rename(f"futures_{contract}")


def futures_closest_contract(daily: pd.DataFrame) -> pd.Series:
    """The proposal's futures baseline: on each date, use the contract whose
    delivery month is closest to that row's `target_date`.

    To build:
      1. Find the delivery month of f1 on each date. A WTI contract stops
         trading 3 business days before the 25th of the month before its
         delivery month (4 if the 25th isn't a business day), so f1 moves
         to the next month around the 20th.
      2. f2, f3 and f4 deliver 1, 2 and 3 months after f1.
      3. For each row, pick the contract whose delivery month is closest
         to `target_date`.
    """
    raise NotImplementedError("Needs the NYMEX expiry calendar; see the docstring.")

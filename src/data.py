"""
Data loading for the WTI forecasting project.

Builds one daily table, on the trading calendar, containing:
  - WTI and Brent spot prices (EIA, daily)
  - NYMEX WTI futures, contracts 1-4 (EIA, daily). EIA stopped publishing
    these after 2024-04-05, so the futures columns are empty after that date.
  - Weekly EIA supply/demand series, each placed on the date it was
    *released*, not the week it describes, so no forecast can see a number
    before it existed.
  - The forecast target: the WTI spot price 20 trading days later.

Usage:
    from src.data import build_dataset
    daily, weekly = build_dataset()

Things to know about the raw data:
  - WTI spot was -36.98 on 2020-04-20. That is a real price (the May 2020
    contract expiry), so it is kept. Avoid log prices/returns because of it.
  - EIA's WTI spot series is missing about 38 trading days between April and
    July 2011. Those days stay on the calendar (futures traded), inputs are
    forward-filled from the past, and the target is left missing.
  - Weekly files only list the week-ending Friday. Release dates are
    estimated: EIA publishes the following Wednesday, a day later for each
    federal holiday that falls Monday-Wednesday of the release week. Rare
    one-off delays are not captured.
  - The files hold EIA's current values, so any revisions are already in them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HORIZON = 20          # forecast horizon in trading days
FILL_LIMIT = 5        # max trading days to carry a daily price forward

# Current chronological splits from the proposal. A forecast belongs to the split of
# its origin date. Origins whose 20-day target lands in a later split are
# marked unusable to avoid data leakage across boundaries.
SPLITS = {
    "train": ("2010-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2025-08-31"),
    "test":  ("2025-09-01", "2026-08-31"),
    "live":  ("2026-09-01", "2026-12-31"),
}

WTI_FILE = "Cushing_OK_WTI_Spot_Price_FOB.csv"
BRENT_FILE = "Europe_Brent_Spot_Price_FOB.csv"
FUTURES_FILE = "PET_PRI_FUT_S1_D.xls"
WEEKLY_FILE = "PET_SUM_SNDW_DCUS_NUS_W.xls"

# (sheet, EIA column name) -> short name
WEEKLY_SERIES = {
    ("Data 1", "Weekly U.S. Field Production of Crude Oil (Thousand Barrels per Day)"): "crude_production",
    ("Data 2", "Weekly U.S. Refiner Net Input of Crude Oil (Thousand Barrels per Day)"): "refinery_crude_input",
    ("Data 2", "Weekly U.S. Percent Utilization of Refinery Operable Capacity (Percent)"): "refinery_utilization",
    ("Data 6", "Weekly U.S. Ending Stocks excluding SPR of Crude Oil (Thousand Barrels)"): "commercial_crude_stocks",
    ("Data 10", "Weekly U.S. Net Imports of Crude Oil (Thousand Barrels per Day)"): "crude_net_imports",
}


# ----- helpers

def _find(filename: str) -> Path:
    """Look for a raw file."""
    for folder in (DATA_DIR / "raw", DATA_DIR):
        path = folder / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} not found in data/raw or data/")


def _clean(name: str) -> str:
    """Collapse double spaces in column names."""
    return " ".join(str(name).split())


def _align(obj, calendar: pd.DatetimeIndex, limit: int = FILL_LIMIT):
    """Put a daily series on the calendar, filling short gaps from the past
    only, and never filling past the series' last real observation."""
    last = obj.dropna(how="all").index.max()
    full = obj.reindex(obj.index.union(calendar)).ffill(limit=limit)
    out = full.reindex(calendar)
    out.loc[out.index > last] = float("nan")
    return out


def _split_of(dates) -> pd.Series:
    """Label each data point by its split."""
    dates = pd.Series(pd.to_datetime(dates))
    out = pd.Series(None, index=dates.index, dtype="object")
    for name, (start, end) in SPLITS.items():
        out[(dates >= start) & (dates <= end)] = name
    return out


# ----- loaders

def load_spot(filename: str) -> pd.Series:
    """EIA spot CSV: 4 lines of metadata, then 'Day,price', newest first."""
    df = pd.read_csv(_find(filename), skiprows=4)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df.set_index("date")["price"].sort_index()


def load_futures() -> pd.DataFrame:
    """NYMEX WTI futures, contracts 1-4 (sheet 'Data 1'). Ends 2024-04-05."""
    df = pd.read_excel(_find(FUTURES_FILE), sheet_name="Data 1", header=2)
    df.columns = ["date", "f1", "f2", "f3", "f4"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def estimate_release_dates(week_end: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Week ending Friday -> released the following Wednesday, one day later
    per federal holiday on Monday-Wednesday of that week."""
    holidays = set(USFederalHolidayCalendar().holidays(
        start=week_end.min(), end=week_end.max() + pd.Timedelta(days=14)))
    releases = []
    for friday in week_end:
        release = friday + pd.Timedelta(days=5)
        delay = sum((friday + pd.Timedelta(days=d)) in holidays for d in (3, 4, 5))
        release += pd.Timedelta(days=delay)
        while release in holidays or release.dayofweek >= 5:
            release += pd.Timedelta(days=1)
        releases.append(release)
    return pd.DatetimeIndex(releases)


def load_weekly() -> pd.DataFrame:
    """Selected weekly series, indexed by week-ending date, with an
    estimated release_date column."""
    path = _find(WEEKLY_FILE)
    sheets = {}
    columns = []
    for (sheet, eia_name), short in WEEKLY_SERIES.items():
        if sheet not in sheets:
            raw = pd.read_excel(path, sheet_name=sheet, header=2)
            raw.columns = ["week_end"] + [_clean(c) for c in raw.columns[1:]]
            sheets[sheet] = raw.set_index(pd.to_datetime(raw["week_end"]))
        columns.append(sheets[sheet][_clean(eia_name)].rename(short))
    weekly = pd.concat(columns, axis=1).sort_index()
    weekly.index.name = "week_end"
    weekly.insert(0, "release_date", estimate_release_dates(weekly.index))
    return weekly


# ----- main build

def build_dataset(start: str = "2009-01-01", save: bool = False):
    """Return (daily, weekly).

    daily  - one row per trading day from `start`, with prices, futures,
             the latest *released* weekly values, the 20-day target, and
             split labels. Use rows where `usable` is True for training
             and evaluation.
    weekly - the weekly series with week_end index and release_date, for
             building "last 12 released weeks" inputs (see weekly_history).

    `start` defaults to 2009 so models have a warm-up window (60 trading
    days, 12 weeks) before the training period begins in 2010.
    """
    wti = load_spot(WTI_FILE)
    brent = load_spot(BRENT_FILE)
    futures = load_futures()
    weekly = load_weekly()

    # Trading calendar: any day with a WTI spot or futures price.
    calendar = wti.index.union(futures.index)
    calendar = calendar[calendar >= start]

    daily = pd.DataFrame(index=calendar)
    daily.index.name = "date"
    daily["wti_raw"] = wti.reindex(calendar)
    daily["wti"] = _align(wti, calendar)
    daily["wti_filled"] = daily["wti_raw"].isna() & daily["wti"].notna()
    daily["brent"] = _align(brent, calendar)
    daily[["f1", "f2", "f3", "f4"]] = _align(futures, calendar)

    # Latest weekly values that had been released by each trading day.
    released = weekly.reset_index().sort_values("release_date")
    asof = pd.merge_asof(
        pd.DataFrame({"date": calendar}), released,
        left_on="date", right_on="release_date", direction="backward",
    ).set_index("date")
    daily["weekly_week_end"] = asof["week_end"]
    for short in WEEKLY_SERIES.values():
        daily[short] = asof[short]

    # Target: WTI spot 20 trading days later (left missing if not observed).
    daily["target_date"] = daily.index.to_series().shift(-HORIZON)
    daily["wti_target"] = daily["wti_raw"].shift(-HORIZON)

    # Splits, with no target crossing into a later split.
    daily["split"] = _split_of(daily.index).values
    target_split = _split_of(daily["target_date"]).values
    daily["usable"] = (
        daily["split"].notna()
        & (daily["split"].values == target_split)
        & daily["wti_target"].notna()
        & daily["wti"].notna()
    )

    if save:
        out = DATA_DIR / "processed"
        out.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out / "daily.csv")
        weekly.to_csv(out / "weekly.csv")

    return daily, weekly


def weekly_history(weekly: pd.DataFrame, as_of, n: int = 12) -> pd.DataFrame:
    """The last `n` weekly observations released on or before `as_of`."""
    return weekly[weekly["release_date"] <= pd.Timestamp(as_of)].tail(n)


def coverage(daily: pd.DataFrame) -> pd.DataFrame:
    """First/last date and missing share for each column, from 2010 on."""
    cols = ["wti", "brent", "f1", "f2", "f3", "f4", *WEEKLY_SERIES.values()]
    recent = daily.loc["2010":]
    return pd.DataFrame({
        "first": [daily[c].first_valid_index() for c in cols],
        "last": [daily[c].last_valid_index() for c in cols],
        "missing_since_2010": [f"{recent[c].isna().mean():.1%}" for c in cols],
    }, index=cols)

"""
LLM forecaster: a frozen LLM (gpt-5.4-mini, used as-is through the OpenAI
API) forecasts the WTI spot price 20 trading days ahead from the same
information the LSTM sees, written out as text.

Input:  a prompt with the last 60 trading days of WTI and Brent spot prices
        and the last 12 weekly EIA reports released by the forecast date.
Output: JSON {"forecast": <price in $/barrel>, "reason": <one sentence>}.
        A JSON schema is enforced, so the reply always parses.
Nothing is trained.

Look-ahead risk: gpt-5.4-mini's knowledge runs to August 31, 2025, which
covers the whole validation period (Jan 2024 - Aug 2025), so it may remember
what oil actually did. By default the prompt uses relative days
("day -59 ... day 0") instead of calendar dates. Price levels can still hint
at the period, so read LLM validation scores with care. The test period
(from Sep 2025) is after the model's knowledge cutoff.

Every reply is saved in results/llm_logs/ with its prompt and settings, so
each forecast is paid for once and re-running the notebook is free.

Needs OPENAI_API_KEY in the project's .env file (see .env.example).

Usage:
    from src.llm_forecast import build_prompt, sample_dates, llm_forecasts
    dates = sample_dates(daily, "val")
    print(build_prompt(daily, weekly, dates[0]))
    llm = llm_forecasts(daily, weekly, dates)    # columns: forecast, reason
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from .data import HORIZON, ROOT, weekly_history

MODEL = "gpt-5.4-mini"
EFFORT = "low"            # reasoning effort: none, low, medium, high, xhigh
LOG_DIR = ROOT / "results" / "llm_logs"
DAYS, WEEKS = 60, 12      # same history the LSTM gets

INSTRUCTIONS = (
    "You are a commodity market analyst. You forecast the WTI crude oil spot "
    "price at Cushing, Oklahoma, in US dollars per barrel. Use only the data "
    "you are given."
)

WEEKLY_COLUMNS = {
    "crude_production": "crude production (thousand b/d)",
    "refinery_crude_input": "refinery crude input (thousand b/d)",
    "refinery_utilization": "refinery utilization (%)",
    "commercial_crude_stocks": "commercial crude stocks (thousand barrels)",
    "crude_net_imports": "crude net imports (thousand b/d)",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "forecast": {"type": "number", "description": "WTI spot price in $/barrel"},
        "reason": {"type": "string", "description": "One short sentence"},
    },
    "required": ["forecast", "reason"],
    "additionalProperties": False,
}


def build_prompt(daily: pd.DataFrame, weekly: pd.DataFrame, date,
                 hide_dates: bool = True) -> str:
    """The exact text the model receives for a forecast made on `date`."""
    date = pd.Timestamp(date)
    prices = daily.loc[:date, ["wti", "brent"]].ffill().tail(DAYS)
    reports = weekly_history(weekly, date, n=WEEKS)

    if hide_dates:
        today = "Day 0 is today."
        day_col, day_labels = "day", range(1 - len(prices), 1)
        week_col, week_labels = "report", range(1 - len(reports), 1)
    else:
        today = f"Today is {date:%Y-%m-%d}."
        day_col, day_labels = "date", [f"{d:%Y-%m-%d}" for d in prices.index]
        week_col, week_labels = "week ending", [f"{d:%Y-%m-%d}" for d in reports.index]

    price_rows = "\n".join(
        f"{label},{wti:.2f},{brent:.2f}"
        for label, (wti, brent) in zip(day_labels, prices.to_numpy()))
    report_rows = "\n".join(
        f"{label}," + ",".join(f"{v:.1f}" for v in values)
        for label, values in zip(week_labels, reports[list(WEEKLY_COLUMNS)].to_numpy()))

    return (
        f"{today} Recent market data is below, oldest first.\n\n"
        f"Daily spot prices ($/barrel), last {len(prices)} trading days:\n"
        f"{day_col},WTI,Brent\n{price_rows}\n\n"
        f"Weekly U.S. EIA reports, the last {len(reports)} released as of today:\n"
        f"{week_col}," + ",".join(WEEKLY_COLUMNS.values()) + f"\n{report_rows}\n\n"
        f"Forecast the WTI spot price {HORIZON} trading days from today "
        f"(about four weeks ahead)."
    )


def sample_dates(daily: pd.DataFrame, split: str = "val", every: int = HORIZON):
    """Every `every`-th usable forecast date in a split. Spacing them 20
    trading days apart means their forecast windows don't overlap."""
    dates = daily.index[daily["usable"] & (daily["split"] == split)]
    return dates[::every]


def _client():
    """OpenAI client using the key in .env."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("No OPENAI_API_KEY found. Copy .env.example to .env and add your key.")
    return OpenAI()


def ask_llm(prompt: str, model: str = MODEL, effort: str = EFFORT, client=None) -> dict:
    """One API call. Returns {"forecast", "reason", "input_tokens", "output_tokens"}."""
    client = client or _client()
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=prompt,
        reasoning={"effort": effort},
        text={"format": {"type": "json_schema", "name": "wti_forecast",
                         "schema": SCHEMA, "strict": True}},
        store=False,
    )
    reply = json.loads(response.output_text)
    reply["input_tokens"] = response.usage.input_tokens
    reply["output_tokens"] = response.usage.output_tokens
    return reply


def llm_forecasts(daily: pd.DataFrame, weekly: pd.DataFrame, dates,
                  model: str = MODEL, effort: str = EFFORT,
                  hide_dates: bool = True, client=None) -> pd.DataFrame:
    """Forecast every date in `dates`, reusing saved replies when the prompt
    and settings are unchanged. Returns a DataFrame indexed by date with the
    columns forecast and reason."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(dates)
    rows, cached, tokens, failures = {}, 0, [0, 0], 0

    for i, date in enumerate(dates, 1):
        prompt = build_prompt(daily, weekly, date, hide_dates=hide_dates)
        key = hashlib.sha256(f"{model}|{effort}|{INSTRUCTIONS}|{prompt}".encode()).hexdigest()[:8]
        path = LOG_DIR / f"{date:%Y-%m-%d}_{model}_{key}.json"

        if path.exists():
            log = json.loads(path.read_text())
            cached += 1
        else:
            print(f"\rasking {model}: {i}/{len(dates)}", end="", flush=True)
            client = client or _client()
            try:
                reply = ask_llm(prompt, model, effort, client)
            except Exception as err:     # one failed call shouldn't stop the run...
                print(f"\n{date:%Y-%m-%d}: request failed ({err})")
                failures += 1
                if failures == 3:        # ...but three in a row means something's wrong
                    print("Stopping after 3 failed requests in a row.")
                    break
                continue
            failures = 0
            tokens[0] += reply["input_tokens"]
            tokens[1] += reply["output_tokens"]
            log = {"date": f"{date:%Y-%m-%d}", "model": model, "effort": effort,
                   "hide_dates": hide_dates,
                   "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **reply, "prompt": prompt}
            path.write_text(json.dumps(log, indent=1))

        rows[date] = {"forecast": float(log["forecast"]), "reason": log["reason"]}

    print(f"\r{len(rows)} forecasts: {cached} from saved logs, {len(rows) - cached} new "
          f"({tokens[0]:,} input / {tokens[1]:,} output tokens)")
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["forecast", "reason"])
    out.index.name = "date"
    return out

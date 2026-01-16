#!/usr/bin/env python3
"""Compute valuation metrics and historical percentiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Iterable

import numpy as np
import polars as pl
import yfinance as yf

from series_utils import (
    empty_series,
    latest_value,
    rows_from_payload,
    series_from_mapping,
    series_from_rows,
    series_rows,
    series_to_dict,
)


def to_series(data: Dict[str, Any]) -> pl.DataFrame:
    return series_from_mapping(data or {})


def find_matching_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(key).lower(): str(key) for key in keys}
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in lookup:
            return lookup[lowered]
    return None


def to_price_series(payload: Dict[str, Any]) -> pl.DataFrame:
    price_payload = payload.get("price_history", {}) or {}
    if not price_payload:
        return empty_series()
    date_key = next(
        (key for key in ["日期", "date", "Date"] if key in price_payload), None
    )
    candidates = ["Close", "Adj Close", "收盘", "close"]
    if date_key:
        value_key = find_matching_key(
            [key for key in price_payload.keys() if key != date_key], candidates
        )
        if not value_key:
            return empty_series()
        rows = rows_from_payload(price_payload, date_key)
        return series_from_rows(rows, date_key, value_key)
    for key in candidates:
        column_map = price_payload.get(key)
        if isinstance(column_map, dict):
            return series_from_mapping(column_map)
    return empty_series()


def align_to_prices(snapshot: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    if prices.height == 0:
        return empty_series()
    prices_sorted = prices.sort("date")
    if snapshot.height == 0:
        return prices_sorted.select(["date"]).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("value")
        )
    snapshot_sorted = snapshot.sort("date")
    aligned = prices_sorted.rename({"value": "price"}).join_asof(
        snapshot_sorted.rename({"value": "snapshot"}),
        on="date",
        strategy="backward",
    )
    return aligned.select(["date", "snapshot"]).rename({"snapshot": "value"})


def fetch_fx_rate(base: Optional[str], quote: Optional[str]) -> Optional[float]:
    if not base or not quote or base == quote:
        return 1.0
    pair = f"{base}{quote}=X"
    for symbol, invert in [(pair, False), (f"{quote}{base}=X", True)]:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="5d", auto_adjust=False)
            close = history.get("Close") if hasattr(history, "get") else None
            if close is None:
                continue
            values = [float(val) for val in list(close) if np.isfinite(val)]
            if not values:
                continue
            rate = values[-1]
            if rate == 0:
                continue
            return 1 / rate if invert else rate
        except Exception:
            continue
    return None


def convert_series(
    series: pl.DataFrame, fx_rate: Optional[float], apply_conversion: bool
) -> pl.DataFrame:
    if series.height == 0 or not apply_conversion:
        return series
    if fx_rate is None:
        return empty_series()
    return series.with_columns((pl.col("value") * fx_rate).alias("value"))


def percentile(current: Optional[float], history: pl.DataFrame) -> Optional[float]:
    if current is None or history.height == 0:
        return None
    values = [
        value
        for _, value in series_rows(history)
        if value is not None and np.isfinite(value) and value > 0
    ]
    if not values:
        return None
    return float(sum(1 for value in values if value <= current) / len(values) * 100)


def join_series(
    left: pl.DataFrame, right: pl.DataFrame, left_name: str, right_name: str
) -> pl.DataFrame:
    if left.height == 0 or right.height == 0:
        return pl.DataFrame()
    left_df = left.rename({"value": left_name})
    right_df = right.rename({"value": right_name})
    return left_df.join(right_df, on="date", how="inner")


def divide_series(
    numerator: pl.DataFrame, denominator: pl.DataFrame, positive_only: bool = False
) -> pl.DataFrame:
    aligned = join_series(numerator, denominator, "num", "den")
    if aligned.height == 0:
        return empty_series()
    if positive_only:
        aligned = aligned.filter(pl.col("den") > 0)
    else:
        aligned = aligned.filter(pl.col("den") != 0)
    if aligned.height == 0:
        return empty_series()
    result = aligned.with_columns((pl.col("num") / pl.col("den")).alias("value"))
    return result.select(["date", "value"]).filter(pl.col("value").is_finite())


def add_series(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    aligned = join_series(left, right, "left", "right")
    if aligned.height == 0:
        return empty_series()
    result = aligned.with_columns((pl.col("left") + pl.col("right")).alias("value"))
    return result.select(["date", "value"]).filter(pl.col("value").is_finite())


def multiply_series(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    aligned = join_series(left, right, "left", "right")
    if aligned.height == 0:
        return empty_series()
    result = aligned.with_columns((pl.col("left") * pl.col("right")).alias("value"))
    return result.select(["date", "value"]).filter(pl.col("value").is_finite())


def compute_dcf(
    free_cash_flow: Optional[float],
    net_debt: Optional[float],
    shares_outstanding: Optional[float],
    discount_rate: float = 0.1,
    growth_rate: float = 0.05,
    terminal_growth: float = 0.02,
    years: int = 5,
) -> Dict[str, Any]:
    if free_cash_flow is None or free_cash_flow <= 0:
        return {}

    pv = 0.0
    for year in range(1, years + 1):
        pv += free_cash_flow * (1 + growth_rate) ** year / (1 + discount_rate) ** year
    terminal_value = (
        free_cash_flow
        * (1 + growth_rate) ** years
        * (1 + terminal_growth)
        / (discount_rate - terminal_growth)
    )
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    enterprise_value = pv + pv_terminal

    result: Dict[str, Any] = {
        "assumptions": {
            "discount_rate": discount_rate,
            "growth_rate": growth_rate,
            "terminal_growth": terminal_growth,
            "years": years,
        },
        "enterprise_value": float(enterprise_value),
    }

    if net_debt is not None:
        equity_value = enterprise_value - net_debt
        result["net_debt"] = float(net_debt)
        result["equity_value"] = float(equity_value)
        if shares_outstanding:
            result["per_share"] = float(equity_value / shares_outstanding)

    return result


def build_valuation(data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    info = data.get("info", {}) or {}
    market_currency = info.get("currency")
    financial_currency = info.get("financialCurrency") or market_currency

    price_series = to_price_series(data)

    currency_mismatch = bool(
        market_currency and financial_currency and market_currency != financial_currency
    )
    fx_rate = (
        fetch_fx_rate(financial_currency, market_currency) if currency_mismatch else 1.0
    )

    eps_ttm = to_series(analysis.get("per_share_ttm", {}).get("eps", {}))
    sales_ttm = to_series(analysis.get("per_share_ttm", {}).get("sales", {}))
    ebitda_ttm = to_series(analysis.get("per_share_ttm", {}).get("ebitda", {}))

    book_per_share = to_series(
        analysis.get("balance_quarterly", {}).get("book_per_share", {})
    )
    net_debt_per_share = to_series(
        analysis.get("balance_quarterly", {}).get("net_debt_per_share", {})
    )
    shares_outstanding = to_series(
        analysis.get("balance_quarterly", {}).get("shares_outstanding", {})
    )

    eps_ttm = convert_series(eps_ttm, fx_rate, currency_mismatch)
    sales_ttm = convert_series(sales_ttm, fx_rate, currency_mismatch)
    ebitda_ttm = convert_series(ebitda_ttm, fx_rate, currency_mismatch)
    book_per_share = convert_series(book_per_share, fx_rate, currency_mismatch)
    net_debt_per_share = convert_series(net_debt_per_share, fx_rate, currency_mismatch)

    eps_daily = align_to_prices(eps_ttm, price_series)
    sales_daily = align_to_prices(sales_ttm, price_series)
    ebitda_daily = align_to_prices(ebitda_ttm, price_series)
    book_daily = align_to_prices(book_per_share, price_series)
    net_debt_daily = align_to_prices(net_debt_per_share, price_series)
    shares_daily = align_to_prices(shares_outstanding, price_series)

    pe_daily = divide_series(price_series, eps_daily, positive_only=True)
    ps_daily = divide_series(price_series, sales_daily, positive_only=True)
    pb_daily = divide_series(price_series, book_daily, positive_only=True)
    ev_to_ebitda_daily = divide_series(
        add_series(price_series, net_debt_daily), ebitda_daily, positive_only=True
    )

    market_cap_daily = multiply_series(price_series, shares_daily)

    price_rows = series_rows(price_series)
    latest_date = price_rows[-1][0] if price_rows else None
    current_price = price_rows[-1][1] if price_rows else None
    current_market_cap = latest_value(market_cap_daily)

    current_metrics = {
        "pe": latest_value(pe_daily),
        "ps": latest_value(ps_daily),
        "pb": latest_value(pb_daily),
        "ev_to_ebitda": latest_value(ev_to_ebitda_daily),
    }

    metric_dates = {
        dt for dt, _ in series_rows(pe_daily)
    } | {dt for dt, _ in series_rows(ps_daily)} | {dt for dt, _ in series_rows(pb_daily)} | {
        dt for dt, _ in series_rows(ev_to_ebitda_daily)
    }
    window_start = min(metric_dates) if metric_dates else None

    fcf_ttm_total = to_series(
        analysis.get("financials_ttm", {}).get("free_cash_flow", {})
    )
    net_debt_total = to_series(
        analysis.get("balance_quarterly", {}).get("net_debt", {})
    )

    fcf_ttm_total = convert_series(fcf_ttm_total, fx_rate, currency_mismatch)
    net_debt_total = convert_series(net_debt_total, fx_rate, currency_mismatch)

    fcf_latest = latest_value(align_to_prices(fcf_ttm_total, price_series))
    net_debt_latest = latest_value(align_to_prices(net_debt_total, price_series))
    shares_latest = latest_value(align_to_prices(shares_outstanding, price_series))

    valuation = {
        "symbol": analysis.get("symbol"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "window": {
            "start": str(window_start.date()) if window_start is not None else None,
            "end": str(latest_date.date()) if latest_date is not None else None,
            "price_points": int(len(price_rows)),
            "valuation_days": int(len(metric_dates)),
            "snapshot_points": {
                "eps_ttm": int(eps_ttm.height),
                "sales_ttm": int(sales_ttm.height),
                "ebitda_ttm": int(ebitda_ttm.height),
                "book_per_share": int(book_per_share.height),
                "net_debt_per_share": int(net_debt_per_share.height),
                "shares_outstanding": int(shares_outstanding.height),
            },
        },
        "current": {
            "date": str(latest_date.date()) if latest_date is not None else None,
            "price": current_price,
            "market_cap": current_market_cap,
        },
        "metrics": current_metrics,
        "history": {
            "pe": series_to_dict(pe_daily),
            "ps": series_to_dict(ps_daily),
            "pb": series_to_dict(pb_daily),
            "ev_to_ebitda": series_to_dict(ev_to_ebitda_daily),
        },
        "percentiles": {
            "pe": percentile(current_metrics["pe"], pe_daily),
            "ps": percentile(current_metrics["ps"], ps_daily),
            "pb": percentile(current_metrics["pb"], pb_daily),
            "ev_to_ebitda": percentile(
                current_metrics["ev_to_ebitda"], ev_to_ebitda_daily
            ),
        },
        "currency": {
            "market": market_currency,
            "financial": financial_currency,
            "fx_rate": fx_rate,
            "converted": bool(currency_mismatch and fx_rate is not None),
        },
        "dcf": compute_dcf(
            fcf_latest,
            net_debt_latest,
            shares_latest,
        ),
    }

    return valuation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute valuation metrics")
    parser.add_argument("--input", required=True, help="Data JSON path")
    parser.add_argument("--analysis", required=True, help="Analysis JSON path")
    parser.add_argument("--output", default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    with open(args.analysis, "r", encoding="utf-8") as handle:
        analysis = json.load(handle)

    valuation = build_valuation(data, analysis)
    output_path = f"{args.output}/{analysis['symbol'].replace('.', '_')}_valuation.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(valuation, handle, ensure_ascii=False, indent=2)

    print(f"Saved valuation to {output_path}")


if __name__ == "__main__":
    main()

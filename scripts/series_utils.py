#!/usr/bin/env python3
"""Utility helpers for date-series handling with polars."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import polars as pl


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        raw = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = None
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return None
    else:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def empty_series() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.Series([], dtype=pl.Datetime),
            "value": pl.Series([], dtype=pl.Float64),
        }
    )


def series_from_mapping(mapping: Dict[str, Any]) -> pl.DataFrame:
    if not mapping:
        return empty_series()
    rows: List[Tuple[datetime, Optional[float]]] = []
    for key, value in mapping.items():
        parsed = parse_datetime(key)
        if parsed is None:
            continue
        rows.append((parsed, to_float(value)))
    if not rows:
        return empty_series()
    df = pl.DataFrame(rows, schema=["date", "value"], orient="row")
    return df.drop_nulls().sort("date")


def series_from_rows(
    rows: Iterable[Dict[str, Any]], date_key: str, value_key: str
) -> pl.DataFrame:
    series_rows: List[Tuple[datetime, Optional[float]]] = []
    for row in rows:
        parsed = parse_datetime(row.get(date_key))
        if parsed is None:
            continue
        series_rows.append((parsed, to_float(row.get(value_key))))
    if not series_rows:
        return empty_series()
    df = pl.DataFrame(series_rows, schema=["date", "value"], orient="row")
    return df.drop_nulls().sort("date")


def series_rows(series: pl.DataFrame) -> List[Tuple[datetime, float]]:
    if series is None or series.height == 0:
        return []
    df = series.drop_nulls().sort("date").filter(pl.col("value").is_finite())
    return [(row[0], float(row[1])) for row in df.select(["date", "value"]).iter_rows()]


def series_to_dict(series: pl.DataFrame) -> Dict[str, float]:
    rows = series_rows(series)
    if not rows:
        return {}
    return {
        (dt.date().isoformat() if isinstance(dt, datetime) else str(dt)): float(value)
        for dt, value in rows
    }


def latest_value(series: pl.DataFrame) -> Optional[float]:
    rows = series_rows(series)
    if not rows:
        return None
    return float(rows[-1][1])


def rows_from_payload(
    payload: Dict[str, Dict[str, Any]], row_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not payload:
        return []
    row_ids: List[str] = []
    if row_key and row_key in payload and isinstance(payload[row_key], dict):
        row_ids = list(payload[row_key].keys())
    if not row_ids:
        first_key = next(iter(payload), None)
        if first_key and isinstance(payload.get(first_key), dict):
            row_ids = list(payload[first_key].keys())
    rows: List[Dict[str, Any]] = []
    for row_id in row_ids:
        row: Dict[str, Any] = {}
        for column, column_map in payload.items():
            if isinstance(column_map, dict):
                row[column] = column_map.get(row_id)
        rows.append(row)
    return rows

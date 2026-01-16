#!/usr/bin/env python3
"""Analyze analyst expectations from fetched data."""

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from series_utils import parse_datetime

logger = logging.getLogger(__name__)

BUY_KEYWORDS = {"buy", "strong buy", "overweight", "outperform", "add"}
HOLD_KEYWORDS = {"hold", "neutral", "market perform", "equal-weight"}
SELL_KEYWORDS = {"sell", "underperform", "underweight", "reduce"}


def grade_bucket(grade: str) -> str:
    normalized = grade.strip().lower()
    if any(key in normalized for key in BUY_KEYWORDS):
        return "buy"
    if any(key in normalized for key in HOLD_KEYWORDS):
        return "hold"
    if any(key in normalized for key in SELL_KEYWORDS):
        return "sell"
    return "other"


def summarize_recommendations(recommendations: dict[str, Any]) -> dict[str, int]:
    if not recommendations or not isinstance(recommendations, dict):
        return {}
    grades = recommendations.get("To Grade", {})
    if not isinstance(grades, dict) or not grades:
        return {}
    rows = []
    for date_key, grade in grades.items():
        parsed = parse_datetime(date_key)
        if parsed is None:
            continue
        rows.append((parsed, str(grade)))
    if not rows:
        return {}
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
    recent = [row for row in rows if row[0] >= cutoff]
    if not recent:
        recent = rows
    buckets = {"buy": 0, "hold": 0, "sell": 0, "other": 0}
    for _, grade in recent:
        bucket = grade_bucket(grade)
        buckets[bucket] += 1
    return {key: value for key, value in buckets.items() if value > 0}


def normalize_summary_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    return str(value)


def summarize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary or not isinstance(summary, dict):
        return {}
    latest: dict[str, Any] = {}
    for key, column_map in summary.items():
        if isinstance(column_map, dict) and column_map:
            value = next(iter(column_map.values()))
        else:
            value = column_map
        latest[key] = normalize_summary_value(value)
    return latest


def build_analyst_report(data: dict[str, Any]) -> dict[str, Any]:
    info = data.get("info", {}) or {}
    analyst = data.get("analyst", {}) or {}

    recommendations = analyst.get("recommendations", {})
    recommendations_summary = analyst.get("recommendations_summary", {})

    return {
        "symbol": data.get("symbol"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rating": {
            "recommendation_key": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
            "summary": summarize_summary(recommendations_summary),
            "recent_distribution": summarize_recommendations(recommendations),
        },
        "price_targets": {
            "mean": info.get("targetMeanPrice"),
            "high": info.get("targetHighPrice"),
            "low": info.get("targetLowPrice"),
            "current": info.get("currentPrice"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze analyst expectations")
    parser.add_argument("--input", required=True, help="Data JSON path")
    parser.add_argument("--output", default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)

    analyst_report = build_analyst_report(data)
    output_path = f"{args.output}/{data['symbol'].replace('.', '_')}_analyst.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(analyst_report, handle, ensure_ascii=False, indent=2)

    logger.info(f"Saved analyst report to {output_path}")


if __name__ == "__main__":
    main()

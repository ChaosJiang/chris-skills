#!/usr/bin/env python3
"""Run the full report pipeline with caching and market inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import analyst as analyst_module
import analyze as analyze_module
import fetch_data as fetch_data_module
import report as report_module
import valuation as valuation_module
import visualize as visualize_module


CHART_FILES = [
    "revenue_net_income.png",
    "margin_trends.png",
    "roe_roa.png",
    "debt_to_equity.png",
    "price_history.png",
]


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hours_since(timestamp: datetime) -> float:
    return (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def is_fresh(path: Path, max_age_hours: float) -> bool:
    if max_age_hours <= 0 or not path.exists():
        return False
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    fetched_at = parse_iso_datetime(payload.get("fetched_at"))
    if fetched_at is None:
        age_hours = hours_since(
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        )
    else:
        age_hours = hours_since(fetched_at)
    return age_hours <= max_age_hours


def needs_update(output_path: Path, input_mtimes: Iterable[float]) -> bool:
    if not output_path.exists():
        return True
    latest_input = max(input_mtimes, default=0)
    return output_path.stat().st_mtime < latest_input


def charts_need_update(charts_dir: Path, analysis_mtime: float) -> bool:
    if not charts_dir.exists():
        return True
    for filename in CHART_FILES:
        chart_path = charts_dir / filename
        if not chart_path.exists() or chart_path.stat().st_mtime < analysis_mtime:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full report with caching and optional steps"
    )
    parser.add_argument("--symbol", required=True, help="Stock symbol")
    parser.add_argument("--market", choices=["US", "CN", "HK", "JP"])
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--price-years", type=int, default=None)
    parser.add_argument("--output", default="./output")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=24,
        help="Reuse cached data if fetched within this window (0 disables cache).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refetch data even if cache is fresh.",
    )
    parser.add_argument("--skip-valuation", action="store_true")
    parser.add_argument("--skip-analyst", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = fetch_data_module.normalize_symbol(args.symbol)
    market = (args.market or fetch_data_module.infer_market(symbol)).upper()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace(".", "_")
    data_path = output_dir / f"{safe_symbol}_data.json"
    analysis_path = output_dir / f"{safe_symbol}_analysis.json"
    valuation_path = output_dir / f"{safe_symbol}_valuation.json"
    analyst_path = output_dir / f"{safe_symbol}_analyst.json"
    report_path = output_dir / f"{safe_symbol}_report.md"
    charts_dir = output_dir / f"{safe_symbol}_charts"

    price_years = args.price_years
    if price_years is None:
        price_years = max(args.years, 6)

    if not args.refresh and is_fresh(data_path, args.max_age_hours):
        data_payload = read_json(data_path)
        print(f"Using cached data: {data_path}")
    else:
        data_payload = fetch_data_module.fetch_data(
            symbol, market, args.years, price_years
        )
        data_payload.update(
            {
                "symbol": symbol,
                "market": market,
                "fetched_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )
        write_json(data_path, data_payload)
        print(f"Saved data to {data_path}")

    data_mtime = data_path.stat().st_mtime

    if needs_update(analysis_path, [data_mtime]):
        analysis_payload = analyze_module.build_analysis(data_payload)
        write_json(analysis_path, analysis_payload)
        print(f"Saved analysis to {analysis_path}")
    else:
        analysis_payload = read_json(analysis_path)
        print(f"Using cached analysis: {analysis_path}")

    analysis_mtime = analysis_path.stat().st_mtime

    valuation_payload: Dict[str, Any] = {}
    if not args.skip_valuation:
        if needs_update(valuation_path, [data_mtime, analysis_mtime]):
            valuation_payload = valuation_module.build_valuation(
                data_payload, analysis_payload
            )
            write_json(valuation_path, valuation_payload)
            print(f"Saved valuation to {valuation_path}")
        else:
            valuation_payload = read_json(valuation_path)
            print(f"Using cached valuation: {valuation_path}")

    analyst_payload: Dict[str, Any] = {}
    if not args.skip_analyst:
        if needs_update(analyst_path, [data_mtime]):
            analyst_payload = analyst_module.build_analyst_report(data_payload)
            write_json(analyst_path, analyst_payload)
            print(f"Saved analyst report to {analyst_path}")
        else:
            analyst_payload = read_json(analyst_path)
            print(f"Using cached analyst report: {analyst_path}")

    if not args.skip_charts:
        if charts_need_update(charts_dir, analysis_mtime):
            visualize_module.generate_charts(analysis_payload, str(charts_dir))
        else:
            print(f"Using cached charts: {charts_dir}")

    if not args.skip_report:
        report_inputs = [analysis_mtime]
        if not args.skip_valuation and valuation_path.exists():
            report_inputs.append(valuation_path.stat().st_mtime)
        if not args.skip_analyst and analyst_path.exists():
            report_inputs.append(analyst_path.stat().st_mtime)
        if needs_update(report_path, report_inputs):
            report_text = report_module.build_report(
                analysis_payload, valuation_payload, analyst_payload
            )
            report_path.write_text(report_text, encoding="utf-8")
            print(f"Saved report to {report_path}")
        else:
            print(f"Using cached report: {report_path}")


if __name__ == "__main__":
    main()

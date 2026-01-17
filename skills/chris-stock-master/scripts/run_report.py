#!/usr/bin/env python3
"""Run the full report pipeline with caching and market inference."""

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import analyst as analyst_module
import analyze as analyze_module
import config
import fetch_data as fetch_data_module
import report as report_module
import valuation as valuation_module
import visualize as visualize_module
from exceptions import FinancialReportError, format_error_for_user
from logging_config import get_module_logger, setup_logging
from progress import step_progress

logger = get_module_logger()


CHART_FILES = [
    "revenue_net_income.png",
    "margin_trends.png",
    "roe_roa.png",
    "debt_to_equity.png",
    "price_history.png",
    "peg_ratio.png",
]


def parse_iso_datetime(value: str | None) -> datetime | None:
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
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


def charts_need_update(charts_dir: Path, input_mtimes: Iterable[float]) -> bool:
    if not charts_dir.exists():
        return True
    latest_input = max(input_mtimes, default=0)
    for filename in CHART_FILES:
        chart_path = charts_dir / filename
        if not chart_path.exists() or chart_path.stat().st_mtime < latest_input:
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
    parser.add_argument("--output", default=str(config.OUTPUT_DIR))

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without executing (no API calls, no file writes)",
    )
    parser.add_argument("--skip-valuation", action="store_true")
    parser.add_argument("--skip-analyst", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Set up logging
    _, dq_logger = setup_logging(log_level="INFO", log_to_file=True)
    analyze_module.data_quality_logger = dq_logger

    args = parse_args()

    try:
        symbol = fetch_data_module.normalize_symbol(args.symbol)
        market = (args.market or fetch_data_module.infer_market(symbol)).upper()
        base_output_dir = Path(args.output)

        safe_symbol = symbol.replace(".", "_")
        company_folder = f"{safe_symbol}_{market}"
        output_dir = base_output_dir / company_folder

        data_path = output_dir / "data.json"
        analysis_path = output_dir / "analysis.json"
        valuation_path = output_dir / "valuation.json"
        analyst_path = output_dir / "analyst.json"
        report_path = output_dir / "report.md"
        charts_dir = output_dir / "charts"

        price_years = args.price_years
        if price_years is None:
            price_years = max(args.years, 6)

        # Dry-run mode: preview operations
        if args.dry_run:
            logger.info(f"DRY RUN MODE - Preview of operations for {symbol}")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Market: {market}")
            logger.info(f"Output directory: {output_dir}")
            logger.info("Operations that would be performed:")

            if not args.refresh and is_fresh(data_path, args.max_age_hours):
                data_payload = read_json(data_path)
                fetched_at = parse_iso_datetime(data_payload.get("fetched_at"))
                if fetched_at is None:
                    fetched_at = datetime.fromtimestamp(
                        data_path.stat().st_mtime, tz=timezone.utc
                    )
                age_hours = hours_since(fetched_at)
                logger.info(f"  - Use cached data (age: {age_hours:.1f} hours)")
            else:
                logger.info(f"  - Fetch data from {market} market for {symbol}")

            logger.info("  - Analyze financial data")

            if not args.skip_valuation:
                logger.info("  - Compute valuation metrics")

            if not args.skip_analyst:
                logger.info("  - Extract analyst recommendations")

            if not args.skip_charts:
                logger.info(f"  - Generate {len(CHART_FILES)} charts")

            if not args.skip_report:
                logger.info("  - Generate markdown report")

            logger.info("No files will be modified in dry-run mode.")
            logger.info("Run without --dry-run to execute these operations.")
            return

        # Calculate total steps for progress tracking
        total_steps = 1  # fetch/use cache
        total_steps += 1  # analyze
        if not args.skip_valuation:
            total_steps += 1
        if not args.skip_analyst:
            total_steps += 1
        if not args.skip_charts:
            total_steps += 1
        if not args.skip_report:
            total_steps += 1

        logger.info(f"Starting report pipeline for {symbol} ({market})")

        with step_progress(total_steps) as sp:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Fetch or use cached data
            use_cache = not args.refresh and is_fresh(data_path, args.max_age_hours)

            if use_cache:
                with sp.step(f"Loading cached data for {symbol}"):
                    data_payload = read_json(data_path)
                    fetched_at = parse_iso_datetime(data_payload.get("fetched_at"))
                    if fetched_at:
                        age_hours = hours_since(fetched_at)
                        logger.info(
                            f"Using cache (fetched {age_hours:.1f} hours ago): {data_path}"
                        )
                    else:
                        logger.info(f"Using cached data: {data_path}")
            else:
                with sp.step(f"Fetching {symbol} from {market} market"):
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
                    logger.info(f"Saved to: {data_path}")

            data_mtime = data_path.stat().st_mtime

            # Step 2: Analyze financial data
            with sp.step("Analyzing financial metrics"):
                if needs_update(analysis_path, [data_mtime]):
                    analysis_payload = analyze_module.build_analysis(data_payload)
                    write_json(analysis_path, analysis_payload)
                    logger.info(f"Saved to: {analysis_path}")
                else:
                    analysis_payload = read_json(analysis_path)
                    logger.info(f"Using cache: {analysis_path}")

            analysis_mtime = analysis_path.stat().st_mtime

            # Step 3: Compute valuation
            valuation_payload: dict[str, Any] = {}
            if not args.skip_valuation:
                with sp.step("Computing valuation metrics"):
                    if needs_update(valuation_path, [data_mtime, analysis_mtime]):
                        valuation_payload = valuation_module.build_valuation(
                            data_payload, analysis_payload
                        )
                        write_json(valuation_path, valuation_payload)
                        logger.info(f"Saved to: {valuation_path}")
                    else:
                        valuation_payload = read_json(valuation_path)
                        logger.info(f"Using cache: {valuation_path}")

            # Step 4: Extract analyst data
            analyst_payload: dict[str, Any] = {}
            if not args.skip_analyst:
                with sp.step("Extracting analyst recommendations"):
                    if needs_update(analyst_path, [data_mtime]):
                        analyst_payload = analyst_module.build_analyst_report(
                            data_payload
                        )
                        write_json(analyst_path, analyst_payload)
                        logger.info(f"Saved to: {analyst_path}")
                    else:
                        analyst_payload = read_json(analyst_path)
                        logger.info(f"Using cache: {analyst_path}")

            # Step 5: Generate charts
            if not args.skip_charts:
                with sp.step("Generating charts"):
                    chart_inputs = [analysis_mtime]
                    if not args.skip_valuation and valuation_path.exists():
                        chart_inputs.append(valuation_path.stat().st_mtime)
                    if charts_need_update(charts_dir, chart_inputs):
                        visualize_module.generate_charts(
                            analysis_payload, str(charts_dir), valuation_payload
                        )
                        logger.info(f"Saved to: {charts_dir}")
                    else:
                        logger.info(f"Using cache: {charts_dir}")

            # Step 6: Generate report
            if not args.skip_report:
                with sp.step("Generating markdown report"):
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
                        logger.info(f"Saved to: {report_path}")
                    else:
                        logger.info(f"Using cache: {report_path}")

        # Print summary
        logger.info(f"Report generation complete for {symbol}")
        company_name = analysis_payload.get("company", {}).get("name", symbol)
        logger.info(f"Company: {company_name}")
        industry = analysis_payload.get("company", {}).get("industry")
        if industry:
            logger.info(f"Industry: {industry}")

        latest_price = analysis_payload.get("price", {}).get("latest")
        currency = analysis_payload.get("company", {}).get("currency")
        if latest_price and currency:
            logger.info(f"Latest Price: {latest_price:.2f} {currency}")

        # Show data quality warnings
        dq = analysis_payload.get("data_quality", {})
        validation = dq.get("validation", {})
        field_matching = dq.get("field_matching", {})

        if (
            validation.get("failed", 0) > 0
            or field_matching.get("fuzzy_matches", 0) > 0
        ):
            logger.warning("Data Quality Warnings:")
            if validation.get("failed", 0) > 0:
                logger.warning(f"  {validation['failed']} validation checks failed")
            if field_matching.get("fuzzy_matches", 0) > 0:
                logger.warning(
                    f"  {field_matching['fuzzy_matches']} fields matched using fuzzy matching"
                )

        logger.info(f"Report saved to: {report_path}")

    except FinancialReportError as e:
        logger.error(format_error_for_user(e))
        exit(1)
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error in report pipeline: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()

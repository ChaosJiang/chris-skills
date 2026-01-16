#!/usr/bin/env python3
"""Generate normalized financial analysis from fetched data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple, List

import polars as pl

from series_utils import (
    empty_series,
    latest_value,
    rows_from_payload,
    series_from_mapping,
    series_from_rows,
    series_rows,
    series_to_dict,
)


ROW_MAP = {
    "revenue": ["Total Revenue", "Revenue", "营业总收入", "营业收入"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
        "Diluted NI Availto Com Stockholders",
        "净利润",
    ],
    "gross_profit": ["Gross Profit", "毛利润", "营业毛利"],
    "operating_income": [
        "Operating Income",
        "Total Operating Income As Reported",
        "营业利润",
        "营业收益",
    ],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "total_assets": ["Total Assets", "资产总计"],
    "total_liabilities": [
        "Total Liabilities",
        "Total Liabilities Net Minority Interest",
        "负债合计",
    ],
    "total_equity": [
        "Total Equity",
        "Total Stockholder Equity",
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
        "所有者权益合计",
    ],
    "free_cash_flow": ["Free Cash Flow", "自由现金流"],
    "diluted_avg_shares": ["Diluted Average Shares"],
    "basic_avg_shares": ["Basic Average Shares"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    "total_debt": ["Total Debt"],
    "net_debt": ["Net Debt"],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
    ],
}


def normalize_label(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def find_matching_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {normalize_label(str(key)): str(key) for key in keys}
    for candidate in candidates:
        normalized = normalize_label(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return None


def find_matching_row_key(
    statement: Dict[str, Any], candidates: Iterable[str]
) -> Optional[str]:
    lookup: Dict[str, str] = {}
    for row_map in statement.values():
        if not isinstance(row_map, dict):
            continue
        for key in row_map.keys():
            normalized = normalize_label(str(key))
            lookup.setdefault(normalized, str(key))
    for candidate in candidates:
        normalized = normalize_label(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return None


def extract_row(
    statement: Dict[str, Dict[str, Any]], candidates: Iterable[str]
) -> pl.DataFrame:
    if not statement:
        return empty_series()
    if "报告日期" in statement:
        metric_key = find_matching_key(
            [key for key in statement.keys() if key != "报告日期"], candidates
        )
        if not metric_key:
            return empty_series()
        rows = rows_from_payload(statement, "报告日期")
        return series_from_rows(rows, "报告日期", metric_key)

    metric_key = find_matching_row_key(statement, candidates)
    if not metric_key:
        return empty_series()
    mapping = {}
    for date_key, row_map in statement.items():
        if isinstance(row_map, dict):
            mapping[str(date_key)] = row_map.get(metric_key)
    return series_from_mapping(mapping)


def series_values(series: pl.DataFrame) -> Tuple[List[datetime], List[float]]:
    rows = series_rows(series)
    dates = [row[0] for row in rows]
    values = [row[1] for row in rows]
    return dates, values


def compute_yoy(series: pl.DataFrame) -> Dict[str, Any]:
    dates, values = series_values(series)
    if len(values) < 2:
        return {}
    result: Dict[str, Any] = {}
    for idx in range(1, len(values)):
        previous = values[idx - 1]
        if previous == 0:
            continue
        result[dates[idx].date().isoformat()] = float(values[idx] / previous - 1)
    return result


def compute_cagr(series: pl.DataFrame) -> Optional[float]:
    dates, values = series_values(series)
    if len(values) < 2:
        return None
    start = float(values[0])
    end = float(values[-1])
    if start == 0:
        return None
    years = max(len(values) - 1, 1)
    return float((end / start) ** (1 / years) - 1)


def compute_ttm_sum(series: pl.DataFrame) -> pl.DataFrame:
    dates, values = series_values(series)
    if len(values) < 4:
        return empty_series()
    rows = []
    for idx in range(3, len(values)):
        rows.append((dates[idx], sum(values[idx - 3 : idx + 1])))
    return pl.DataFrame(rows, schema=["date", "value"], orient="row")


def align_on_date(
    left: pl.DataFrame, right: pl.DataFrame, left_name: str, right_name: str
) -> pl.DataFrame:
    if left.height == 0 or right.height == 0:
        return pl.DataFrame()
    left_df = left.rename({"value": left_name})
    right_df = right.rename({"value": right_name})
    return left_df.join(right_df, on="date", how="inner")


def divide_series(numerator: pl.DataFrame, denominator: pl.DataFrame) -> pl.DataFrame:
    aligned = align_on_date(numerator, denominator, "num", "den")
    if aligned.height == 0:
        return empty_series()
    aligned = aligned.filter(pl.col("den") != 0)
    if aligned.height == 0:
        return empty_series()
    result = aligned.with_columns((pl.col("num") / pl.col("den")).alias("value"))
    return result.select(["date", "value"]).filter(pl.col("value").is_finite())


def compute_per_share(numerator: pl.DataFrame, denominator: pl.DataFrame) -> pl.DataFrame:
    return divide_series(numerator, denominator)


def compute_average_balance(series: pl.DataFrame) -> pl.DataFrame:
    dates, values = series_values(series)
    if len(values) < 2:
        return empty_series()
    rows = []
    for idx in range(1, len(values)):
        rows.append((dates[idx], (values[idx - 1] + values[idx]) / 2))
    return pl.DataFrame(rows, schema=["date", "value"], orient="row")


def compute_ttm_ratio(
    numerator_ttm: pl.DataFrame, denominator_avg: pl.DataFrame
) -> pl.DataFrame:
    return divide_series(numerator_ttm, denominator_avg)


def extract_quarterly_metrics(
    income: Dict[str, Dict[str, Any]],
    balance: Dict[str, Dict[str, Any]],
    cashflow: Dict[str, Dict[str, Any]],
) -> Dict[str, pl.DataFrame]:
    return {
        "revenue": extract_row(income, ROW_MAP["revenue"]),
        "net_income": extract_row(income, ROW_MAP["net_income"]),
        "gross_profit": extract_row(income, ROW_MAP["gross_profit"]),
        "operating_income": extract_row(income, ROW_MAP["operating_income"]),
        "ebitda": extract_row(income, ROW_MAP["ebitda"]),
        "diluted_avg_shares": extract_row(income, ROW_MAP["diluted_avg_shares"]),
        "basic_avg_shares": extract_row(income, ROW_MAP["basic_avg_shares"]),
        "total_assets": extract_row(balance, ROW_MAP["total_assets"]),
        "total_liabilities": extract_row(balance, ROW_MAP["total_liabilities"]),
        "total_equity": extract_row(balance, ROW_MAP["total_equity"]),
        "shares_outstanding": extract_row(balance, ROW_MAP["shares_outstanding"]),
        "total_debt": extract_row(balance, ROW_MAP["total_debt"]),
        "net_debt": extract_row(balance, ROW_MAP["net_debt"]),
        "cash": extract_row(balance, ROW_MAP["cash"]),
        "free_cash_flow": extract_row(cashflow, ROW_MAP["free_cash_flow"]),
    }


def extract_metrics(
    income: Dict[str, Dict[str, Any]],
    balance: Dict[str, Dict[str, Any]],
    cashflow: Dict[str, Dict[str, Any]],
) -> Dict[str, pl.DataFrame]:
    return {
        "revenue": extract_row(income, ROW_MAP["revenue"]),
        "net_income": extract_row(income, ROW_MAP["net_income"]),
        "gross_profit": extract_row(income, ROW_MAP["gross_profit"]),
        "operating_income": extract_row(income, ROW_MAP["operating_income"]),
        "ebitda": extract_row(income, ROW_MAP["ebitda"]),
        "total_assets": extract_row(balance, ROW_MAP["total_assets"]),
        "total_liabilities": extract_row(balance, ROW_MAP["total_liabilities"]),
        "total_equity": extract_row(balance, ROW_MAP["total_equity"]),
        "free_cash_flow": extract_row(cashflow, ROW_MAP["free_cash_flow"]),
    }


def compute_ratios(
    metrics: Dict[str, pl.DataFrame],
) -> Dict[str, Dict[str, Any]]:
    revenue = metrics["revenue"]
    net_income = metrics["net_income"]
    gross_profit = metrics["gross_profit"]
    total_assets = metrics["total_assets"]
    total_equity = metrics["total_equity"]
    total_liabilities = metrics["total_liabilities"]

    ratios = {
        "gross_margin": series_to_dict(divide_series(gross_profit, revenue)),
        "net_margin": series_to_dict(divide_series(net_income, revenue)),
        "roe": series_to_dict(divide_series(net_income, total_equity)),
        "roa": series_to_dict(divide_series(net_income, total_assets)),
        "debt_to_equity": series_to_dict(divide_series(total_liabilities, total_equity)),
    }
    return ratios


def extract_price_series(price_payload: Dict[str, Dict[str, Any]]) -> pl.DataFrame:
    if not price_payload:
        return empty_series()
    date_key = next(
        (key for key in ["日期", "date", "Date"] if key in price_payload), None
    )
    candidates = ["Close", "Adj Close", "收盘", "close", "close_price"]
    if date_key:
        value_key = find_matching_key(
            [key for key in price_payload.keys() if key != date_key], candidates
        )
        if not value_key:
            return empty_series()
        rows = rows_from_payload(price_payload, date_key)
        return series_from_rows(rows, date_key, value_key)

    for candidate in candidates:
        column_map = price_payload.get(candidate)
        if isinstance(column_map, dict):
            return series_from_mapping(column_map)
    return empty_series()


def build_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    info = payload.get("info", {}) or {}

    income = payload.get("financials", {}).get("income_statement", {}) or {}
    balance = payload.get("financials", {}).get("balance_sheet", {}) or {}
    cashflow = payload.get("financials", {}).get("cashflow", {}) or {}

    quarterly_income = (
        payload.get("financials_quarterly", {}).get("income_statement", {}) or {}
    )
    quarterly_balance = (
        payload.get("financials_quarterly", {}).get("balance_sheet", {}) or {}
    )
    quarterly_cashflow = (
        payload.get("financials_quarterly", {}).get("cashflow", {}) or {}
    )

    price_payload = payload.get("price_history", {}) or {}

    metrics = extract_metrics(income, balance, cashflow)
    quarterly_metrics = extract_quarterly_metrics(
        quarterly_income, quarterly_balance, quarterly_cashflow
    )

    price_series = extract_price_series(price_payload)

    revenue_q = quarterly_metrics["revenue"]
    net_income_q = quarterly_metrics["net_income"]
    ebitda_q = quarterly_metrics["ebitda"]
    free_cash_flow_q = quarterly_metrics["free_cash_flow"]
    diluted_avg_shares_q = quarterly_metrics["diluted_avg_shares"]

    eps_q = compute_per_share(net_income_q, diluted_avg_shares_q)
    eps_ttm = compute_ttm_sum(eps_q)

    sales_per_share_q = compute_per_share(revenue_q, diluted_avg_shares_q)
    sales_per_share_ttm = compute_ttm_sum(sales_per_share_q)

    ebitda_per_share_q = compute_per_share(ebitda_q, diluted_avg_shares_q)
    ebitda_per_share_ttm = compute_ttm_sum(ebitda_per_share_q)

    fcf_per_share_q = compute_per_share(free_cash_flow_q, diluted_avg_shares_q)
    fcf_per_share_ttm = compute_ttm_sum(fcf_per_share_q)

    revenue_ttm = compute_ttm_sum(revenue_q)
    net_income_ttm = compute_ttm_sum(net_income_q)
    ebitda_ttm = compute_ttm_sum(ebitda_q)
    free_cash_flow_ttm = compute_ttm_sum(free_cash_flow_q)

    total_equity_q = quarterly_metrics["total_equity"]
    total_assets_q = quarterly_metrics["total_assets"]
    net_debt_q = quarterly_metrics["net_debt"]
    total_debt_q = quarterly_metrics["total_debt"]
    cash_q = quarterly_metrics["cash"]
    shares_outstanding_q = quarterly_metrics["shares_outstanding"]

    book_per_share_q = compute_per_share(total_equity_q, shares_outstanding_q)
    net_debt_per_share_q = compute_per_share(net_debt_q, shares_outstanding_q)

    equity_avg_q = compute_average_balance(total_equity_q)
    assets_avg_q = compute_average_balance(total_assets_q)
    roe_ttm = compute_ttm_ratio(net_income_ttm, equity_avg_q)
    roa_ttm = compute_ttm_ratio(net_income_ttm, assets_avg_q)

    analysis = {
        "symbol": payload.get("symbol"),
        "market": payload.get("market"),
        "data_fetched_at": payload.get("fetched_at"),
        "company": {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currency": info.get("currency"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "financials": {key: series_to_dict(value) for key, value in metrics.items()},
        "financials_quarterly": {
            "revenue": series_to_dict(revenue_q),
            "net_income": series_to_dict(net_income_q),
            "ebitda": series_to_dict(ebitda_q),
            "free_cash_flow": series_to_dict(free_cash_flow_q),
            "diluted_avg_shares": series_to_dict(diluted_avg_shares_q),
        },
        "financials_ttm": {
            "revenue": series_to_dict(revenue_ttm),
            "net_income": series_to_dict(net_income_ttm),
            "ebitda": series_to_dict(ebitda_ttm),
            "free_cash_flow": series_to_dict(free_cash_flow_ttm),
        },
        "per_share_quarterly": {
            "eps": series_to_dict(eps_q),
            "sales": series_to_dict(sales_per_share_q),
            "ebitda": series_to_dict(ebitda_per_share_q),
            "free_cash_flow": series_to_dict(fcf_per_share_q),
        },
        "per_share_ttm": {
            "eps": series_to_dict(eps_ttm),
            "sales": series_to_dict(sales_per_share_ttm),
            "ebitda": series_to_dict(ebitda_per_share_ttm),
            "free_cash_flow": series_to_dict(fcf_per_share_ttm),
        },
        "balance_quarterly": {
            "total_equity": series_to_dict(total_equity_q),
            "shares_outstanding": series_to_dict(shares_outstanding_q),
            "book_per_share": series_to_dict(book_per_share_q),
            "net_debt": series_to_dict(net_debt_q),
            "total_debt": series_to_dict(total_debt_q),
            "cash": series_to_dict(cash_q),
            "net_debt_per_share": series_to_dict(net_debt_per_share_q),
        },
        "ratios": compute_ratios(metrics),
        "ratios_ttm": {
            "roe": series_to_dict(roe_ttm),
            "roa": series_to_dict(roa_ttm),
        },
        "growth": {
            "revenue_yoy": compute_yoy(metrics["revenue"]),
            "net_income_yoy": compute_yoy(metrics["net_income"]),
            "revenue_cagr": compute_cagr(metrics["revenue"]),
            "net_income_cagr": compute_cagr(metrics["net_income"]),
        },
        "price": {
            "history": series_to_dict(price_series),
            "latest": latest_value(price_series),
        },
    }

    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fetched financial data")
    parser.add_argument("--input", required=True, help="Path to data JSON")
    parser.add_argument("--output", default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    analysis = build_analysis(payload)
    output_path = f"{args.output}/{analysis['symbol'].replace('.', '_')}_analysis.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)

    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()

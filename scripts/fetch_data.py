#!/usr/bin/env python3
"""Fetch multi-market financial reports and price data."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np
import yfinance as yf
import akshare as ak


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def infer_market(symbol: str) -> str:
    upper_symbol = symbol.upper()
    if upper_symbol.endswith((".SH", ".SZ", ".BJ")):
        return "CN"
    if upper_symbol.endswith(".HK"):
        return "HK"
    if upper_symbol.endswith(".T"):
        return "JP"
    return "US"


def df_to_dict(df: Any | None) -> Dict[str, Dict[str, Any]]:
    if df is None or getattr(df, "empty", False):
        return {}
    try:
        sanitized = df.copy()
    except Exception:
        return {}
    try:
        if hasattr(sanitized, "columns"):
            sanitized.columns = [str(col) for col in sanitized.columns]
        if hasattr(sanitized, "index"):
            sanitized.index = [str(idx) for idx in sanitized.index]
        if hasattr(sanitized, "replace"):
            sanitized = sanitized.replace({np.nan: None})
    except Exception:
        pass
    if hasattr(sanitized, "to_dict"):
        try:
            return sanitized.to_dict()
        except Exception:
            return {}
    return {}


def get_ticker_info(ticker: yf.Ticker) -> Dict[str, Any]:
    try:
        return ticker.get_info()
    except Exception:
        try:
            return ticker.info
        except Exception:
            return {}


def get_income_statement(ticker: yf.Ticker) -> Any:
    if hasattr(ticker, "income_stmt"):
        return ticker.income_stmt
    if hasattr(ticker, "financials"):
        return ticker.financials
    return {}


def get_balance_sheet(ticker: yf.Ticker) -> Any:
    if hasattr(ticker, "balance_sheet"):
        return ticker.balance_sheet
    if hasattr(ticker, "balancesheet"):
        return ticker.balancesheet
    return {}


def get_cashflow(ticker: yf.Ticker) -> Any:
    if hasattr(ticker, "cashflow"):
        return ticker.cashflow
    if hasattr(ticker, "cash_flow"):
        return ticker.cash_flow
    return {}


def get_quarterly_income_statement(ticker: yf.Ticker) -> Any:
    if hasattr(ticker, "quarterly_income_stmt"):
        return ticker.quarterly_income_stmt
    if hasattr(ticker, "quarterly_incomestmt"):
        return ticker.quarterly_incomestmt
    if hasattr(ticker, "quarterly_financials"):
        return ticker.quarterly_financials
    return {}


def get_quarterly_balance_sheet(ticker: yf.Ticker) -> Any:
    if hasattr(ticker, "quarterly_balance_sheet"):
        return ticker.quarterly_balance_sheet
    if hasattr(ticker, "quarterly_balancesheet"):
        return ticker.quarterly_balancesheet
    return {}


def get_quarterly_cashflow(ticker: yf.Ticker) -> Any:
    if hasattr(ticker, "quarterly_cash_flow"):
        return ticker.quarterly_cash_flow
    if hasattr(ticker, "quarterly_cashflow"):
        return ticker.quarterly_cashflow
    return {}


def fetch_yfinance(symbol: str, years: int, price_years: int) -> Dict[str, Any]:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=f"{price_years}y", auto_adjust=False)
    recommendations = getattr(ticker, "recommendations", None)
    recommendations_summary = getattr(ticker, "recommendations_summary", None)
    analyst_price_target = getattr(ticker, "analyst_price_target", None)

    return {
        "info": get_ticker_info(ticker),
        "financials": {
            "income_statement": df_to_dict(get_income_statement(ticker)),
            "balance_sheet": df_to_dict(get_balance_sheet(ticker)),
            "cashflow": df_to_dict(get_cashflow(ticker)),
        },
        "financials_quarterly": {
            "income_statement": df_to_dict(get_quarterly_income_statement(ticker)),
            "balance_sheet": df_to_dict(get_quarterly_balance_sheet(ticker)),
            "cashflow": df_to_dict(get_quarterly_cashflow(ticker)),
        },
        "price_history": df_to_dict(history),
        "analyst": {
            "recommendations": df_to_dict(recommendations),
            "recommendations_summary": df_to_dict(recommendations_summary),
            "price_target": df_to_dict(analyst_price_target),
        },
    }


def fetch_cn(symbol: str, years: int) -> Dict[str, Any]:
    code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")

    income = ak.stock_financial_report_sina(stock=code, symbol="利润表")
    balance = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
    cashflow = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
    history = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start_date, end_date=end_date
    )

    return {
        "info": {},
        "financials": {
            "income_statement": df_to_dict(income),
            "balance_sheet": df_to_dict(balance),
            "cashflow": df_to_dict(cashflow),
        },
        "price_history": df_to_dict(history),
        "analyst": {},
    }


def fetch_data(
    symbol: str, market: str, years: int, price_years: int
) -> Dict[str, Any]:
    if market in {"US", "HK", "JP"}:
        return fetch_yfinance(symbol, years, price_years)
    if market == "CN":
        return fetch_cn(symbol, years)
    raise ValueError(f"Unsupported market: {market}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch financial data for analysis")
    parser.add_argument("--symbol", required=True, help="Stock symbol")
    parser.add_argument("--market", choices=["US", "CN", "HK", "JP"])
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument(
        "--price-years",
        type=int,
        default=None,
        help="Years of price history to fetch (defaults to max(--years, 6) for yfinance markets)",
    )
    parser.add_argument("--output", default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = normalize_symbol(args.symbol)
    market = (args.market or infer_market(symbol)).upper()

    price_years = args.price_years
    if price_years is None:
        price_years = max(args.years, 6)

    payload = fetch_data(symbol, market, args.years, price_years)
    payload.update(
        {
            "symbol": symbol,
            "market": market,
            "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{symbol.replace('.', '_')}_data.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

    print(f"Saved data to {output_path}")


if __name__ == "__main__":
    main()

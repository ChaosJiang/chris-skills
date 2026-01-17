#!/usr/bin/env python3
"""Fetch multi-market financial reports and price data."""

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import akshare as ak
import numpy as np
import yfinance as yf
from exceptions import APIError, DataFetchError, SymbolNotFoundError
from logging_config import get_module_logger

logger = get_module_logger()


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


def df_to_dict(df: Any | None) -> dict[str, dict[str, Any]]:
    """Convert DataFrame to dict, handling various edge cases."""
    if df is None or getattr(df, "empty", False):
        return {}

    try:
        sanitized = df.copy()
    except AttributeError as e:
        logger.error(f"Failed to copy dataframe: {e}")
        raise DataFetchError("Invalid dataframe object") from e
    except Exception as e:
        logger.error(f"Unexpected error copying dataframe: {e}", exc_info=True)
        raise DataFetchError("Failed to process dataframe") from e

    try:
        if hasattr(sanitized, "columns"):
            sanitized.columns = [str(col) for col in sanitized.columns]
        if hasattr(sanitized, "index"):
            sanitized.index = [str(idx) for idx in sanitized.index]
        if hasattr(sanitized, "replace"):
            sanitized = sanitized.replace({np.nan: None})
    except Exception as e:
        logger.warning(f"Failed to sanitize dataframe fields: {e}")
        # Continue with unsanitized data

    if hasattr(sanitized, "to_dict"):
        try:
            return sanitized.to_dict()
        except Exception as e:
            logger.error(f"Failed to convert dataframe to dict: {e}", exc_info=True)
            raise DataFetchError("Failed to serialize dataframe") from e

    logger.warning("Object is not a dataframe, returning empty dict")
    return {}


def parse_period_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        raw = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def trim_statement_columns(df: Any, max_periods: int) -> Any:
    if df is None or max_periods <= 0:
        return df
    columns = getattr(df, "columns", None)
    if columns is None:
        return df
    column_list = list(columns)
    if not column_list:
        return df
    if len(column_list) <= max_periods:
        return df

    dated_columns = []
    for col in column_list:
        parsed = parse_period_date(col)
        if parsed is not None:
            dated_columns.append((parsed, col))

    if dated_columns:
        dated_columns.sort(key=lambda pair: pair[0], reverse=True)
        keep_set = {col for _, col in dated_columns[:max_periods]}
        keep = [col for col in column_list if col in keep_set]
    else:
        keep = column_list[:max_periods]

    try:
        return df[keep]
    except Exception as e:
        logger.warning(f"Failed to trim statement columns: {e}")
        return df


def trim_statement_rows(df: Any, max_periods: int) -> Any:
    if df is None or max_periods <= 0:
        return df
    columns = getattr(df, "columns", None)
    if columns is None:
        return df
    column_list = list(columns)
    if not column_list:
        return df
    date_column = next(
        (
            col
            for col in ["报告日期", "报表日期", "报告期", "date", "Date"]
            if col in column_list
        ),
        None,
    )
    if not date_column:
        return trim_statement_columns(df, max_periods)

    try:
        values = list(df[date_column])
    except Exception as e:
        logger.warning(f"Failed to access statement date column: {e}")
        return df

    dated_rows = []
    for idx, value in enumerate(values):
        parsed = parse_period_date(value)
        if parsed is not None:
            dated_rows.append((parsed, idx))

    if not dated_rows:
        return df

    dated_rows.sort(key=lambda pair: pair[0], reverse=True)
    keep_indices = sorted(idx for _, idx in dated_rows[:max_periods])

    try:
        return df.iloc[keep_indices]
    except Exception as e:
        logger.warning(f"Failed to trim statement rows: {e}")
        return df


def get_ticker_info(ticker: yf.Ticker) -> dict[str, Any]:
    """Get ticker info, trying multiple access methods."""
    # Try get_info() first (newer API)
    try:
        return ticker.get_info()
    except AttributeError:
        # Method doesn't exist, try direct attribute access
        pass
    except Exception as e:
        logger.warning(f"Failed to get ticker info via get_info(): {e}")

    # Try direct info attribute (older API)
    try:
        return ticker.info
    except AttributeError:
        logger.error("Ticker object has no info attribute or get_info() method")
        raise DataFetchError("Unable to retrieve ticker information") from None
    except Exception as e:
        logger.error(f"Failed to get ticker info: {e}", exc_info=True)
        raise DataFetchError("Failed to retrieve ticker information") from e


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


def fetch_yfinance(symbol: str, years: int, price_years: int) -> dict[str, Any]:
    """Fetch data from Yahoo Finance."""
    logger.info(f"Fetching yfinance data for {symbol}")

    try:
        ticker = yf.Ticker(symbol)
    except Exception as e:
        logger.error(f"Failed to create yfinance Ticker for {symbol}: {e}")
        raise DataFetchError(f"Failed to initialize ticker: {symbol}") from e

    # Get ticker info first to validate symbol exists
    try:
        info = get_ticker_info(ticker)
        if not info:
            raise SymbolNotFoundError(symbol)
    except DataFetchError:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting ticker info for {symbol}: {e}", exc_info=True)
        raise DataFetchError(f"Failed to retrieve info for {symbol}") from e

    # Fetch price history
    try:
        history = ticker.history(period=f"{price_years}y", auto_adjust=False)
        if history.empty:
            logger.warning(f"No price history found for {symbol}")
    except Exception as e:
        logger.error(f"Failed to fetch price history for {symbol}: {e}")
        raise DataFetchError("Failed to fetch price history") from e

    # Get analyst data (optional, don't fail if missing)
    recommendations = getattr(ticker, "recommendations", None)
    recommendations_summary = getattr(ticker, "recommendations_summary", None)
    analyst_price_target = getattr(ticker, "analyst_price_target", None)

    return {
        "info": info,
        "financials": {
            "income_statement": df_to_dict(
                trim_statement_columns(get_income_statement(ticker), years)
            ),
            "balance_sheet": df_to_dict(
                trim_statement_columns(get_balance_sheet(ticker), years)
            ),
            "cashflow": df_to_dict(trim_statement_columns(get_cashflow(ticker), years)),
        },
        "financials_quarterly": {
            "income_statement": df_to_dict(
                trim_statement_columns(
                    get_quarterly_income_statement(ticker), years * 4
                )
            ),
            "balance_sheet": df_to_dict(
                trim_statement_columns(get_quarterly_balance_sheet(ticker), years * 4)
            ),
            "cashflow": df_to_dict(
                trim_statement_columns(get_quarterly_cashflow(ticker), years * 4)
            ),
        },
        "price_history": df_to_dict(history),
        "analyst": {
            "recommendations": df_to_dict(recommendations),
            "recommendations_summary": df_to_dict(recommendations_summary),
            "price_target": df_to_dict(analyst_price_target),
        },
    }


def fetch_cn(symbol: str, years: int) -> dict[str, Any]:
    """Fetch data for Chinese A-share stocks."""
    code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    logger.info(f"Fetching CN market data for {code}")

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")

    # Fetch financial statements
    try:
        income = trim_statement_rows(
            ak.stock_financial_report_sina(stock=code, symbol="利润表"), years
        )
    except Exception as e:
        logger.error(f"Failed to fetch income statement for {code}: {e}")
        raise DataFetchError(f"Failed to fetch income statement for {symbol}") from e

    try:
        balance = trim_statement_rows(
            ak.stock_financial_report_sina(stock=code, symbol="资产负债表"), years
        )
    except Exception as e:
        logger.error(f"Failed to fetch balance sheet for {code}: {e}")
        raise DataFetchError(f"Failed to fetch balance sheet for {symbol}") from e

    try:
        cashflow = trim_statement_rows(
            ak.stock_financial_report_sina(stock=code, symbol="现金流量表"), years
        )
    except Exception as e:
        logger.error(f"Failed to fetch cashflow for {code}: {e}")
        raise DataFetchError(f"Failed to fetch cashflow for {symbol}") from e

    # Fetch price history
    try:
        history = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start_date, end_date=end_date
        )
        if history.empty:
            logger.warning(f"No price history found for {code}")
            raise SymbolNotFoundError(symbol, market="CN")
    except SymbolNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch price history for {code}: {e}", exc_info=True)
        raise DataFetchError(f"Failed to fetch price history for {symbol}") from e

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
) -> dict[str, Any]:
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
    from exceptions import format_error_for_user
    from logging_config import setup_logging

    # Set up logging
    _, _ = setup_logging(log_level="INFO", log_to_file=True)

    args = parse_args()
    symbol = normalize_symbol(args.symbol)
    market = (args.market or infer_market(symbol)).upper()

    logger.info(f"Fetching data for {symbol} (Market: {market})")

    price_years = args.price_years
    if price_years is None:
        price_years = max(args.years, 6)

    try:
        payload = fetch_data(symbol, market, args.years, price_years)
        payload.update(
            {
                "symbol": symbol,
                "market": market,
                "fetched_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )

        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"{symbol.replace('.', '_')}_data.json")

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Successfully saved data to {output_path}")

    except (DataFetchError, SymbolNotFoundError, APIError) as e:
        logger.error(format_error_for_user(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
